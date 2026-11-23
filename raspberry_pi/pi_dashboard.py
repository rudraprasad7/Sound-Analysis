#!/usr/bin/env python3
"""Raspberry Pi 3: Sensor fetcher, 30s audio uploader to Render, and local display dashboard."""
import os
import time
import uuid
import base64
import atexit
import logging
import threading
import subprocess
from datetime import datetime, timezone
from collections import deque

import requests
import board
import busio
import adafruit_bmp280
import adafruit_scd4x
import RPi.GPIO as GPIO
from flask import Flask, jsonify, request, Response, render_template_string

# ---------- EDIT THESE SETTINGS ----------
DEVICE_ID = "raspberrypi3-01"
RENDER_API_URL = "https://your-service-name.onrender.com/analyze"  # Replace with your live Render URL
RENDER_TIMEOUT_SEC = 90             # Allows for Render free-tier cold start

SENSOR_INTERVAL = 1.0               # Read sensors every 1 second
MIC = "plughw:2,0"                  # ALSA microphone device
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_WINDOW_SECONDS = 30
LED_PIN = 17                        # BCM 17 = physical pin 11
TEMP_WAV_PATH = "/tmp/current_window.wav"
# -----------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Hardware initialization: BMP280 at 0x76, SCD4X, GPIO17 LED
i2c = busio.I2C(board.SCL, board.SDA)
bmp = adafruit_bmp280.Adafruit_BMP280_I2C(i2c, address=0x76)
scd = adafruit_scd4x.SCD4X(i2c)
scd.start_periodic_measurement()
logging.info("Waiting 5s for SCD4X initial measurement...")
time.sleep(5)

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

app = Flask(__name__)
lock = threading.RLock()
led_lock = threading.Lock()
audio_lock = threading.Lock()
stop_event = threading.Event()
audio_loop_event = threading.Event()
cleanup_done = threading.Event()
audio_thread = None

sample_number = 0
last_scd = {"co2_ppm": None, "scd_temp": None, "humidity": None}
latest = {}
history = deque(maxlen=300)
status = {"state": "online", "audio_loop_active": False, "message": "Continuous sensor streaming active"}
acks = deque(maxlen=30)
latest_analysis = {}
latest_plot_bytes = None
audio_windows = deque(maxlen=50)


def flash_led(count=2, on_time=0.18, off_time=0.18):
    if cleanup_done.is_set():
        return
    with led_lock:
        for _ in range(count):
            if cleanup_done.is_set():
                break
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(on_time)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(off_time)


def set_status(state, **extra):
    global status
    with lock:
        status = {
            "device_id": DEVICE_ID,
            "state": state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **extra,
        }


def add_ack(command_id, state, message=""):
    with lock:
        acks.appendleft({
            "device_id": DEVICE_ID,
            "command_id": command_id,
            "status": state,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


def read_sensors_once():
    global sample_number
    sample_number += 1
    now = datetime.now(timezone.utc)
    row = {
        "device_id": DEVICE_ID,
        "sample": sample_number,
        "timestamp": now.isoformat(),
        "timestamp_local": now.astimezone().isoformat(timespec="seconds"),
        "epoch": time.time(),
        "pressure_hPa": round(float(bmp.pressure), 2),
        "bmp_temp": round(float(bmp.temperature), 2),
    }
    if scd.data_ready:
        last_scd.update({
            "co2_ppm": int(scd.CO2),
            "scd_temp": round(float(scd.temperature), 2),
            "humidity": round(float(scd.relative_humidity), 2),
        })
    row.update(last_scd)
    return row


def sensor_loop():
    """Continuously read BMP280 & SCD4X every second."""
    global latest
    next_tick = time.monotonic()
    while not stop_event.is_set():
        try:
            row = read_sensors_once()
            with lock:
                latest = row
                history.append(row)
            # TODO: Insert `row` into Supabase sensor table here
        except Exception:
            logging.exception("Sensor read error")
        next_tick += SENSOR_INTERVAL
        stop_event.wait(max(0.0, next_tick - time.monotonic()))


def average_sensors_in_window(start_epoch, end_epoch):
    """Average all 1-second sensor readings captured during a 30s audio window."""
    with lock:
        samples = [s for s in history if start_epoch <= s["epoch"] <= end_epoch]
    if not samples:
        return {}

    def avg(key):
        vals = [s[key] for s in samples if s.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "samples_in_window": len(samples),
        "pressure_hPa": avg("pressure_hPa"),
        "bmp_temp": avg("bmp_temp"),
        "co2_ppm": avg("co2_ppm"),
        "scd_temp": avg("scd_temp"),
        "humidity": avg("humidity"),
    }


def record_and_process_window(session_id):
    """Record 30s WAV to /tmp, POST to Render for GMM + dBFS preprocessing, update display state."""
    global latest_analysis, latest_plot_bytes
    if not audio_lock.acquire(blocking=False):
        return
    try:
        start_dt = datetime.now().astimezone()
        start_epoch = time.time()
        window_start = start_dt.isoformat(timespec="seconds")
        filename = f"audio_{start_dt.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.wav"

        set_status(
            "audio_recording",
            session_id=session_id,
            filename=filename,
            recorded_at=window_start,
            audio_loop_active=audio_loop_event.is_set(),
            message=f"Recording {AUDIO_WINDOW_SECONDS}s window ({filename})...",
        )

        cmd = [
            "arecord", "-D", MIC, "-f", "S16_LE",
            "-r", str(AUDIO_RATE), "-c", str(AUDIO_CHANNELS),
            "-t", "wav", "-d", str(AUDIO_WINDOW_SECONDS), TEMP_WAV_PATH,
        ]
        subprocess.run(cmd, check=True, timeout=AUDIO_WINDOW_SECONDS + 20)
        end_epoch = time.time()
        sensor_avg = average_sensors_in_window(start_epoch, end_epoch)

        set_status(
            "audio_uploading",
            session_id=session_id,
            filename=filename,
            recorded_at=window_start,
            audio_loop_active=audio_loop_event.is_set(),
            message=f"Sending {filename} to Render for GMM + dBFS analysis...",
        )

        with open(TEMP_WAV_PATH, "rb") as audio_fp:
            resp = requests.post(
                RENDER_API_URL,
                files={"audio": (filename, audio_fp, "audio/wav")},
                data={"window_start": window_start, "window_seconds": str(AUDIO_WINDOW_SECONDS)},
                timeout=RENDER_TIMEOUT_SEC,
            )
        resp.raise_for_status()
        result = resp.json()

        if result.get("status") != "ok":
            raise RuntimeError(result.get("message") or "Render analysis failed")

        plot_b64 = result.pop("plot_png_b64", None)
        plot_bytes = base64.b64decode(plot_b64) if plot_b64 else None
        result["sensor_reading_avg"] = sensor_avg
        result["has_plot"] = plot_bytes is not None

        if result.get("alert"):
            threading.Thread(target=flash_led, args=(5, 0.1, 0.1), daemon=True).start()

        with lock:
            latest_analysis = result
            if plot_bytes is not None:
                latest_plot_bytes = plot_bytes
            audio_windows.appendleft(result)

        set_status(
            "audio_received",
            session_id=session_id,
            filename=filename,
            recorded_at=window_start,
            audio_loop_active=audio_loop_event.is_set(),
            message=f"Analyzed {filename} · GMM: {result.get('gmm_cluster')} · dB: {result.get('db_cluster')}",
        )

        # TODO: Insert `result` into Supabase audio windows table here
        logging.info(
            "30s window complete [%s]: gmm=%s db=%s (%.2f dBFS) alert=%s",
            filename, result.get("gmm_cluster"), result.get("db_cluster"),
            result.get("avg_db", 0.0), result.get("alert"),
        )
    except Exception as exc:
        logging.exception("Audio capture or Render upload failed")
        set_status(
            "audio_error",
            session_id=session_id,
            audio_loop_active=audio_loop_event.is_set(),
            message=f"Error: {exc}",
        )
        if audio_loop_event.is_set():
            stop_event.wait(5.0)
    finally:
        if os.path.exists(TEMP_WAV_PATH):
            try:
                os.remove(TEMP_WAV_PATH)
            except OSError:
                pass
        audio_lock.release()


def continuous_audio_loop(session_id):
    set_status(
        "audio_loop_active",
        session_id=session_id,
        audio_loop_active=True,
        message=f"Recording {AUDIO_WINDOW_SECONDS}-second windows continuously",
    )
    while audio_loop_event.is_set() and not stop_event.is_set():
        record_and_process_window(session_id)
    set_status(
        "audio_loop_stopped",
        session_id=session_id,
        audio_loop_active=False,
        message="Audio recording is OFF",
    )


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Sensor Dashboard</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#e8eef9;font:15px Arial,sans-serif}main{max-width:1250px;margin:auto;padding:24px}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}h1{font-size:25px;margin:0}.muted{color:#9aabc4}.pill{padding:8px 12px;border-radius:20px;background:#26364d}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:13px;margin:22px 0}.card,.panel{background:#152238;border:1px solid #263954;border-radius:14px;padding:17px}.label{color:#9aabc4;font-size:13px}.value{font-size:27px;font-weight:700;margin-top:9px}.unit{font-size:13px;color:#9aabc4}
.panel{margin:15px 0}button{background:#4d8cff;color:white;border:0;border-radius:9px;padding:11px 16px;font-weight:700;cursor:pointer}input{background:#0c1728;color:white;border:1px solid #344965;border-radius:8px;padding:10px;width:140px}.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:9px;border-bottom:1px solid #2b3d56;white-space:nowrap}th{color:#9aabc4}a{color:#8db8ff}.led{display:inline-block;width:11px;height:11px;border-radius:50%;background:#526078;margin-right:8px}
.badge{display:inline-block;padding:4px 10px;border-radius:12px;font-size:12px;font-weight:700}.badge-normal{background:#194d33;color:#5ce69b;border:1px solid #2ecc71}.badge-alert{background:#5c1a1a;color:#ff8a8a;border:1px solid #e74c3c}.badge-wait{background:#26364d;color:#9aabc4}
.cgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin-top:12px}.cbox{background:#0f1a2d;border-radius:10px;padding:11px;border-left:4px solid #526078}
.c-opt{border-left-color:#2ecc71}.c-norm{border-left-color:#27ae60}.c-warn{border-left-color:#f39c12}.c-sev{border-left-color:#c0392b}
.c-quiet{border-left-color:#3498db}.c-soft{border-left-color:#2ecc71}.c-mod{border-left-color:#f39c12}.c-loud{border-left-color:#e74c3c}
.plot-img{width:100%;border-radius:10px;margin-top:14px;border:1px solid #2b3d56;background:#fff}
</style></head><body><main><header><div><h1>Raspberry Pi Continuous Live Monitoring</h1><div class="muted">Pi 3 Fetch &amp; Display · Render GMM + dBFS · BMP280 · SCD4X · Audio</div></div><div class="pill"><span id="deviceLed" class="led"></span><span id="deviceState">Waiting for Pi</span></div></header>
<div class="grid">
<div class="card"><div class="label">Pressure</div><div class="value" id="pressure">--</div><span class="unit">hPa</span></div>
<div class="card"><div class="label">BMP280 Temperature</div><div class="value" id="bmpTemp">--</div><span class="unit">°C</span></div>
<div class="card"><div class="label">CO₂</div><div class="value" id="co2">--</div><span class="unit">ppm</span></div>
<div class="card"><div class="label">SCD4X Temperature</div><div class="value" id="scdTemp">--</div><span class="unit">°C</span></div>
<div class="card"><div class="label">Relative Humidity</div><div class="value" id="humidity">--</div><span class="unit">%</span></div>
<div class="card"><div class="label">Sample number</div><div class="value" id="sample">--</div><span class="unit" id="sampleTime">Waiting for samples</span></div></div>
<section class="panel"><h3>Signal / LED confirmation</h3><p class="muted">Test the local GPIO17 LED and command acknowledgement.</p><button onclick="sendCommand('ping')">Test signal / LED</button> <span id="pingResult" class="muted"></span></section>
<section class="panel"><h3>Continuous audio recording</h3><p class="muted">Turn ON to record consecutive 30-second WAV windows and send them to Render for GMM + decibel clustering. Turn OFF to finish the current window and stop the loop.</p><div class="row"><button id="audioToggle" onclick="toggleAudioLoop()">Start audio · OFF</button><span id="audioState" class="muted">Audio recording is OFF</span></div></section>
<section class="panel"><div class="row" style="justify-content:space-between"><div><h3 style="margin:0 0 4px">Acoustic clusters (GMM + Decibel)</h3><div id="analysisMeta" class="muted">Waiting for first 30-second Render analysis...</div></div><span id="analysisBadge" class="badge badge-wait">WAITING</span></div>
<div class="cgrid">
<div class="cbox c-opt"><div class="label">GMM Optimal (≥ -60)</div><div class="value" style="font-size:21px" id="cOpt">--</div><span class="unit">sec</span></div>
<div class="cbox c-norm"><div class="label">GMM Normal (-60..-85)</div><div class="value" style="font-size:21px" id="cNorm">--</div><span class="unit">sec</span></div>
<div class="cbox c-warn"><div class="label">GMM Warning (-85..-110)</div><div class="value" style="font-size:21px" id="cWarn">--</div><span class="unit">sec</span></div>
<div class="cbox c-sev"><div class="label">GMM Severe (&lt; -110)</div><div class="value" style="font-size:21px" id="cSev">--</div><span class="unit">sec</span></div>
</div>
<div class="cgrid">
<div class="cbox c-quiet"><div class="label">dB Quiet (&lt; -45 dBFS)</div><div class="value" style="font-size:21px" id="dQuiet">--</div><span class="unit">sec</span></div>
<div class="cbox c-soft"><div class="label">dB Soft (-45..-30)</div><div class="value" style="font-size:21px" id="dSoft">--</div><span class="unit">sec</span></div>
<div class="cbox c-mod"><div class="label">dB Moderate (-30..-18)</div><div class="value" style="font-size:21px" id="dMod">--</div><span class="unit">sec</span></div>
<div class="cbox c-loud"><div class="label">dB Loud (≥ -18 dBFS)</div><div class="value" style="font-size:21px" id="dLoud">--</div><span class="unit">sec</span></div>
</div>
<div id="plotBox"></div></section>
<section class="panel"><h3>Processed 30-second audio windows</h3><p class="muted">Each 30-second window processed by Render with its GMM cluster, decibel cluster, and window-averaged sensors.</p><div id="audioFiles" class="muted">No audio windows processed yet.</div></section>
<section class="panel"><h3>Recent live readings</h3><div style="overflow:auto"><table><thead><tr><th>Timestamp</th><th>Sample</th><th>Pressure hPa</th><th>BMP °C</th><th>CO₂ ppm</th><th>SCD °C</th><th>Humidity %</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section class="panel"><h3>Latest command acknowledgement</h3><div id="ack" class="muted">Waiting for acknowledgement</div></section>
</main><script>
let lastPlotFile='';
async function refresh(){try{const r=await fetch('/api/live');const d=await r.json();const x=d.latest||{};
for(const [id,key] of [['pressure','pressure_hPa'],['bmpTemp','bmp_temp'],['co2','co2_ppm'],['scdTemp','scd_temp'],['humidity','humidity']])document.getElementById(id).textContent=x[key]??'--';
document.getElementById('sample').textContent=x.sample??'--';document.getElementById('sampleTime').textContent=x.timestamp_local||x.timestamp||'Waiting for samples';
const state=d.status?.state||'waiting_for_device';const online=!!x.timestamp&&((Date.now()-new Date(x.timestamp).getTime())<15000);document.getElementById('deviceLed').style.background=online?'#35d07f':'#526078';document.getElementById('deviceState').textContent=online?'LIVE · '+state:state;
const audioOn=!!d.status?.audio_loop_active || ['audio_loop_active','audio_recording','audio_uploading','audio_loop_stopping'].includes(state);
const audioButton=document.getElementById('audioToggle');
audioButton.textContent=audioOn?'Stop audio · ON':'Start audio · OFF';
audioButton.style.background=audioOn?'#d9534f':'#23865b';
if(state.startsWith('audio_'))document.getElementById('audioState').textContent=d.status.message||state;
const a=d.latest_analysis||{};
if(a.class){
  const badge=document.getElementById('analysisBadge');
  badge.textContent=a.alert?`ANOMALY (${a.anomalous_seconds??0}s)`:'NORMAL BASELINE';
  badge.className='badge '+(a.alert?'badge-alert':'badge-normal');
  document.getElementById('analysisMeta').textContent=`${a.filename||''} · ${a.window_start||''} → ${a.window_end||''} · Avg: ${a.avg_db??'--'} dBFS · Max: ${a.max_db??'--'} dBFS`;
  const cd=a.cluster_distribution||{};
  document.getElementById('cOpt').textContent=cd.optimal_sec??'--';document.getElementById('cNorm').textContent=cd.normal_sec??'--';
  document.getElementById('cWarn').textContent=cd.warning_sec??'--';document.getElementById('cSev').textContent=cd.severe_sec??'--';
  const db=a.db_cluster_distribution||{};
  document.getElementById('dQuiet').textContent=db.quiet_sec??'--';document.getElementById('dSoft').textContent=db.soft_sec??'--';
  document.getElementById('dMod').textContent=db.moderate_sec??'--';document.getElementById('dLoud').textContent=db.loud_sec??'--';
  if(a.has_plot && a.filename!==lastPlotFile){
    lastPlotFile=a.filename;
    document.getElementById('plotBox').innerHTML=`<img class="plot-img" src="/api/plot.png?t=${Date.now()}" alt="GMM and Decibel Cluster Plot">`;
  }
}
document.getElementById('audioFiles').innerHTML=(d.audio_windows||[]).map(w=>{
  const sr=w.sensor_reading_avg||{};
  const tag=w.alert?`<span class="badge badge-alert">ANOMALY (${w.anomalous_seconds??0}s)</span>`:`<span class="badge badge-normal">NORMAL</span>`;
  return `<div style="padding:10px 0;border-bottom:1px solid #2b3d56"><div class="row" style="justify-content:space-between"><strong>${w.window_start||w.filename} → ${w.window_end||''}</strong>${tag}</div><div class="muted" style="font-size:12.5px;margin-top:4px">GMM: ${w.gmm_cluster||'--'} · dB: ${w.db_cluster||'--'} (${w.avg_db??'--'} dBFS) · Avg CO₂: ${sr.co2_ppm??'--'} ppm · Avg Temp: ${sr.scd_temp??sr.bmp_temp??'--'} °C</div></div>`;
}).join('')||'No audio windows processed yet.';
document.getElementById('rows').innerHTML=(d.history||[]).slice(-15).reverse().map(v=>`<tr><td>${v.timestamp_local||v.timestamp||''}</td><td>${v.sample??''}</td><td>${v.pressure_hPa??''}</td><td>${v.bmp_temp??''}</td><td>${v.co2_ppm??''}</td><td>${v.scd_temp??''}</td><td>${v.humidity??''}</td></tr>`).join('');
if(d.acks?.length){const ack=d.acks[0];document.getElementById('ack').textContent=ack.status+': '+(ack.message||'')+' · '+(ack.command_id||'');}
}catch(e){console.log(e)}}
async function sendCommand(action,extra={}){try{const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,...extra})});const d=await r.json();document.getElementById('pingResult').textContent=d.message||d.error||'Sent';}catch(e){document.getElementById('pingResult').textContent='Dashboard request failed';}}
async function toggleAudioLoop(){const d=await (await fetch('/api/live')).json();const active=!!d.status?.audio_loop_active||['audio_loop_active','audio_recording','audio_uploading','audio_loop_stopping'].includes(d.status?.state);const action=active?'stop_audio_loop':'start_audio_loop';document.getElementById('audioState').textContent=action==='start_audio_loop'?'Starting continuous 30s audio...':'Stopping after current 30-second window finishes...';sendCommand(action);}
setInterval(refresh,1000);refresh();</script></body></html>'''


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/live")
def api_live():
    with lock:
        return jsonify({
            "latest": latest,
            "history": list(history),
            "status": status,
            "acks": list(acks),
            "latest_analysis": latest_analysis,
            "audio_windows": list(audio_windows),
        })


@app.route("/api/plot.png")
def api_plot():
    with lock:
        data = latest_plot_bytes
    if not data:
        return ("No plot available yet", 404)
    return Response(data, mimetype="image/png", headers={"Cache-Control": "no-store"})


@app.route("/api/command", methods=["POST"])
def api_command():
    global audio_thread
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    command_id = uuid.uuid4().hex

    if action == "ping":
        threading.Thread(target=flash_led, daemon=True).start()
        add_ack(command_id, "received", "GPIO17 LED flashed")
        return jsonify({"message": "LED flashed", "command_id": command_id})

    if action == "start_audio_loop":
        if audio_loop_event.is_set():
            add_ack(command_id, "already_active", "Continuous audio recording is already ON")
            return jsonify({"message": "Already running", "command_id": command_id})
        audio_loop_event.set()
        audio_thread = threading.Thread(
            target=continuous_audio_loop,
            args=(command_id,),
            name="continuous-audio-loop",
            daemon=True,
        )
        audio_thread.start()
        threading.Thread(target=flash_led, daemon=True).start()
        add_ack(command_id, "started", f"Continuous {AUDIO_WINDOW_SECONDS}s audio monitoring started")
        return jsonify({"message": "Audio loop started", "command_id": command_id})

    if action == "stop_audio_loop":
        if not audio_loop_event.is_set():
            add_ack(command_id, "already_stopped", "Continuous audio recording is already OFF")
            return jsonify({"message": "Already stopped", "command_id": command_id})
        audio_loop_event.clear()
        set_status(
            "audio_loop_stopping",
            audio_loop_active=False,
            message="Finishing current 30-second window; no next recording will start",
        )
        add_ack(command_id, "stop_requested", "OFF received; finishing current 30-second window")
        return jsonify({"message": "Stop requested", "command_id": command_id})

    return jsonify({"error": "Unsupported action"}), 400


def cleanup():
    if cleanup_done.is_set():
        return
    cleanup_done.set()
    stop_event.set()
    audio_loop_event.clear()
    try:
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()
    except Exception:
        pass


atexit.register(cleanup)

if __name__ == "__main__":
    threading.Thread(target=sensor_loop, name="sensor-loop", daemon=True).start()
    logging.info("Pi 3 Dashboard running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)