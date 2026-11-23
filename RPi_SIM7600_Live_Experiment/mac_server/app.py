#!/usr/bin/env python3
"""Mac Flask dashboard: live MQTT telemetry, commands, and WAV reconstruction."""
import os, json, base64, uuid, threading
from datetime import datetime, timezone
from collections import deque
from flask import Flask, jsonify, request, render_template_string, send_from_directory
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from supabase import create_client

# MUST MATCH raspberry_pi/pi_agent.py
BROKER = "broker.hivemq.com"       # Public test broker only
PORT = 1883
TOPIC_ROOT = "rpi-experiment/change-this-unique-id"
TOPIC_TELEMETRY = f"{TOPIC_ROOT}/telemetry"
TOPIC_STATUS = f"{TOPIC_ROOT}/status"
TOPIC_COMMAND = f"{TOPIC_ROOT}/command"
TOPIC_ACK = f"{TOPIC_ROOT}/ack"
TOPIC_AUDIO = f"{TOPIC_ROOT}/audio"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
JSONL_PATH = os.path.join(DATA_DIR, "sensor_data.jsonl")
AUDIO_INDEX_PATH = os.path.join(DATA_DIR, "audio_index.jsonl")
os.makedirs(AUDIO_DIR, exist_ok=True)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_KEY in mac_server/.env")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
lock = threading.RLock()
latest = {}
history = deque(maxlen=300)  # Dashboard's recent view; full history is appended to JSONL on disk.
status = {"state": "waiting_for_device"}
acks = deque(maxlen=30)
audio_transfers = {}


def append_sample(payload):
    # Append-only log: it does not discard old samples when the dashboard buffer rolls over.
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def store_sample_in_supabase(payload):
    row = {
        "device_id": payload["device_id"],
        "sample": payload["sample"],
        "timestamp": payload["timestamp"],
        "timestamp_local": payload.get("timestamp_local"),
        "sensor_data": payload,
        "audio_path": payload.get("audio_path") or "",
    }
    supabase.table("sound_samples").insert(row).execute()


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        for topic in (TOPIC_TELEMETRY, TOPIC_STATUS, TOPIC_ACK, TOPIC_AUDIO):
            client.subscribe(topic, qos=1)
        print("MQTT connected; listening for Pi telemetry, status, ACKs and audio")
    else:
        print("MQTT connection error:", rc)


def on_message(client, userdata, msg):
    global latest, status
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        if msg.topic == TOPIC_TELEMETRY:
            with lock:
                latest = payload
                history.append(payload)
            append_sample(payload)
            try:
                store_sample_in_supabase(payload)
                print(f"Saved sensor sample {payload.get('sample')} to Supabase")
            except Exception:
                logging.exception(
                    "Supabase insert failed for sensor sample %s",
                    payload.get("sample"),
                )
        elif msg.topic == TOPIC_STATUS:
            with lock:
                status = payload
        elif msg.topic == TOPIC_ACK:
            with lock:
                acks.appendleft(payload)
        elif msg.topic == TOPIC_AUDIO:
            key = str(payload["command_id"])
            index = int(payload["chunk_index"])
            total = int(payload["total_chunks"])
            if total < 1 or total > 10000 or index < 0 or index >= total:
                raise ValueError("Invalid audio chunk metadata")
            raw = base64.b64decode(payload["data_b64"], validate=True)
            with lock:
                transfer = audio_transfers.setdefault(key, {
                    "filename": os.path.basename(payload["filename"]),
                    "recorded_at": payload.get("recorded_at"),
                    "session_id": payload.get("session_id"),
                    "total": total, "chunks": {}})
                if transfer["total"] != total:
                    raise ValueError("Audio chunk count changed during transfer")
                transfer["chunks"][index] = raw
                received = len(transfer["chunks"])
                if received == total:
                    filename = transfer["filename"]
                    output = os.path.join(AUDIO_DIR, filename)
                    with open(output, "wb") as f:
                        for chunk_index in range(total):
                            f.write(transfer["chunks"][chunk_index])
                    recorded_at = transfer.get("recorded_at")
                    audio_record = {
                        "filename": filename,
                        "recorded_at": recorded_at,
                        "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "bytes": os.path.getsize(output),
                        "url": f"/audio/{filename}",
                        "command_id": key,
                        "session_id": transfer.get("session_id")
                    }
                    with open(AUDIO_INDEX_PATH, "a", encoding="utf-8") as index_file:
                        index_file.write(json.dumps(audio_record, ensure_ascii=False) + "\n")
                    del audio_transfers[key]
                    status = {"state": "audio_received", "filename": filename,
                              "recorded_at": recorded_at, "url": f"/audio/{filename}",
                              "command_id": key, "audio_loop_active": bool(status.get("audio_loop_active", bool(transfer.get("session_id"))))}
            if received == total:
                print("Audio file reconstructed:", output)
    except Exception as exc:
        print("MQTT message processing error:", exc)


mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,
                          client_id=f"mac-live-dashboard-{uuid.uuid4().hex[:8]}")
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.reconnect_delay_set(min_delay=2, max_delay=30)
mqtt_client.connect_async(BROKER, PORT, 60)
mqtt_client.loop_start()

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Sensor Dashboard</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#e8eef9;font:15px Arial,sans-serif}main{max-width:1250px;margin:auto;padding:24px}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}h1{font-size:25px;margin:0}.muted{color:#9aabc4}.pill{padding:8px 12px;border-radius:20px;background:#26364d}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:13px;margin:22px 0}.card,.panel{background:#152238;border:1px solid #263954;border-radius:14px;padding:17px}.label{color:#9aabc4;font-size:13px}.value{font-size:27px;font-weight:700;margin-top:9px}.unit{font-size:13px;color:#9aabc4}
.panel{margin:15px 0}button{background:#4d8cff;color:white;border:0;border-radius:9px;padding:11px 16px;font-weight:700;cursor:pointer}input{background:#0c1728;color:white;border:1px solid #344965;border-radius:8px;padding:10px;width:140px}.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:9px;border-bottom:1px solid #2b3d56;white-space:nowrap}th{color:#9aabc4}a{color:#8db8ff}.led{display:inline-block;width:11px;height:11px;border-radius:50%;background:#526078;margin-right:8px}
</style></head><body><main><header><div><h1>Raspberry Pi Continuous Live Monitoring</h1><div class="muted">SIM7600EI · Jio · MQTT · BMP280 · SCD4X · Audio</div></div><div class="pill"><span id="deviceLed" class="led"></span><span id="deviceState">Waiting for Pi</span></div></header>
<div class="grid">
<div class="card"><div class="label">Pressure</div><div class="value" id="pressure">--</div><span class="unit">hPa</span></div>
<div class="card"><div class="label">BMP280 Temperature</div><div class="value" id="bmpTemp">--</div><span class="unit">°C</span></div>
<div class="card"><div class="label">CO₂</div><div class="value" id="co2">--</div><span class="unit">ppm</span></div>
<div class="card"><div class="label">SCD4X Temperature</div><div class="value" id="scdTemp">--</div><span class="unit">°C</span></div>
<div class="card"><div class="label">Relative Humidity</div><div class="value" id="humidity">--</div><span class="unit">%</span></div>
<div class="card"><div class="label">Sample number</div><div class="value" id="sample">--</div><span class="unit" id="sampleTime">Waiting for samples</span></div></div>
<section class="panel"><h3>Signal / LED confirmation</h3><p class="muted">Send a ping command. The Pi flashes GPIO17 and returns an acknowledgement.</p><button onclick="sendCommand('ping')">Test signal / LED</button> <span id="pingResult" class="muted"></span></section>
<section class="panel"><h3>Continuous audio recording</h3><p class="muted">Turn ON to record and upload consecutive 30-second WAV files. Files are sent one by one and saved on this Mac. Turn OFF to finish the current file and stop the loop.</p><div class="row"><button id="audioToggle" onclick="toggleAudioLoop()">Start audio · OFF</button><span id="audioState" class="muted">Audio recording is OFF</span></div></section>
<section class="panel"><h3>Saved audio recordings</h3><p class="muted">Each completed WAV is stored in <code>mac_server/data/audio/</code> with its recording time.</p><div id="audioFiles" class="muted">No audio files received yet.</div></section>
<section class="panel"><h3>Recent live readings</h3><div style="overflow:auto"><table><thead><tr><th>Timestamp</th><th>Sample</th><th>Pressure hPa</th><th>BMP °C</th><th>CO₂ ppm</th><th>SCD °C</th><th>Humidity %</th></tr></thead><tbody id="rows"></tbody></table></div><p><a href="/download/jsonl">Download complete sensor history (JSON Lines)</a></p></section>
<section class="panel"><h3>Latest command acknowledgement</h3><div id="ack" class="muted">Waiting for acknowledgement</div></section>
</main><script>
async function refresh(){try{const r=await fetch('/api/live');const d=await r.json();const x=d.latest||{};
for(const [id,key] of [['pressure','pressure_hPa'],['bmpTemp','bmp_temp'],['co2','co2_ppm'],['scdTemp','scd_temp'],['humidity','humidity']])document.getElementById(id).textContent=x[key]??'--';
document.getElementById('sample').textContent=x.sample??'--';document.getElementById('sampleTime').textContent=x.timestamp_local||x.timestamp||'Waiting for samples';
const state=d.status?.state||'waiting_for_device';const online=!!x.timestamp&&((Date.now()-new Date(x.timestamp).getTime())<15000);document.getElementById('deviceLed').style.background=online?'#35d07f':'#526078';document.getElementById('deviceState').textContent=online?'LIVE · '+state:state;
const audioOn=!!d.status?.audio_loop_active || ['audio_loop_active','audio_recording','audio_uploading','audio_uploaded','audio_loop_stopping'].includes(state);
const audioButton=document.getElementById('audioToggle');
audioButton.textContent=audioOn?'Stop audio · ON':'Start audio · OFF';
audioButton.style.background=audioOn?'#d9534f':'#23865b';
if(state.startsWith('audio_'))document.getElementById('audioState').textContent=d.status.message||state;
if(state==='audio_received')document.getElementById('audioState').textContent='Saved: '+(d.status.filename||'WAV file')+' · '+(d.status.recorded_at||'recording time unavailable');
document.getElementById('audioFiles').innerHTML=(d.audio_files||[]).map(f=>`<div style="padding:10px 0;border-bottom:1px solid #2b3d56"><strong>${f.recorded_at||f.filename}</strong><br><a href="${f.url}" target="_blank">Play / open WAV</a> · <a href="${f.url}" download>Download</a></div>`).join('')||'No audio files received yet.';
document.getElementById('rows').innerHTML=(d.history||[]).slice(-15).reverse().map(v=>`<tr><td>${v.timestamp_local||v.timestamp||''}</td><td>${v.sample??''}</td><td>${v.pressure_hPa??''}</td><td>${v.bmp_temp??''}</td><td>${v.co2_ppm??''}</td><td>${v.scd_temp??''}</td><td>${v.humidity??''}</td></tr>`).join('');
if(d.acks?.length){const a=d.acks[0];document.getElementById('ack').textContent=a.status+': '+(a.message||'')+' · '+(a.command_id||'');}
}catch(e){console.log(e)}}
async function sendCommand(action,extra={}){try{const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,...extra})});const d=await r.json();document.getElementById('pingResult').textContent=d.message||d.error||'Sent';}catch(e){document.getElementById('pingResult').textContent='Dashboard request failed';}}
async function toggleAudioLoop(){const d=await (await fetch('/api/live')).json();const active=!!d.status?.audio_loop_active||['audio_loop_active','audio_recording','audio_uploading','audio_uploaded','audio_loop_stopping'].includes(d.status?.state);const action=active?'stop_audio_loop':'start_audio_loop';document.getElementById('audioState').textContent=action==='start_audio_loop'?'Starting continuous audio...':'Stopping after current 30-second file uploads...';sendCommand(action); }
setInterval(refresh,1000);refresh();</script></body></html>'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/api/live')
def api_live():
    with lock:
        audio_files = []
        indexed_names = set()
        if os.path.exists(AUDIO_INDEX_PATH):
            with open(AUDIO_INDEX_PATH, "r", encoding="utf-8") as index_file:
                for line in index_file:
                    try:
                        item = json.loads(line)
                        indexed_names.add(item.get("filename"))
                        audio_files.append(item)
                    except (json.JSONDecodeError, TypeError):
                        continue
        # Include WAVs already present in the folder, even if they predate this index.
        for filename in os.listdir(AUDIO_DIR):
            if filename.lower().endswith(".wav") and filename not in indexed_names:
                full_path = os.path.join(AUDIO_DIR, filename)
                audio_files.append({
                    "filename": filename,
                    "recorded_at": datetime.fromtimestamp(os.path.getmtime(full_path)).astimezone().isoformat(timespec="seconds"),
                    "url": f"/audio/{filename}"
                })
        audio_files.sort(key=lambda item: item.get("recorded_at") or "", reverse=True)
        return jsonify({"latest": latest, "history": list(history), "status": status,
                        "acks": list(acks), "audio_files": audio_files[:200]})

@app.route('/api/command', methods=['POST'])
def api_command():
    body = request.get_json(silent=True) or {}
    action = body.get('action')
    if action not in ('ping', 'start_audio_loop', 'stop_audio_loop'):
        return jsonify({"error": "Unsupported action"}), 400
    command = {"command_id": uuid.uuid4().hex, "action": action,
               "timestamp": datetime.now(timezone.utc).isoformat()}
    info = mqtt_client.publish(TOPIC_COMMAND, json.dumps(command), qos=1)
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        return jsonify({"error": "MQTT publish failed"}), 503
    return jsonify({"message": "Command published", "command_id": command['command_id']})

@app.route('/download/jsonl')
def download_jsonl():
    if not os.path.exists(JSONL_PATH):
        open(JSONL_PATH, 'a').close()
    return send_from_directory(DATA_DIR, 'sensor_data.jsonl', as_attachment=True)

@app.route('/audio/<path:filename>')
def audio_file(filename):
    return send_from_directory(AUDIO_DIR, filename)

if __name__ == '__main__':
    print('Open http://127.0.0.1:5000 on this Mac')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
