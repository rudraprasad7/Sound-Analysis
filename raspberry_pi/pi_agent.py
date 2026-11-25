#!/usr/bin/env python3
"""Continuous Raspberry Pi sensor publisher and on-demand audio recorder."""
import os, json, time, uuid, base64, logging, threading, subprocess
from datetime import datetime, timezone

import board
import busio
import adafruit_bmp280
import adafruit_scd4x
import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO

# ---------- EDIT THESE SETTINGS ----------
DEVICE_ID = "raspberrypi3-01"
BROKER = "broker.hivemq.com"       # Public test broker; use private TLS broker for deployment
PORT = 1883
TOPIC_ROOT = "rpi-experiment/change-this-unique-id"  # Change to same unique value on Mac
SENSOR_INTERVAL = 1.0               # Send one sample every second
MIC = "plughw:2,0"                  # Your existing ALSA microphone
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_MAX_SECONDS = 300              # Safety cap: 5 minutes per request
AUDIO_CHUNK_BYTES = 6000             # Base64 payload stays small for MQTT
LED_PIN = 17                         # BCM 17 = physical pin 11
BASE = os.path.expanduser("~/Experiment")
# -----------------------------------------

TOPIC_TELEMETRY = f"{TOPIC_ROOT}/telemetry"
TOPIC_STATUS = f"{TOPIC_ROOT}/status"
TOPIC_COMMAND = f"{TOPIC_ROOT}/command"
TOPIC_ACK = f"{TOPIC_ROOT}/ack"
TOPIC_AUDIO = f"{TOPIC_ROOT}/audio"
os.makedirs(BASE, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Existing I2C sensors: BMP280 at 0x76 and SCD4X.
i2c = busio.I2C(board.SCL, board.SDA)
bmp = adafruit_bmp280.Adafruit_BMP280_I2C(i2c, address=0x76)
scd = adafruit_scd4x.SCD4X(i2c)
scd.start_periodic_measurement()
logging.info("Waiting for SCD4X initial measurement...")
time.sleep(5)

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=f"{DEVICE_ID}-{uuid.uuid4().hex[:8]}")
connected = threading.Event()
stop_event = threading.Event()
audio_lock = threading.Lock()
audio_loop_event = threading.Event()
audio_loop_thread = None
sample_number = 0
last_scd_reading = {"co2_ppm": None, "scd_temp": None, "humidity": None}


def publish_json(topic, payload, qos=1):
    if not client.is_connected():
        logging.warning("MQTT disconnected; sample/command response not sent")
        return False
    info = client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=qos)
    info.wait_for_publish(timeout=15)
    return info.rc == mqtt.MQTT_ERR_SUCCESS


def send_status(state, **extra):
    payload = {"device_id": DEVICE_ID, "state": state,
               "timestamp": datetime.now(timezone.utc).isoformat(), **extra}
    publish_json(TOPIC_STATUS, payload)


def acknowledge(command_id, state, message=""):
    publish_json(TOPIC_ACK, {"device_id": DEVICE_ID, "command_id": command_id,
                            "status": state, "message": message,
                            "timestamp": datetime.now(timezone.utc).isoformat()})


def flash_led(count=2, on_time=0.18, off_time=0.18):
    for _ in range(count):
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(on_time)
        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(off_time)


def record_and_send_audio(seconds, command_id, session_id=None):
    seconds = max(1, min(int(seconds), AUDIO_MAX_SECONDS))
    recorded_at = datetime.now().astimezone().isoformat(timespec="seconds")
    filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{command_id[:8]}.wav"
    path = os.path.join(BASE, filename)
    # Do not allow two recordings to use the same microphone simultaneously.
    if not audio_lock.acquire(blocking=False):
        acknowledge(command_id, "busy", "Another audio recording is already running")
        return
    try:
        send_status("audio_recording", command_id=command_id, session_id=session_id, seconds=seconds, filename=filename, recorded_at=recorded_at, audio_loop_active=bool(session_id and audio_loop_event.is_set()))
        command = ["arecord", "-D", MIC, "-f", "S16_LE", "-r", str(AUDIO_RATE),
                   "-c", str(AUDIO_CHANNELS), "-t", "wav", "-d", str(seconds), path]
        subprocess.run(command, check=True, timeout=seconds + 20)
        file_size = os.path.getsize(path)
        total = (file_size + AUDIO_CHUNK_BYTES - 1) // AUDIO_CHUNK_BYTES
        # Announce transfer before chunks; Mac assembles them by index.
        publish_json(TOPIC_STATUS, {"device_id": DEVICE_ID, "state": "audio_uploading",
                                    "command_id": command_id, "filename": filename, "recorded_at": recorded_at, "session_id": session_id,
                                    "bytes": file_size, "total_chunks": total, "audio_loop_active": bool(session_id and audio_loop_event.is_set())})
        with open(path, "rb") as audio_file:
            for index in range(total):
                raw = audio_file.read(AUDIO_CHUNK_BYTES)
                packet = {"device_id": DEVICE_ID, "command_id": command_id,
                          "filename": filename, "recorded_at": recorded_at, "session_id": session_id,
                          "chunk_index": index, "total_chunks": total,
                          "data_b64": base64.b64encode(raw).decode("ascii")}
                if not publish_json(TOPIC_AUDIO, packet, qos=1):
                    raise RuntimeError(f"MQTT audio chunk {index + 1}/{total} failed")
        send_status("audio_uploaded", command_id=command_id, session_id=session_id, filename=filename,
                    recorded_at=recorded_at, bytes=file_size, seconds=seconds, audio_loop_active=bool(session_id and audio_loop_event.is_set()))
        acknowledge(command_id, "completed", f"Audio transferred: {filename}")
        logging.info("Audio saved and transferred: %s (%s bytes)", path, file_size)
    except Exception as exc:
        logging.exception("Audio recording/transfer failed")
        send_status("audio_error", command_id=command_id, filename=filename, error=str(exc))
        acknowledge(command_id, "error", str(exc))
    finally:
        audio_lock.release()



def continuous_audio_loop(session_id):
    """Record and upload consecutive 30-second WAV files until the dashboard turns audio OFF.
    The current 30-second recording and its upload are completed before the next file is started.
    """
    logging.info("Continuous audio loop started: %s", session_id)
    send_status("audio_loop_active", session_id=session_id, audio_loop_active=True,
                message="Recording 30-second WAV files continuously")
    while audio_loop_event.is_set() and not stop_event.is_set():
        recording_id = uuid.uuid4().hex
        record_and_send_audio(30, recording_id, session_id=session_id)
    send_status("audio_loop_stopped", session_id=session_id, audio_loop_active=False,
                message="Continuous audio recording stopped")



def on_connect(mqtt_client, userdata, flags, rc):
    if rc == 0:
        connected.set()
        mqtt_client.subscribe(TOPIC_COMMAND, qos=1)
        send_status("online", message="Continuous sensor streaming active")
        logging.info("MQTT connected; subscribed to %s", TOPIC_COMMAND)
    else:
        logging.error("MQTT connection refused: %s", rc)


def on_disconnect(mqtt_client, userdata, rc):
    connected.clear()
    logging.warning("MQTT disconnected (rc=%s); Paho will reconnect", rc)


def on_message(mqtt_client, userdata, msg):
    try:
        command = json.loads(msg.payload.decode("utf-8"))
        command_id = str(command.get("command_id") or uuid.uuid4().hex)
        action = command.get("action")
        # Flash immediately so the physical LED confirms command receipt.
        threading.Thread(target=flash_led, daemon=True).start()
        if action == "ping":
            acknowledge(command_id, "received", "Pi received dashboard ping; GPIO17 LED flashed")
        elif action == "start_audio_loop":
            global audio_loop_thread
            if audio_loop_event.is_set():
                acknowledge(command_id, "already_active", "Continuous audio recording is already ON")
                return
            session_id = command_id
            audio_loop_event.set()
            audio_loop_thread = threading.Thread(
                target=continuous_audio_loop, args=(session_id,),
                name="continuous-audio-loop", daemon=True)
            audio_loop_thread.start()
            acknowledge(command_id, "started", "Continuous 30-second audio recording started")
        elif action == "stop_audio_loop":
            if not audio_loop_event.is_set():
                acknowledge(command_id, "already_stopped", "Continuous audio recording is already OFF")
                return
            audio_loop_event.clear()
            send_status("audio_loop_stopping", audio_loop_active=False,
                        message="Finishing the current recording/upload; no next recording will start")
            acknowledge(command_id, "stop_requested",
                        "OFF received; the current 30-second file will finish uploading, then recording stops")
        else:
            acknowledge(command_id, "error", "Unknown action")
    except Exception as exc:
        logging.exception("Invalid MQTT command")


client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
client.reconnect_delay_set(min_delay=2, max_delay=30)


def connect_mqtt():
    while not stop_event.is_set():
        try:
            logging.info("Connecting to %s:%s ...", BROKER, PORT)
            client.connect(BROKER, PORT, keepalive=60)
            client.loop_start()
            if connected.wait(25):
                return
            client.loop_stop()
            client.disconnect()
        except Exception:
            logging.exception("MQTT connection failed; retrying in 5 seconds")
        time.sleep(5)
    raise RuntimeError("Stopped before MQTT connection")


def read_sensors():
    global sample_number
    sample_number += 1
    now = datetime.now(timezone.utc)
    row = {
        "device_id": DEVICE_ID,
        "sample": sample_number,
        "timestamp": now.isoformat(),
        "timestamp_local": now.astimezone().isoformat(timespec="seconds"),
        "pressure_hPa": round(float(bmp.pressure), 2),
        "bmp_temp": round(float(bmp.temperature), 2),
    }
    if scd.data_ready:
        last_scd_reading.update({"co2_ppm": int(scd.CO2),
                                 "scd_temp": round(float(scd.temperature), 2),
                                 "humidity": round(float(scd.relative_humidity), 2)})
    # SCD4X updates more slowly than the one-second loop; retain its latest valid value.
    row.update(last_scd_reading)
    return row


def sensor_loop():
    """Run indefinitely. No fixed cycle count or experiment duration."""
    next_tick = time.monotonic()
    logging.info("Continuous sensor streaming started; interval %.1f seconds", SENSOR_INTERVAL)
    while not stop_event.is_set():
        try:
            row = read_sensors()
            sent = publish_json(TOPIC_TELEMETRY, row, qos=1)
            logging.info("sample=%s sent=%s data=%s", row["sample"], sent, row)
        except Exception:
            logging.exception("Sensor read/publish error; continuing")
        next_tick += SENSOR_INTERVAL
        delay = max(0, next_tick - time.monotonic())
        stop_event.wait(delay)


def main():
    try:
        connect_mqtt()
        sensor_loop()
    except KeyboardInterrupt:
        logging.info("Ctrl+C received; stopping agent")
    finally:
        stop_event.set()
        audio_loop_event.clear()
        try:
            send_status("offline", message="Agent stopping")
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
