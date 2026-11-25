#!/usr/bin/env python3
"""Live Flask Dashboard: MQTT telemetry, Pi commands, Supabase Storage raw/cleaned/plot audio, and on-demand ML analysis page."""
import os
import io
import gc
import time
import json
import base64
import uuid
import logging
import tempfile
import threading
from datetime import datetime, timezone
from collections import deque

import numpy as np
import librosa
import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template_string, send_from_directory
import paho.mqtt.client as mqtt
from supabase import create_client

from clean_audio import clean_farm_audio_ultimate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# MUST MATCH your Raspberry Pi script
BROKER = "broker.hivemq.com"       # Public test broker only
PORT = 1883
TOPIC_ROOT = "rpi-experiment/change-this-unique-id"
TOPIC_TELEMETRY = f"{TOPIC_ROOT}/telemetry"
TOPIC_STATUS = f"{TOPIC_ROOT}/status"
TOPIC_COMMAND = f"{TOPIC_ROOT}/command"
TOPIC_ACK = f"{TOPIC_ROOT}/ack"
TOPIC_AUDIO = f"{TOPIC_ROOT}/audio"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
JSONL_PATH = os.path.join(DATA_DIR, "sensor_data.jsonl")
CACHE_PATH = os.path.join(DATA_DIR, "analysis_cache.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Supabase configuration (for Storage bucket raw/cleaned/plots and optional sound_samples table)
load_dotenv(os.path.join(BASE_DIR, ".env"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
AUDIO_BUCKET = os.getenv("SUPABASE_AUDIO_BUCKET", "poultry-audio")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) in .env or Render Environment"
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Acoustic & ML Threshold Configuration
AUDIO_RATE = 16000
ANOMALY_THRESHOLD = -85.00
MIN_ANOMALOUS_SECONDS = 2
DB_CALIBRATION_OFFSET = 0.0
DB_QUIET_MAX = -45.0        # < -45 dBFS  -> quiet
DB_SOFT_MAX = -30.0         # -45 to -30  -> soft
DB_MODERATE_MAX = -18.0     # -30 to -18  -> moderate
                            # >= -18 dBFS -> loud

# Load ML models once at worker startup
scaler = None
gmm = None
pca = None
model_load_error = None
try:
    scaler = joblib.load(os.path.join(BASE_DIR, "pi_scaler.joblib"))
    gmm = joblib.load(os.path.join(BASE_DIR, "pi_gmm.joblib"))
    pca = joblib.load(os.path.join(BASE_DIR, "pi_pca.joblib"))
except Exception as exc:
    model_load_error = f"Failed to load ML models (.joblib): {exc}"
    logging.warning(model_load_error)

app = Flask(__name__)
lock = threading.RLock()
clean_lock = threading.Lock()
plot_lock = threading.Lock()

latest = {}
history = deque(maxlen=300)
status = {"state": "waiting_for_device"}
acks = deque(maxlen=30)
audio_transfers = {}

# Lightweight 3-second cache for /api/live so 1-second UI polling doesn't hammer Supabase Storage
recordings_cache = {"ts": 0.0, "items": []}
CACHE_TTL_SEC = 3.0

analysis_state = {
    "running": False,
    "processed": 0,
    "total": 0,
    "current_file": None,
    "last_run": None,
    "error": None,
}


def load_local_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, dict):
                    data.setdefault("recordings", {})
                    data.setdefault("results", {})
                    return data
        except Exception:
            pass
    return {"recordings": {}, "results": {}}


def save_local_cache(cache_obj):
    with open(CACHE_PATH, "w", encoding="utf-8") as fp:
        json.dump(cache_obj, fp, indent=2, ensure_ascii=False)


def invalidate_recordings_cache():
    with lock:
        recordings_cache["ts"] = 0.0


def upload_bytes_to_supabase(storage_path, file_bytes, content_type):
    """Upload bytes to Supabase Storage and return the public URL."""
    supabase.storage.from_(AUDIO_BUCKET).upload(
        path=storage_path,
        file=file_bytes,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    return supabase.storage.from_(AUDIO_BUCKET).get_public_url(storage_path)


def download_from_supabase(storage_path, local_dest):
    """Download an object from Supabase Storage to a local temporary path."""
    data = supabase.storage.from_(AUDIO_BUCKET).download(storage_path)
    with open(local_dest, "wb") as fp:
        fp.write(data)


def list_raw_wav_files(limit=200, use_cache=True):
    """List .wav files directly from the 'raw/' folder in Supabase Storage merged with local cache."""
    now = time.monotonic()
    if use_cache:
        with lock:
            if now - recordings_cache["ts"] < CACHE_TTL_SEC:
                return list(recordings_cache["items"])

    files = supabase.storage.from_(AUDIO_BUCKET).list("raw") or []
    with lock:
        cache = load_local_cache()
        cached_recs = cache.get("recordings", {})
        cached_res = cache.get("results", {})

    items = []
    for f in files:
        name = f.get("name", "")
        if not name.lower().endswith(".wav"):
            continue
        stem = os.path.splitext(name)[0]
        raw_path = f"raw/{name}"
        raw_url = supabase.storage.from_(AUDIO_BUCKET).get_public_url(raw_path)
        cleaned_path = f"cleaned/cleaned_{name}"
        cleaned_url = supabase.storage.from_(AUDIO_BUCKET).get_public_url(cleaned_path)
        plot_path = f"plots/plot_{stem}.png"
        plot_url = supabase.storage.from_(AUDIO_BUCKET).get_public_url(plot_path)

        prev = cached_recs.get(name, {})
        res = cached_res.get(name)
        recorded_at = prev.get("recorded_at") or f.get("created_at") or f.get("updated_at") or name
        cleaning_state = prev.get("cleaning_state") or ("ready" if isinstance(res, dict) and res.get("status") == "ok" else "pending")

        items.append({
            "filename": name,
            "recorded_at": recorded_at,
            "raw_path": raw_path,
            "raw_url": raw_url,
            "cleaned_path": cleaned_path,
            "cleaned_url": prev.get("cleaned_url") or cleaned_url,
            "cleaning_state": cleaning_state,
            "cleaning_error": prev.get("cleaning_error"),
            "plot_path": plot_path,
            "plot_url": (res.get("plot_url") if isinstance(res, dict) and res.get("plot_url") else plot_url),
            "analysis_result": res,
        })

    items.sort(key=lambda x: x["filename"], reverse=True)
    items = items[:limit]

    with lock:
        recordings_cache["items"] = items
        recordings_cache["ts"] = now
    return list(items)


def append_sample(payload):
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


def process_incoming_audio_worker(wav_bytes, filename, recorded_at, command_id, session_id):
    """Upload raw WAV to Supabase Storage 'raw/', then clean and upload cleaned WAV to 'cleaned/'."""
    global status
    raw_storage_path = f"raw/{filename}"
    cleaned_filename = f"cleaned_{filename}"
    cleaned_storage_path = f"cleaned/{cleaned_filename}"

    raw_path = None
    clean_path = None
    try:
        # 1. Upload raw WAV to Supabase Storage 'raw/'
        raw_url = upload_bytes_to_supabase(raw_storage_path, wav_bytes, "audio/wav")
        with lock:
            cache = load_local_cache()
            cache["recordings"][filename] = {
                "filename": filename,
                "recorded_at": recorded_at,
                "command_id": command_id,
                "session_id": session_id,
                "raw_path": raw_storage_path,
                "raw_url": raw_url,
                "cleaned_path": cleaned_storage_path,
                "cleaning_state": "cleaning",
                "cleaning_error": None,
            }
            save_local_cache(cache)
        invalidate_recordings_cache()

        with lock:
            status = {
                "state": "audio_received",
                "filename": filename,
                "recorded_at": recorded_at,
                "url": raw_url,
                "command_id": command_id,
                "audio_loop_active": bool(status.get("audio_loop_active", bool(session_id))),
            }
        print(f"Uploaded raw WAV to Supabase Storage: {raw_storage_path}")

        # 2. Clean in temp files
        raw_fd, raw_path = tempfile.mkstemp(suffix="_raw.wav")
        clean_fd, clean_path = tempfile.mkstemp(suffix="_clean.wav")
        os.close(raw_fd)
        os.close(clean_fd)

        with open(raw_path, "wb") as fp:
            fp.write(wav_bytes)

        with clean_lock:
            clean_farm_audio_ultimate(raw_path, clean_path)

        with open(clean_path, "rb") as fp:
            cleaned_bytes = fp.read()

        # 3. Upload cleaned WAV to Supabase Storage 'cleaned/'
        cleaned_url = upload_bytes_to_supabase(cleaned_storage_path, cleaned_bytes, "audio/wav")
        with lock:
            cache = load_local_cache()
            rec = cache["recordings"].setdefault(filename, {"filename": filename})
            rec.update({
                "recorded_at": recorded_at,
                "raw_path": raw_storage_path,
                "raw_url": raw_url,
                "cleaned_path": cleaned_storage_path,
                "cleaned_url": cleaned_url,
                "cleaning_state": "ready",
                "cleaning_error": None,
            })
            save_local_cache(cache)
        invalidate_recordings_cache()

        with lock:
            status = {
                "state": "audio_cleaned",
                "filename": cleaned_filename,
                "recorded_at": recorded_at,
                "url": cleaned_url,
                "command_id": command_id,
                "audio_loop_active": bool(status.get("audio_loop_active", False)),
            }
        print(f"Uploaded cleaned WAV to Supabase Storage: {cleaned_storage_path}")
    except Exception as exc:
        logging.exception("Supabase audio upload/cleaning failed for %s", filename)
        try:
            with lock:
                cache = load_local_cache()
                rec = cache["recordings"].setdefault(filename, {"filename": filename})
                rec.update({
                    "cleaning_state": "error",
                    "cleaning_error": str(exc),
                })
                save_local_cache(cache)
            invalidate_recordings_cache()
        except Exception:
            pass
    finally:
        for path in (raw_path, clean_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        gc.collect()


def render_plot_png_bytes(X_pca, scores, db_values, filename):
    """Render side-by-side 2D PCA GMM + 1-second dBFS plot to PNG bytes in memory."""
    with plot_lock:
        fig = Figure(figsize=(13, 5), dpi=115)
        FigureCanvasAgg(fig)
        ax1, ax2 = fig.subplots(1, 2)

        # --- Left Panel: 2D PCA GMM Decision Boundary & Discrete Clusters ---
        x_min, x_max = float(X_pca[:, 0].min()) - 5.0, float(X_pca[:, 0].max()) + 5.0
        y_min, y_max = float(X_pca[:, 1].min()) - 5.0, float(X_pca[:, 1].max()) + 5.0
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 70), np.linspace(y_min, y_max, 70))
        grid_scores = gmm.score_samples(
            pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
        ).reshape(xx.shape)

        g_min = float(grid_scores.min())
        g_max = float(grid_scores.max())
        z_low = min(g_min - 1.0, ANOMALY_THRESHOLD - 1.0)
        z_high = max(g_max + 1.0, ANOMALY_THRESHOLD + 1.0)

        ax1.contourf(
            xx, yy, grid_scores,
            levels=[z_low, ANOMALY_THRESHOLD, z_high],
            colors=["#ffcccc", "#ccffcc"],
            alpha=0.55,
        )
        if g_min < ANOMALY_THRESHOLD < g_max:
            ax1.contour(
                xx, yy, grid_scores,
                levels=[ANOMALY_THRESHOLD],
                colors="#cc0000",
                linewidths=2.0,
                linestyles="dashed",
            )

        gmm_colors = [
            "#2ecc71" if s >= -60.0 else
            "#27ae60" if s >= ANOMALY_THRESHOLD else
            "#f39c12" if s >= -110.0 else
            "#c0392b"
            for s in scores
        ]
        ax1.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=gmm_colors, edgecolors="black", linewidths=0.5, s=55, alpha=0.9,
        )
        ax1.text(
            x_max - 1.5, y_max - 1.5, f"NORMAL (≥ {ANOMALY_THRESHOLD})",
            color="darkgreen", fontsize=9.5, fontweight="bold", ha="right", va="top",
        )
        ax1.text(
            x_max - 1.5, y_min + 1.5, f"ANOMALY (< {ANOMALY_THRESHOLD})",
            color="darkred", fontsize=9.5, fontweight="bold", ha="right", va="bottom",
        )
        ax1.legend(
            handles=[
                Patch(facecolor="#2ecc71", edgecolor="black", label=f"Optimal: {int(np.sum(scores >= -60.0))}s"),
                Patch(facecolor="#27ae60", edgecolor="black", label=f"Normal: {int(np.sum((scores < -60.0) & (scores >= ANOMALY_THRESHOLD)))}s"),
                Patch(facecolor="#f39c12", edgecolor="black", label=f"Warning: {int(np.sum((scores < ANOMALY_THRESHOLD) & (scores >= -110.0)))}s"),
                Patch(facecolor="#c0392b", edgecolor="black", label=f"Severe: {int(np.sum(scores < -110.0))}s"),
            ],
            loc="upper left", fontsize=8, framealpha=0.9,
        )
        ax1.set_title(f"GMM 2D PCA Clusters ({filename})")
        ax1.set_xlabel("PCA Component 1")
        ax1.set_ylabel("PCA Component 2")

        # --- Right Panel: 1-Second Decibel (dBFS) Clusters ---
        secs = np.arange(len(db_values))
        clamped_db = [max(-80.0, min(0.0, float(d))) for d in db_values]
        db_colors = [
            "#3498db" if d < DB_QUIET_MAX else
            "#2ecc71" if d < DB_SOFT_MAX else
            "#f39c12" if d < DB_MODERATE_MAX else
            "#e74c3c"
            for d in db_values
        ]
        ax2.bar(
            secs, [d + 80.0 for d in clamped_db],
            bottom=-80.0, color=db_colors, edgecolor="black", linewidth=0.5,
        )
        ax2.axhline(DB_QUIET_MAX, color="#3498db", linestyle="--", alpha=0.6)
        ax2.axhline(DB_SOFT_MAX, color="#2ecc71", linestyle="--", alpha=0.6)
        ax2.axhline(DB_MODERATE_MAX, color="#e74c3c", linestyle="--", alpha=0.6)
        ax2.set_ylim(-80.0, 0.0)
        ax2.set_title("1-Second Decibel Clusters (dBFS)")
        ax2.set_xlabel("Second in Window")
        ax2.set_ylabel("Level (dBFS)")
        ax2.legend(
            handles=[
                Patch(facecolor="#3498db", edgecolor="black", label=f"Quiet (< {DB_QUIET_MAX})"),
                Patch(facecolor="#2ecc71", edgecolor="black", label=f"Soft ({DB_QUIET_MAX}..{DB_SOFT_MAX})"),
                Patch(facecolor="#f39c12", edgecolor="black", label=f"Moderate ({DB_SOFT_MAX}..{DB_MODERATE_MAX})"),
                Patch(facecolor="#e74c3c", edgecolor="black", label=f"Loud (≥ {DB_MODERATE_MAX})"),
            ],
            loc="upper right", fontsize=8, framealpha=0.9,
        )

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        fig.clear()
        return buf.getvalue()


def analyze_recording_item(item):
    """Fetch raw WAV from Storage 'raw/', clean it, run GMM+PCA+dBFS analysis, and upload plot PNG."""
    if model_load_error:
        raise RuntimeError(model_load_error)

    filename = item["filename"]
    recorded_at = item.get("recorded_at") or filename
    raw_storage_path = item["raw_path"]
    raw_url = item["raw_url"]
    cleaned_storage_path = item["cleaned_path"]
    cleaned_url = item["cleaned_url"]

    raw_fd, raw_path = tempfile.mkstemp(suffix="_raw.wav")
    clean_fd, clean_path = tempfile.mkstemp(suffix="_clean.wav")
    os.close(raw_fd)
    os.close(clean_fd)

    try:
        # 1. Fetch raw WAV directly from Supabase Storage 'raw/'
        download_from_supabase(raw_storage_path, raw_path)
        y_raw, sr_raw = librosa.load(raw_path, sr=AUDIO_RATE, mono=True)

        # 2. Clean audio & upload cleaned WAV to Supabase Storage 'cleaned/'
        with clean_lock:
            clean_farm_audio_ultimate(raw_path, clean_path)
        with open(clean_path, "rb") as fp:
            cleaned_url = upload_bytes_to_supabase(cleaned_storage_path, fp.read(), "audio/wav")

        y_clean, sr_clean = librosa.load(clean_path, sr=AUDIO_RATE, mono=True)

        # 3. Extract 40D MFCCs (from cleaned) & 1-second dBFS (from raw)
        mfccs = librosa.feature.mfcc(y=y_clean, sr=sr_clean, n_mfcc=20, hop_length=512)
        frames_per_sec = int(sr_clean / 512)
        samples_per_sec = int(sr_raw)

        features_list = []
        db_values = []
        sec_idx = 0
        for i in range(0, mfccs.shape[1], frames_per_sec):
            chunk = mfccs[:, i:i + frames_per_sec]
            audio_chunk = y_raw[sec_idx * samples_per_sec:(sec_idx + 1) * samples_per_sec]
            if chunk.shape[1] == frames_per_sec and len(audio_chunk) == samples_per_sec:
                features_list.append(np.hstack((np.mean(chunk, axis=1), np.std(chunk, axis=1))))
                rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                db_values.append(float(20.0 * np.log10(rms + 1e-10) + DB_CALIBRATION_OFFSET))
                sec_idx += 1

        if not features_list:
            raise ValueError("Audio too short (< 1 second)")

        X = np.array(features_list)
        X_scaled = scaler.transform(X)
        scores = gmm.score_samples(X_scaled)
        X_pca = pca.transform(X_scaled)

        optimal_sec = int(np.sum(scores >= -60.0))
        normal_sec = int(np.sum((scores < -60.0) & (scores >= ANOMALY_THRESHOLD)))
        warning_sec = int(np.sum((scores < ANOMALY_THRESHOLD) & (scores >= -110.0)))
        severe_sec = int(np.sum(scores < -110.0))

        anomalous_seconds = warning_sec + severe_sec
        is_abnormal = anomalous_seconds >= MIN_ANOMALOUS_SECONDS
        lowest_score = float(np.min(scores))
        avg_score = float(np.mean(scores))

        if severe_sec >= MIN_ANOMALOUS_SECONDS:
            gmm_cluster = "severe"
        elif is_abnormal:
            gmm_cluster = "warning"
        elif optimal_sec >= normal_sec:
            gmm_cluster = "optimal"
        else:
            gmm_cluster = "normal"

        overall_rms = float(np.sqrt(np.mean(y_raw ** 2)))
        avg_db = round(float(20.0 * np.log10(overall_rms + 1e-10) + DB_CALIBRATION_OFFSET), 2)
        max_db = round(float(np.max(db_values)), 2)

        if avg_db < DB_QUIET_MAX:
            db_cluster = "quiet"
        elif avg_db < DB_SOFT_MAX:
            db_cluster = "soft"
        elif avg_db < DB_MODERATE_MAX:
            db_cluster = "moderate"
        else:
            db_cluster = "loud"

        db_arr = np.array(db_values)
        db_dist = {
            "quiet_sec": int(np.sum(db_arr < DB_QUIET_MAX)),
            "soft_sec": int(np.sum((db_arr >= DB_QUIET_MAX) & (db_arr < DB_SOFT_MAX))),
            "moderate_sec": int(np.sum((db_arr >= DB_SOFT_MAX) & (db_arr < DB_MODERATE_MAX))),
            "loud_sec": int(np.sum(db_arr >= DB_MODERATE_MAX)),
        }

        # 4. Render plot PNG in memory & upload to Supabase Storage 'plots/'
        plot_storage_path = item["plot_path"]
        png_bytes = render_plot_png_bytes(X_pca, scores, db_values, filename)
        plot_url = upload_bytes_to_supabase(plot_storage_path, png_bytes, "image/png")

        analyzed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        result = {
            "status": "ok",
            "filename": filename,
            "recorded_at": recorded_at,
            "analyzed_at": analyzed_at,
            "alert": bool(is_abnormal),
            "class": "abnormal_sound" if is_abnormal else "normal",
            "gmm_cluster": gmm_cluster,
            "gmm_avg_score": round(avg_score, 2),
            "confidence": float(round(abs(lowest_score), 2)),
            "anomalous_seconds": anomalous_seconds,
            "total_seconds": len(features_list),
            "avg_db": avg_db,
            "max_db": max_db,
            "db_cluster": db_cluster,
            "cluster_distribution": {
                "optimal_sec": optimal_sec,
                "normal_sec": normal_sec,
                "warning_sec": warning_sec,
                "severe_sec": severe_sec,
            },
            "db_cluster_distribution": db_dist,
            "raw_url": raw_url,
            "cleaned_url": cleaned_url,
            "plot_url": plot_url,
        }

        with lock:
            cache = load_local_cache()
            cache["recordings"][filename] = {
                "filename": filename,
                "recorded_at": recorded_at,
                "raw_path": raw_storage_path,
                "raw_url": raw_url,
                "cleaned_path": cleaned_storage_path,
                "cleaned_url": cleaned_url,
                "cleaning_state": "ready",
                "cleaning_error": None,
            }
            cache["results"][filename] = result
            save_local_cache(cache)
        invalidate_recordings_cache()

        del y_raw, y_clean, mfccs, features_list, X, X_scaled, X_pca, scores, db_arr, png_bytes
        return result
    finally:
        for path in (raw_path, clean_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        gc.collect()


def run_batch_analysis_worker(force=False):
    """Fetch .wav files directly from Supabase Storage 'raw/' and analyze unanalyzed items."""
    try:
        if model_load_error:
            with lock:
                analysis_state["error"] = model_load_error
            return

        items = list_raw_wav_files(limit=200, use_cache=False)
        with lock:
            analysis_state["total"] = len(items)
            analysis_state["processed"] = 0
            analysis_state["error"] = None

        for idx, item in enumerate(items, start=1):
            filename = item["filename"]
            with lock:
                analysis_state["current_file"] = filename

            existing_res = item.get("analysis_result")
            needs_run = (
                force
                or not isinstance(existing_res, dict)
                or existing_res.get("status") != "ok"
                or not existing_res.get("plot_url")
            )

            if needs_run:
                try:
                    analyze_recording_item(item)
                except Exception as exc:
                    logging.exception("Analysis failed for %s", filename)
                    err_res = {
                        "status": "error",
                        "filename": filename,
                        "recorded_at": item.get("recorded_at") or filename,
                        "error": str(exc),
                        "raw_url": item.get("raw_url"),
                    }
                    with lock:
                        cache = load_local_cache()
                        cache["results"][filename] = err_res
                        save_local_cache(cache)
                    invalidate_recordings_cache()

            with lock:
                analysis_state["processed"] = idx

        with lock:
            analysis_state["last_run"] = datetime.now().astimezone().isoformat(timespec="seconds")
    except Exception as exc:
        logging.exception("Batch analysis failed")
        with lock:
            analysis_state["error"] = str(exc)
    finally:
        with lock:
            analysis_state["running"] = False
            analysis_state["current_file"] = None


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

            completed_job = None
            with lock:
                transfer = audio_transfers.setdefault(key, {
                    "filename": os.path.basename(payload["filename"]),
                    "recorded_at": payload.get("recorded_at"),
                    "session_id": payload.get("session_id"),
                    "total": total,
                    "chunks": {},
                })
                if transfer["total"] != total:
                    raise ValueError("Audio chunk count changed during transfer")
                transfer["chunks"][index] = raw
                received = len(transfer["chunks"])
                if received == total:
                    wav_bytes = b"".join(transfer["chunks"][i] for i in range(total))
                    filename = transfer["filename"]
                    recorded_at = transfer.get("recorded_at") or datetime.now().astimezone().isoformat(timespec="seconds")
                    session_id = transfer.get("session_id")
                    del audio_transfers[key]
                    completed_job = (wav_bytes, filename, recorded_at, key, session_id)

            if completed_job is not None:
                print(f"Reconstructed {completed_job[1]} ({len(completed_job[0])} bytes); uploading to Supabase raw/...")
                threading.Thread(
                    target=process_incoming_audio_worker,
                    args=completed_job,
                    daemon=True,
                ).start()
    except Exception as exc:
        print("MQTT message processing error:", exc)


mqtt_client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION1,
    client_id=f"live-dashboard-{uuid.uuid4().hex[:8]}",
)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.reconnect_delay_set(min_delay=2, max_delay=30)
mqtt_client.connect_async(BROKER, PORT, 60)
mqtt_client.loop_start()


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Sensor & Cleaned Audio Dashboard</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#e8eef9;font:15px Arial,sans-serif}main{max-width:1250px;margin:auto;padding:24px}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}h1{font-size:25px;margin:0}.muted{color:#9aabc4}.pill{padding:8px 12px;border-radius:20px;background:#26364d}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:13px;margin:22px 0}.card,.panel{background:#152238;border:1px solid #263954;border-radius:14px;padding:17px}.label{color:#9aabc4;font-size:13px}.value{font-size:27px;font-weight:700;margin-top:9px}.unit{font-size:13px;color:#9aabc4}
.panel{margin:15px 0}button{background:#4d8cff;color:white;border:0;border-radius:9px;padding:11px 16px;font-weight:700;cursor:pointer}.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:9px;border-bottom:1px solid #2b3d56;white-space:nowrap}th{color:#9aabc4}a{color:#8db8ff}.led{display:inline-block;width:11px;height:11px;border-radius:50%;background:#526078;margin-right:8px}
.badge{display:inline-block;padding:3px 9px;border-radius:12px;font-size:12px;font-weight:700;margin-left:8px}
.badge-ready{background:#1e4d38;color:#6ef2b1}.badge-cleaning{background:#4d3e1e;color:#f7c96e}.badge-error{background:#4d1e24;color:#ff8a96}
audio{height:34px;margin-top:8px;width:100%;max-width:460px}
</style></head><body><main><header><div><h1>Raspberry Pi Continuous Live Monitoring</h1><div class="muted">SIM7600EI · MQTT · BMP280 · SCD4X · 4-Stage Denoised Audio</div></div><div class="row"><a href="/analysis" style="text-decoration:none"><button type="button">Audio Analysis</button></a><div class="pill"><span id="deviceLed" class="led"></span><span id="deviceState">Waiting for Pi</span></div></div></header>
<div class="grid">
<div class="card"><div class="label">Pressure</div><div class="value" id="pressure">--</div><span class="unit">hPa</span></div>
<div class="card"><div class="label">BMP280 Temperature</div><div class="value" id="bmpTemp">--</div><span class="unit">°C</span></div>
<div class="card"><div class="label">CO₂</div><div class="value" id="co2">--</div><span class="unit">ppm</span></div>
<div class="card"><div class="label">SCD4X Temperature</div><div class="value" id="scdTemp">--</div><span class="unit">°C</span></div>
<div class="card"><div class="label">Relative Humidity</div><div class="value" id="humidity">--</div><span class="unit">%</span></div>
<div class="card"><div class="label">Sample number</div><div class="value" id="sample">--</div><span class="unit" id="sampleTime">Waiting for samples</span></div></div>
<section class="panel"><h3>Signal / LED confirmation</h3><p class="muted">Send a ping command. The Pi flashes GPIO17 and returns an acknowledgement.</p><button onclick="sendCommand('ping')">Test signal / LED</button> <span id="pingResult" class="muted"></span></section>
<section class="panel"><h3>Continuous audio recording</h3><p class="muted">Turn ON to record and upload consecutive 30-second WAV files. Each file is saved to Supabase Storage <code>raw/</code> and denoised.</p><div class="row"><button id="audioToggle" onclick="toggleAudioLoop()">Start audio · OFF</button><span id="audioState" class="muted">Audio recording is OFF</span></div></section>
<section class="panel"><h3>Cleaned & raw audio recordings</h3><p class="muted">Fetched directly from Supabase Storage <code>raw/</code>.</p><div id="audioFiles" class="muted">No audio files received yet.</div></section>
<section class="panel"><h3>Recent live readings</h3><div style="overflow:auto"><table><thead><tr><th>Timestamp</th><th>Sample</th><th>Pressure hPa</th><th>BMP °C</th><th>CO₂ ppm</th><th>SCD °C</th><th>Humidity %</th></tr></thead><tbody id="rows"></tbody></table></div><p><a href="/download/jsonl">Download complete sensor history (JSON Lines)</a></p></section>
<section class="panel"><h3>Latest command acknowledgement</h3><div id="ack" class="muted">Waiting for acknowledgement</div></section>
</main><script>
let lastAudioSignature = '';

function renderAudioFiles(files){
  const sig = JSON.stringify(files || []);
  if (sig === lastAudioSignature) return;
  lastAudioSignature = sig;
  const el = document.getElementById('audioFiles');
  if (!files || !files.length){
    el.innerHTML = 'No audio files found in Supabase Storage raw/ yet.';
    return;
  }
  el.innerHTML = files.map(f => {
    let badge = '<span class="badge badge-cleaning">Raw Ready</span>';
    let cleanedControls = '';
    if (f.cleaning_state === 'ready' && f.cleaned_url) {
      badge = '<span class="badge badge-ready">Cleaned Ready</span>';
      cleanedControls = `
        <div><audio controls preload="none" src="${f.cleaned_url}"></audio></div>
        <div style="margin-top:6px">
          <a href="${f.cleaned_url}" target="_blank">Open Cleaned WAV</a> ·
          <a href="${f.cleaned_url}" download>Download Cleaned WAV</a> ·
          <a href="${f.url}" download class="muted">Download Raw</a>
        </div>`;
    } else if (f.cleaning_state === 'cleaning') {
      badge = '<span class="badge badge-cleaning">Denoising...</span>';
      cleanedControls = `<div style="margin-top:6px"><a href="${f.url}" download>Download Raw</a></div>`;
    } else if (f.cleaning_state === 'error') {
      badge = '<span class="badge badge-error">Clean Failed</span>';
      cleanedControls = `<div class="muted" style="margin-top:4px">${f.cleaning_error || ''}</div><div style="margin-top:6px"><a href="${f.url}" download>Download Raw</a></div>`;
    } else {
      cleanedControls = `<div style="margin-top:6px"><a href="${f.url}" target="_blank">Open Raw WAV</a> · <a href="${f.url}" download>Download Raw</a></div>`;
    }
    return `<div style="padding:12px 0;border-bottom:1px solid #2b3d56"><strong>${f.recorded_at || f.filename}</strong> ${badge}${cleanedControls}</div>`;
  }).join('');
}

async function refresh(){try{
  const r=await fetch('/api/live');const d=await r.json();const x=d.latest||{};
  for(const [id,key] of [['pressure','pressure_hPa'],['bmpTemp','bmp_temp'],['co2','co2_ppm'],['scdTemp','scd_temp'],['humidity','humidity']])document.getElementById(id).textContent=x[key]??'--';
  document.getElementById('sample').textContent=x.sample??'--';document.getElementById('sampleTime').textContent=x.timestamp_local||x.timestamp||'Waiting for samples';
  const state=d.status?.state||'waiting_for_device';const online=!!x.timestamp&&((Date.now()-new Date(x.timestamp).getTime())<15000);
  document.getElementById('deviceLed').style.background=online?'#35d07f':'#526078';document.getElementById('deviceState').textContent=online?'LIVE · '+state:state;
  const audioOn=!!d.status?.audio_loop_active || ['audio_loop_active','audio_recording','audio_uploading','audio_uploaded','audio_loop_stopping'].includes(state);
  const audioButton=document.getElementById('audioToggle');
  audioButton.textContent=audioOn?'Stop audio · ON':'Start audio · OFF';
  audioButton.style.background=audioOn?'#d9534f':'#23865b';
  if(state.startsWith('audio_'))document.getElementById('audioState').textContent=d.status.message||state;
  if(state==='audio_received')document.getElementById('audioState').textContent='Uploaded raw WAV to Supabase raw/, running noise reduction: '+(d.status.filename||'');
  if(state==='audio_cleaned')document.getElementById('audioState').textContent='Cleaned audio saved to Supabase cleaned/: '+(d.status.filename||'');
  renderAudioFiles(d.audio_files);
  document.getElementById('rows').innerHTML=(d.history||[]).slice(-15).reverse().map(v=>`<tr><td>${v.timestamp_local||v.timestamp||''}</td><td>${v.sample??''}</td><td>${v.pressure_hPa??''}</td><td>${v.bmp_temp??''}</td><td>${v.co2_ppm??''}</td><td>${v.scd_temp??''}</td><td>${v.humidity??''}</td></tr>`).join('');
  if(d.acks?.length){const a=d.acks[0];document.getElementById('ack').textContent=a.status+': '+(a.message||'')+' · '+(a.command_id||'');}
}catch(e){console.log(e)}}

async function sendCommand(action,extra={}){try{
  const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,...extra})});
  const d=await r.json();document.getElementById('pingResult').textContent=d.message||d.error||'Sent';
}catch(e){document.getElementById('pingResult').textContent='Dashboard request failed';}}

async function toggleAudioLoop(){
  const d=await (await fetch('/api/live')).json();
  const active=!!d.status?.audio_loop_active||['audio_loop_active','audio_recording','audio_uploading','audio_uploaded','audio_loop_stopping'].includes(d.status?.state);
  const action=active?'stop_audio_loop':'start_audio_loop';
  document.getElementById('audioState').textContent=action==='start_audio_loop'?'Starting continuous audio...':'Stopping after current 30-second file uploads...';
  sendCommand(action);
}
setInterval(refresh,1000);refresh();</script></body></html>'''


ANALYSIS_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audio ML Analysis & Plots</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#e8eef9;font:15px Arial,sans-serif}main{max-width:1250px;margin:auto;padding:24px}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}h1{font-size:25px;margin:0}.muted{color:#9aabc4}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:13px;margin:22px 0}.card,.panel{background:#152238;border:1px solid #263954;border-radius:14px;padding:17px}.label{color:#9aabc4;font-size:13px}.value{font-size:27px;font-weight:700;margin-top:9px}.unit{font-size:13px;color:#9aabc4}
.panel{margin:15px 0}button{background:#4d8cff;color:white;border:0;border-radius:9px;padding:11px 16px;font-weight:700;cursor:pointer}.btn-secondary{background:#263954}.row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
a{color:#8db8ff}.badge{display:inline-block;padding:4px 10px;border-radius:12px;font-size:12px;font-weight:700;text-transform:uppercase}
.b-optimal,.b-normal{background:#1e4d38;color:#6ef2b1}.b-warning,.b-moderate{background:#4d3e1e;color:#f7c96e}.b-severe,.b-loud{background:#4d1e24;color:#ff8a96}.b-quiet,.b-soft{background:#1e354d;color:#7ec8ff}
.plot-img{width:100%;border-radius:10px;margin-top:12px;border:1px solid #2b3d56;background:#fff}
audio{height:34px;margin-top:8px;width:100%;max-width:420px}
</style></head><body><main>
<header>
  <div><h1>Collected Audio ML Analysis</h1><div class="muted">Supabase Storage raw/ · 4-Stage Denoising · 40D MFCC · GMM 2D PCA & 1-Second dBFS Plots</div></div>
  <div class="row">
    <button type="button" onclick="triggerAnalysis(false)">Analyze New Files</button>
    <button type="button" class="btn-secondary" onclick="triggerAnalysis(true)">Re-run All</button>
    <a href="/" style="text-decoration:none"><button type="button" class="btn-secondary">← Back to Live Dashboard</button></a>
  </div>
</header>

<div class="grid">
  <div class="card"><div class="label">Collected Audio Files</div><div class="value" id="totalFiles">--</div><span class="unit">WAV files in Supabase raw/</span></div>
  <div class="card"><div class="label">Analyzed</div><div class="value" id="analyzedCount">--</div><span class="unit" id="runState">Starting analysis...</span></div>
  <div class="card"><div class="label">Abnormal Alerts</div><div class="value" id="alertCount">--</div><span class="unit">Warning / Severe clips</span></div>
  <div class="card"><div class="label">Average Level</div><div class="value" id="meanDb">--</div><span class="unit">dBFS across clips</span></div>
</div>

<div id="errorBanner" class="panel" style="display:none;border-color:#c0392b;color:#ff8a96"></div>
<div id="resultsContainer" class="muted">Loading analysis results...</div>

</main><script>
let lastRenderSig = '';

async function triggerAnalysis(force=false){
  try{
    await fetch('/api/analysis/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({force})
    });
    pollAnalysis();
  }catch(e){
    console.log(e);
  }
}

function renderResults(items){
  const sig = JSON.stringify(items || []);
  if (sig === lastRenderSig) return;
  lastRenderSig = sig;
  const container = document.getElementById('resultsContainer');
  if (!items || !items.length){
    container.innerHTML = '<section class="panel"><div class="muted">No .wav files found in Supabase Storage <code>raw/</code> yet. Record audio on the Live Dashboard first.</div></section>';
    return;
  }
  container.innerHTML = items.map(item => {
    if (item.status !== 'ok') {
      return `<section class="panel">
        <div class="row" style="justify-content:space-between">
          <strong>${item.recorded_at || item.filename} (${item.filename})</strong>
          <span class="badge b-severe">Error</span>
        </div>
        <div class="muted" style="margin-top:8px">${item.error || 'Processing failed'}</div>
      </section>`;
    }
    const cd = item.cluster_distribution || {};
    const dbd = item.db_cluster_distribution || {};
    return `<section class="panel">
      <div class="row" style="justify-content:space-between">
        <div>
          <strong style="font-size:17px">${item.recorded_at || item.filename}</strong>
          <div class="muted" style="font-size:13px">${item.filename} · Analyzed ${item.analyzed_at || ''}</div>
        </div>
        <div class="row">
          <span class="badge b-${item.gmm_cluster}">GMM: ${item.gmm_cluster}</span>
          <span class="badge b-${item.db_cluster}">Level: ${item.db_cluster} (${item.avg_db} dBFS)</span>
        </div>
      </div>
      <div class="muted" style="margin-top:10px;font-size:13px">
        GMM Avg Score: <strong>${item.gmm_avg_score}</strong> ·
        Anomalous: <strong>${item.anomalous_seconds}s / ${item.total_seconds}s</strong> ·
        Peak: <strong>${item.max_db} dBFS</strong> ·
        Bands (Opt/Norm/Warn/Sev): <strong>${cd.optimal_sec||0}/${cd.normal_sec||0}/${cd.warning_sec||0}/${cd.severe_sec||0}s</strong> ·
        dB Bands (Q/S/M/L): <strong>${dbd.quiet_sec||0}/${dbd.soft_sec||0}/${dbd.moderate_sec||0}/${dbd.loud_sec||0}s</strong>
      </div>
      <div class="row" style="margin-top:8px">
        <audio controls preload="none" src="${item.cleaned_url}"></audio>
        <a href="${item.cleaned_url}" download>Download Cleaned WAV</a> ·
        <a href="${item.raw_url}" download class="muted">Download Raw</a> ·
        <a href="${item.plot_url}" target="_blank">Open Plot PNG</a>
      </div>
      <img class="plot-img" src="${item.plot_url}?t=${encodeURIComponent(item.analyzed_at || '')}" alt="GMM and dBFS plot for ${item.filename}" loading="lazy">
    </section>`;
  }).join('');
}

async function pollAnalysis(){
  try{
    const r = await fetch('/api/analysis/status');
    const d = await r.json();
    const st = d.state || {};
    const items = d.results || [];
    const okItems = items.filter(x => x.status === 'ok');

    document.getElementById('totalFiles').textContent = d.total_files ?? 0;
    document.getElementById('analyzedCount').textContent = okItems.length;
    document.getElementById('alertCount').textContent = okItems.filter(x => x.alert).length;
    if (okItems.length) {
      const avg = okItems.reduce((acc, x) => acc + (x.avg_db || 0), 0) / okItems.length;
      document.getElementById('meanDb').textContent = avg.toFixed(1);
    } else {
      document.getElementById('meanDb').textContent = '--';
    }

    if (st.running) {
      document.getElementById('runState').textContent = `Processing ${st.processed}/${st.total}: ${st.current_file || ''}`;
    } else if (st.last_run) {
      document.getElementById('runState').textContent = `Up to date (${st.last_run})`;
    } else {
      document.getElementById('runState').textContent = 'Idle';
    }

    const errEl = document.getElementById('errorBanner');
    if (st.error) {
      errEl.style.display = 'block';
      errEl.textContent = st.error;
    } else {
      errEl.style.display = 'none';
    }

    renderResults(items);
  }catch(e){
    console.log(e);
  }
}

triggerAnalysis(false);
setInterval(pollAnalysis, 1500);
</script></body></html>'''


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "supabase_bucket": AUDIO_BUCKET,
        "models_loaded": model_load_error is None,
    })


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/analysis")
def analysis_page():
    return render_template_string(ANALYSIS_HTML)


@app.route("/api/analysis/run", methods=["POST"])
def api_analysis_run():
    body = request.get_json(silent=True) or {}
    force = bool(body.get("force", False))
    with lock:
        if analysis_state["running"]:
            return jsonify({"status": "already_running", "state": dict(analysis_state)})
        analysis_state["running"] = True
        analysis_state["error"] = None
    threading.Thread(target=run_batch_analysis_worker, args=(force,), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/analysis/status")
def api_analysis_status():
    try:
        items_meta = list_raw_wav_files(limit=200, use_cache=True)
        results = []
        for item in items_meta:
            res = item.get("analysis_result")
            if isinstance(res, dict) and res:
                results.append(res)
        with lock:
            state_copy = dict(analysis_state)
        return jsonify({
            "total_files": len(items_meta),
            "state": state_copy,
            "results": results,
        })
    except Exception as exc:
        with lock:
            state_copy = dict(analysis_state)
            state_copy["error"] = str(exc)
        return jsonify({"total_files": 0, "state": state_copy, "results": []}), 500


@app.route("/api/live")
def api_live():
    try:
        items_meta = list_raw_wav_files(limit=200, use_cache=True)
    except Exception as exc:
        logging.warning("Failed to list raw/ audio files: %s", exc)
        items_meta = []

    audio_files = [
        {
            "filename": r.get("filename"),
            "recorded_at": r.get("recorded_at"),
            "url": r.get("raw_url"),
            "cleaned_url": r.get("cleaned_url"),
            "cleaning_state": r.get("cleaning_state") or "pending",
            "cleaning_error": r.get("cleaning_error"),
        }
        for r in items_meta
    ]

    with lock:
        return jsonify({
            "latest": latest,
            "history": list(history),
            "status": status,
            "acks": list(acks),
            "audio_files": audio_files,
        })


@app.route("/api/command", methods=["POST"])
def api_command():
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    if action not in ("ping", "start_audio_loop", "stop_audio_loop"):
        return jsonify({"error": "Unsupported action"}), 400
    command = {
        "command_id": uuid.uuid4().hex,
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    info = mqtt_client.publish(TOPIC_COMMAND, json.dumps(command), qos=1)
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        return jsonify({"error": "MQTT publish failed"}), 503
    return jsonify({"message": "Command published", "command_id": command["command_id"]})


@app.route("/download/jsonl")
def download_jsonl():
    if not os.path.exists(JSONL_PATH):
        open(JSONL_PATH, "a").close()
    return send_from_directory(DATA_DIR, "sensor_data.jsonl", as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Open http://127.0.0.1:{port} in your browser")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)