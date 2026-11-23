#!/usr/bin/env python3
"""Render API: 40D MFCC extraction, GMM + dBFS 30s window analysis, and thread-safe 2D PCA plotting."""
import os
import io
import gc
import base64
import tempfile
import threading
from datetime import datetime, timedelta

import numpy as np
import librosa
import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch
from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Acoustic & Threshold Configuration
AUDIO_RATE = 16000
ANOMALY_THRESHOLD = -85.00
MIN_ANOMALOUS_SECONDS = 2

DB_CALIBRATION_OFFSET = 0.0
DB_QUIET_MAX = -45.0        # < -45 dBFS  -> quiet
DB_SOFT_MAX = -30.0         # -45 to -30  -> soft
DB_MODERATE_MAX = -18.0     # -30 to -18  -> moderate
                            # >= -18 dBFS -> loud

# Load trained models once at worker startup
scaler = joblib.load(os.path.join(BASE_DIR, "pi_scaler.joblib"))
gmm = joblib.load(os.path.join(BASE_DIR, "pi_gmm.joblib"))
pca = joblib.load(os.path.join(BASE_DIR, "pi_pca.joblib"))

plot_lock = threading.Lock()
app = Flask(__name__)


def render_plot_b64(X_pca, scores, db_values, filename):
    """Render side-by-side 2D PCA GMM + 1-second dBFS plot to a Base64 PNG (thread-safe)."""
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

        # Guarantee strictly increasing contour levels even if entire grid is normal or anomalous
        g_min = float(grid_scores.min())
        g_max = float(grid_scores.max())
        z_low = min(g_min - 1.0, ANOMALY_THRESHOLD - 1.0)
        z_high = max(g_max + 1.0, ANOMALY_THRESHOLD + 1.0)

        ax1.contourf(
            xx, yy, grid_scores,
            levels=[z_low, ANOMALY_THRESHOLD, z_high],
            colors=["#ffcccc", "#ccffcc"],
            alpha=0.55
        )
        if g_min < ANOMALY_THRESHOLD < g_max:
            ax1.contour(
                xx, yy, grid_scores,
                levels=[ANOMALY_THRESHOLD],
                colors="#cc0000",
                linewidths=2.0,
                linestyles="dashed"
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
            c=gmm_colors, edgecolors="black", linewidths=0.5, s=55, alpha=0.9
        )
        ax1.text(
            x_max - 1.5, y_max - 1.5, f"NORMAL (≥ {ANOMALY_THRESHOLD})",
            color="darkgreen", fontsize=9.5, fontweight="bold", ha="right", va="top"
        )
        ax1.text(
            x_max - 1.5, y_min + 1.5, f"ANOMALY (< {ANOMALY_THRESHOLD})",
            color="darkred", fontsize=9.5, fontweight="bold", ha="right", va="bottom"
        )
        ax1.legend(
            handles=[
                Patch(facecolor="#2ecc71", edgecolor="black", label=f"Optimal: {int(np.sum(scores >= -60.0))}s"),
                Patch(facecolor="#27ae60", edgecolor="black", label=f"Normal: {int(np.sum((scores < -60.0) & (scores >= ANOMALY_THRESHOLD)))}s"),
                Patch(facecolor="#f39c12", edgecolor="black", label=f"Warning: {int(np.sum((scores < ANOMALY_THRESHOLD) & (scores >= -110.0)))}s"),
                Patch(facecolor="#c0392b", edgecolor="black", label=f"Severe: {int(np.sum(scores < -110.0))}s"),
            ],
            loc="upper left", fontsize=8, framealpha=0.9
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
            bottom=-80.0, color=db_colors, edgecolor="black", linewidth=0.5
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
            loc="upper right", fontsize=8, framealpha=0.9
        )

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        fig.clear()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"status": "error", "message": "Missing 'audio' file field"}), 400

    audio_file = request.files["audio"]
    filename = os.path.basename(audio_file.filename or "window.wav")
    window_start = request.form.get("window_start") or datetime.now().isoformat(timespec="seconds")

    try:
        window_seconds = max(1, int(request.form.get("window_seconds", 30)))
    except (TypeError, ValueError):
        window_seconds = 30

    try:
        window_end = (
            datetime.fromisoformat(window_start) + timedelta(seconds=window_seconds)
        ).isoformat(timespec="seconds")
    except ValueError:
        window_end = window_start

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        if os.path.getsize(tmp_path) == 0:
            return jsonify({"status": "error", "message": "Uploaded audio file is empty"}), 400

        y, sr = librosa.load(tmp_path, sr=AUDIO_RATE)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
        frames_per_sec = int(sr / 512)
        samples_per_sec = int(sr)

        features_list = []
        db_values = []
        sec_idx = 0
        for i in range(0, mfccs.shape[1], frames_per_sec):
            chunk = mfccs[:, i:i + frames_per_sec]
            audio_chunk = y[sec_idx * samples_per_sec:(sec_idx + 1) * samples_per_sec]
            if chunk.shape[1] == frames_per_sec and len(audio_chunk) == samples_per_sec:
                features_list.append(np.hstack((np.mean(chunk, axis=1), np.std(chunk, axis=1))))
                rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                db_values.append(float(20.0 * np.log10(rms + 1e-10) + DB_CALIBRATION_OFFSET))
                sec_idx += 1

        if not features_list:
            return jsonify({"status": "error", "message": "Audio too short (< 1 second)"}), 400

        X = np.array(features_list)
        X_scaled = scaler.transform(X)
        scores = gmm.score_samples(X_scaled)
        X_pca = pca.transform(X_scaled)

        # GMM 4-band frame counts
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

        # Decibel (dBFS) window metrics & 4-band frame counts
        overall_rms = float(np.sqrt(np.mean(y ** 2)))
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

        plot_b64 = render_plot_b64(X_pca, scores, db_values, filename)

        result = {
            "status": "ok",
            "filename": filename,
            "window_start": window_start,
            "window_end": window_end,
            "window_seconds": window_seconds,
            "alert": bool(is_abnormal),
            "class": "abnormal_sound" if is_abnormal else "normal",
            "gmm_cluster": gmm_cluster,
            "gmm_avg_score": round(avg_score, 2),
            "confidence": float(round(abs(lowest_score), 2)),
            "anomalous_seconds": anomalous_seconds,
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
            "plot_png_b64": plot_b64,
        }

        del y, mfccs, features_list, X, X_scaled, X_pca, scores, db_arr
        gc.collect()
        return jsonify(result)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)