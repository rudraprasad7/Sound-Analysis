import os
import glob
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, iirnotch, lfilter
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from normalize_audio import peak_normalize
from convert_audio import ensure_optimal_samplerate
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAMNET_PATH = os.path.join(BASE_DIR, "YamNet.tflite")

tflite = None
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from ai_edge_litert import interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            tflite = None

_YAMNET_INTERPRETER = None


def _get_yamnet_interpreter(model_path):
    global _YAMNET_INTERPRETER
    if _YAMNET_INTERPRETER is not None:
        return _YAMNET_INTERPRETER
    if tflite is None or not os.path.exists(model_path):
        return None
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    _YAMNET_INTERPRETER = interpreter
    return _YAMNET_INTERPRETER


def silence_human_speech_yamnet(y, sr, model_path=DEFAULT_YAMNET_PATH):
    """Hardware-safe AI Voice Radar using YAMNet TFLite (cached per worker)."""
    if sr != 16000:
        print(f"  -> YAMNet strictly requires 16000Hz (Found {sr}Hz). Bypassing.")
        return y

    try:
        interpreter = _get_yamnet_interpreter(model_path)
    except Exception as e:
        print(f"  -> TFLite Engine failure: {e}. Bypassing.")
        return y

    if interpreter is None:
        print("  -> YAMNet model or TFLite runtime not found. Bypassing voice suppression.")
        return y

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    y_norm = np.clip(y, -1.0, 1.0).astype(np.float32)
    gain_mask = np.ones_like(y)

    frame_length = 15600
    step_size = 7800
    speech_detected = False
    cooldown_timer = 0

    for i in range(0, len(y_norm), step_size):
        frame = y_norm[i:i + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode="constant")

        interpreter.set_tensor(input_details["index"], frame)
        interpreter.invoke()

        scores = interpreter.get_tensor(output_details["index"])
        mean_scores = np.mean(scores.reshape(-1, scores.shape[-1]), axis=0)
        if mean_scores[0] > 0.15:
            speech_detected = True
            cooldown_timer = i + frame_length + int(sr * 0.8)
            actual_start = max(0, i - int(sr * 0.30))
            actual_end = min(i + frame_length, len(y))
            gain_mask[actual_start:actual_end] = 0.0
        elif i < cooldown_timer:
            actual_end = min(i + frame_length, len(y))
            gain_mask[i:actual_end] = 0.0

    if speech_detected:
        print("  -> Human conversation bridged & muted.")
    smooth_mask = gaussian_filter1d(gain_mask, sigma=int(sr * 0.2))
    return y * smooth_mask
            
    # Apply a 200ms Gaussian crossfade to prevent digital clicking when the volume drops
    # smooth_mask = gaussian_filter1d(gain_mask, sigma=int(sr * 0.2))
    
    # return y * smooth_mask

def apply_high_pass_filter(data, sr, cutoff=150.0):
    sos = butter(4, cutoff, btype='high', fs=sr, output='sos')
    return sosfilt(sos, data)

def apply_gentle_treble_cut(data, sr, cutoff=5500.0):
    sos = butter(1, cutoff, btype='low', fs=sr, output='sos')
    return sosfilt(sos, data)

def apply_notch_filter(data, sr, freq, q=10):
    digital_freq = freq / (sr / 2.0)
    if digital_freq >= 1.0:
        return data
    b, a = iirnotch(digital_freq, q)
    return lfilter(b, a, data)


def apply_adaptive_spectral_subtraction(y, alpha=1.2, beta=0.08):
    # Halved the FFT window for percussive audio. 
    # This prevents temporal smearing and keeps the transient attack razor-sharp.
    n_fft = 1024
    hop_length = 128
    
    S_full, phase = librosa.magphase(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    P_full = S_full ** 2
    
    P_noise = np.median(P_full, axis=1, keepdims=True)
    P_noise = uniform_filter1d(P_noise, size=51, axis=1) 
    
    # Raised beta to 0.08 so we don't create a digital vacuum that sounds unnatural
    spectral_floor = beta * P_noise
    P_clean = np.maximum(P_full - (alpha * P_noise), spectral_floor)
    
    S_clean = np.sqrt(P_clean)
    y_clean = librosa.istft(S_clean * phase, hop_length=hop_length)
    
    return y_clean

"""
def smart_isolate_clucks(y, sr):
    if np.max(np.abs(y)) == 0:
        return y
        
    db_profile = librosa.amplitude_to_db(np.abs(y), ref=np.max)
    hist, bin_edges = np.histogram(db_profile, bins=100)
    measured_noise_floor = bin_edges[np.argmax(hist)]
    
    cluck_db = np.percentile(db_profile, 85)
    dynamic_range = cluck_db - measured_noise_floor
    separation_threshold = measured_noise_floor + (dynamic_range * 0.20)

    is_background = (db_profile < separation_threshold).astype(float)
    smooth_gate = gaussian_filter1d(is_background, sigma=int(sr * 0.6))
    
    # 20% floor to mask the hiss contrast
    gain_curve = 1.0 - (smooth_gate * 0.65) 
    return y * gain_curve
"""

def clean_farm_audio_ultimate(file_path, output_path):
    print(f"\nProcessing: {os.path.basename(file_path)}")
    
    y_raw, sr = ensure_optimal_samplerate(file_path, max_sr=16000)
    y_raw = y_raw.astype(np.float32)

    # 1. MUTE VOICES FIRST (Using hardware-safe YAMNet TFLite)
    y_clean = silence_human_speech_yamnet(y_raw, sr)

    # 2. MECHANICAL CLEANUP
    y_clean = apply_high_pass_filter(y_clean, sr, cutoff=150.0)
    
    target_whirs = [100.0, 580.0, 620.0, 660.0]
    for whir in target_whirs:
        y_clean = apply_notch_filter(y_clean, sr, freq=whir, q=8)

    # 3. DYNAMIC SNR SUBTRACTION 
    y_sq = y_clean ** 2
    smoothed_sq = uniform_filter1d(y_sq, size=2048)
    rms = np.sqrt(np.maximum(0.0, smoothed_sq))
    
    cluck_energy = np.percentile(rms, 98)  
    room_energy = np.percentile(rms, 5)   
    
    if room_energy < 0.0005:
        noise_ratio = 0.0
        print("  -> File is relatively silent. Locking to baseline subtraction.")
    else:
        noise_ratio = room_energy / (cluck_energy + 1e-6)
    
    intensity_scale = np.clip(noise_ratio * 3.0, 0.0, 1.0)
    auto_alpha = 1.2 + (1.0 * intensity_scale)
        
    y_clean = apply_adaptive_spectral_subtraction(y_clean, alpha=auto_alpha, beta=0.10)
    
    # 4. SMART NORMALIZATION (Target peak safely expanded for ML models)
    y_final = peak_normalize(y_clean, target_peak=0.95, sr=sr, dynamic=True)
    
    sf.write(output_path, y_final, sr)
    print(f"  -> Saved clean file to: {output_path}")

def main():
    input_dir = "test_audio_2/raw"
    output_dir = "test_audio_2/cleaned"
    os.makedirs(output_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    if not audio_files:
        print(f"No .wav files found in {input_dir}.")
        return
        
    print(f"Found {len(audio_files)} files. Initializing Bioacoustic Pipeline...\n")
    
    for file_path in audio_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        clean_farm_audio_ultimate(file_path, output_path)
        
    print("\nBatch processing complete!")

if __name__ == "__main__":
    main()