import librosa

def ensure_optimal_samplerate(file_path, max_sr=16000):
    """
    Scans the file's native sample rate before loading. 
    Only downsamples if the rate is strictly greater than max_sr.
    """
    native_sr = librosa.get_samplerate(file_path)
    
    if native_sr > max_sr:
        print(f"  -> High resolution detected ({native_sr}Hz). Downsampling to {max_sr}Hz for Pi optimization...")
        y, sr = librosa.load(file_path, sr=max_sr)
    else:
        y, sr = librosa.load(file_path, sr=None)
        
    return y, sr