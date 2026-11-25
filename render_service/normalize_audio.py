import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d

def peak_normalize(y, target_peak=0.95, sr=None, dynamic=True):
    print("Normalizing audio volume...")
    
    max_peak = np.max(np.abs(y))
    if max_peak == 0:
        return y
        
    if not dynamic or sr is None:
        scaling_factor = target_peak / max_peak
        return y * scaling_factor

    db_profile = librosa.amplitude_to_db(np.abs(y), ref=np.max)
    
    hist, bin_edges = np.histogram(db_profile, bins=100)
    measured_noise_floor = bin_edges[np.argmax(hist)]
    
    cluck_db = np.percentile(db_profile, 85)
    dynamic_range = cluck_db - measured_noise_floor
    
    # Lowered threshold to 5% so the gate catches the faint, surrounding clucks
    separation_threshold = measured_noise_floor + (dynamic_range * 0.05)
    
    is_loud = (db_profile > separation_threshold).astype(float)
    
    # REVERTED: Removed the 300ms hold. Strictly using the tight 50ms envelope 
    # to maintain the crisp attack while snapping shut instantly to block the hiss.
    smooth_fade = gaussian_filter1d(is_loud, sigma=int(sr * 0.1))
    
    cluck_regions = np.abs(y)[is_loud > 0]
    if len(cluck_regions) > 0:
        average_cluck_peak = np.percentile(cluck_regions, 90)
        
        # 1. STRONGER GRAVITY CURVE
        # 0.55 applies a stronger fractional exponent, pulling the volume higher across the board.
        adaptive_target = average_cluck_peak ** 0.55
        
        # Raised the internal safety ceiling to 95% to allow for the extra volume
        adaptive_target = min(0.95, adaptive_target)
        
        measured_boost = adaptive_target / (average_cluck_peak + 1e-6)
        measured_boost = max(1.0, measured_boost)
    else:
        average_cluck_peak = 0.0
        adaptive_target = 0.0
        measured_boost = 1.0

    print(f"  -> File Dynamic Range: {dynamic_range:.1f} dB")
    print(f"  -> Original Cluck Peak: {average_cluck_peak:.3f}")
    print(f"  -> Curved Target Volume: {adaptive_target:.3f}")
    print(f"  -> Applied Relative Boost: {measured_boost:.2f}x")

    dynamic_multiplier = 1.0 + ((measured_boost - 1.0) * smooth_fade)
    y_smart = y * dynamic_multiplier
    
    # Transparent digital peak scaler ensures zero shrill distortion
    final_peak = np.max(np.abs(y_smart))
    if final_peak > target_peak:
        y_smart = y_smart * (target_peak / final_peak)
        
    return y_smart