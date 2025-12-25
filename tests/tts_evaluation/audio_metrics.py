"""
Utility: Audio Quality Metrics

Provides functions for calculating TTS quality metrics:
- MOS prediction (NISQA)
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- Prosody analysis (F0, formants, speech rate)
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def calculate_mos_nisqa(audio_path: str) -> Optional[float]:
    """
    Calculate MOS (Mean Opinion Score) using NISQA model.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        MOS score (1.0-5.0) or None if error
    """
    try:
        from nisqa.NISQA_lib import nisqaModel, predict_mos
        
        # Load NISQA model
        model = nisqaModel()
        
        # Predict MOS
        mos_score = predict_mos(model, audio_path)
        
        return float(mos_score)
    
    except Exception as e:
        print(f"Warning: NISQA MOS calculation failed: {e}")
        return None

def calculate_pesq(reference_path: str, degraded_path: str, sr: int = 16000) -> Optional[float]:
    """
    Calculate PESQ score between reference and degraded audio.
    
    Args:
        reference_path: Path to reference (human) audio
        degraded_path: Path to degraded (TTS) audio
        sr: Sample rate (8000 or 16000 Hz)
        
    Returns:
        PESQ score (-0.5 to 4.5) or None if error
    """
    try:
        from pypesq import pesq
        
        # Load audio files
        ref, ref_sr = librosa.load(reference_path, sr=sr)
        deg, deg_sr = librosa.load(degraded_path, sr=sr)
        
        # Ensure same length
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        
        # Calculate PESQ (mode='wb' for wideband 16kHz, 'nb' for narrowband 8kHz)
        mode = 'wb' if sr == 16000 else 'nb'
        pesq_score = pesq(ref, deg, sr)
        
        return float(pesq_score)
    
    except Exception as e:
        print(f"Warning: PESQ calculation failed: {e}")
        return None

def calculate_stoi(reference_path: str, degraded_path: str, sr: int = 16000) -> Optional[float]:
    """
    Calculate STOI (Short-Time Objective Intelligibility) score.
    
    Args:
        reference_path: Path to reference (human) audio
        degraded_path: Path to degraded (TTS) audio
        sr: Sample rate
        
    Returns:
        STOI score (0.0-1.0) or None if error
    """
    try:
        from pystoi import stoi
        
        # Load audio files
        ref, _ = librosa.load(reference_path, sr=sr)
        deg, _ = librosa.load(degraded_path, sr=sr)
        
        # Ensure same length
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        
        # Calculate STOI
        stoi_score = stoi(ref, deg, sr, extended=False)
        
        return float(stoi_score)
    
    except Exception as e:
        print(f"Warning: STOI calculation failed: {e}")
        return None

def calculate_wer_with_whisper(audio_path: str, reference_text: str, model_name: str = "medium") -> Optional[float]:
    """
    Calculate WER (Word Error Rate) using Whisper ASR round-trip.
    
    Args:
        audio_path: Path to TTS audio file
        reference_text: Original text
        model_name: Whisper model ('tiny', 'base', 'small', 'medium')
        
    Returns:
        WER as percentage (0.0-100.0) or None if error
    """
    try:
        import whisper
        from jiwer import wer
        
        # Load Whisper model
        model = whisper.load_model(model_name)
        
        # Transcribe audio
        result = model.transcribe(audio_path)
        hypothesis_text = result["text"]
        
        # Calculate WER
        wer_score = wer(reference_text, hypothesis_text) * 100
        
        return float(wer_score)
    
    except Exception as e:
        print(f"Warning: WER calculation failed: {e}")
        return None

def extract_prosody_features(audio_path: str) -> Dict[str, float]:
    """
    Extract prosody features from audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary of prosody features
    """
    try:
        import parselmouth
        from parselmouth.praat import call
        
        # Load audio with Parselmouth (Praat)
        sound = parselmouth.Sound(audio_path)
        
        # Extract pitch (F0)
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
        std_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        min_f0 = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        max_f0 = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        
        # Speech rate (syllable nuclei)
        intensity = call(sound, "To Intensity", 75, 0)
        mean_intensity = call(intensity, "Get mean", 0, 0)
        
        # Duration
        duration = call(sound, "Get total duration")
        
        # Load with librosa for additional features
        y, sr = librosa.load(audio_path)
        
        # Zero-crossing rate (voicing stability)
        zcr = librosa.feature.zero_crossing_rate(y)
        mean_zcr = float(np.mean(zcr))
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_spectral_centroid = float(np.mean(spectral_centroid))
        
        # Energy
        rms = librosa.feature.rms(y=y)
        mean_rms = float(np.mean(rms))
        
        prosody_features = {
            "mean_f0": float(mean_f0) if not np.isnan(mean_f0) else 0.0,
            "std_f0": float(std_f0) if not np.isnan(std_f0) else 0.0,
            "min_f0": float(min_f0) if not np.isnan(min_f0) else 0.0,
            "max_f0": float(max_f0) if not np.isnan(max_f0) else 0.0,
            "f0_range": float(max_f0 - min_f0) if not (np.isnan(max_f0) or np.isnan(min_f0)) else 0.0,
            "mean_intensity": float(mean_intensity),
            "duration": float(duration),
            "mean_zcr": mean_zcr,
            "mean_spectral_centroid": mean_spectral_centroid,
            "mean_rms": mean_rms
        }
        
        return prosody_features
    
    except Exception as e:
        print(f"Warning: Prosody extraction failed: {e}")
        return {
            "mean_f0": 0.0,
            "std_f0": 0.0,
            "min_f0": 0.0,
            "max_f0": 0.0,
            "f0_range": 0.0,
            "mean_intensity": 0.0,
            "duration": 0.0,
            "mean_zcr": 0.0,
            "mean_spectral_centroid": 0.0,
            "mean_rms": 0.0
        }

def calculate_all_metrics(
    audio_path: str,
    reference_text: str,
    reference_audio_path: Optional[str] = None
) -> Dict:
    """
    Calculate all quality metrics for a TTS audio file.
    
    Args:
        audio_path: Path to TTS audio
        reference_text: Original text
        reference_audio_path: Optional path to human reference audio
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # MOS prediction (no reference needed)
    metrics["mos_nisqa"] = calculate_mos_nisqa(audio_path)
    
    # PESQ and STOI (require reference audio)
    if reference_audio_path:
        metrics["pesq"] = calculate_pesq(reference_audio_path, audio_path)
        metrics["stoi"] = calculate_stoi(reference_audio_path, audio_path)
    else:
        metrics["pesq"] = None
        metrics["stoi"] = None
    
    # WER via Whisper (no reference audio needed, uses text)
    metrics["wer"] = calculate_wer_with_whisper(audio_path, reference_text)
    
    # Prosody features
    prosody = extract_prosody_features(audio_path)
    metrics.update(prosody)
    
    return metrics

if __name__ == "__main__":
    # Test with example audio
    print("Audio Metrics Utility - Test Mode")
    print("="*60)
    print("This utility provides quality metric calculations.")
    print("Use from other scripts, not standalone.")
    print("="*60)
