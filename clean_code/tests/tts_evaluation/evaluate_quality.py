"""
Script 3: Evaluate TTS Quality (Simplified)

Calculates quality metrics for generated TTS audio:
- Audio duration and speech rate
- Basic prosody features (pitch, energy)
- Generation time analysis
- RTF (Real-Time Factor)

Note: Advanced metrics (MOS, WER) require additional dependencies
"""

import csv
import json
import wave
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
        return duration
    except:
        # Fallback to librosa
        y, sr = librosa.load(audio_path)
        return len(y) / sr

def extract_basic_features(audio_path: str) -> dict:
    """Extract basic audio features."""
    try:
        # Load audio
        y, sr = librosa.load(audio_path)
        
        # Duration
        duration = len(y) / sr
        
        # Pitch (F0) estimation
        f0 = librosa.yin(y, fmin=75, fmax=600, sr=sr)
        f0_clean = f0[f0 > 0]  # Remove unvoiced frames
        
        if len(f0_clean) > 0:
            mean_f0 = float(np.mean(f0_clean))
            std_f0 = float(np.std(f0_clean))
            min_f0 = float(np.min(f0_clean))
            max_f0 = float(np.max(f0_clean))
        else:
            mean_f0 = std_f0 = min_f0 = max_f0 = 0.0
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=y)
        mean_rms = float(np.mean(rms))
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        mean_zcr = float(np.mean(zcr))
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_spectral_centroid = float(np.mean(spectral_centroid))
        
        return {
            'duration': duration,
            'mean_f0': mean_f0,
            'std_f0': std_f0,
            'min_f0': min_f0,
            'max_f0': max_f0,
            'f0_range': max_f0 - min_f0,
            'mean_rms': mean_rms,
            'mean_zcr': mean_zcr,
            'mean_spectral_centroid': mean_spectral_centroid
        }
    except Exception as e:
        print(f"Warning: Feature extraction failed for {audio_path}: {e}")
        return {
            'duration': 0.0,
            'mean_f0': 0.0,
            'std_f0': 0.0,
            'min_f0': 0.0,
            'max_f0': 0.0,
            'f0_range': 0.0,
            'mean_rms': 0.0,
            'mean_zcr': 0.0,
            'mean_spectral_centroid': 0.0
        }

def calculate_speech_rate(word_count: int, duration: float) -> float:
    """Calculate words per minute."""
    if duration > 0:
        return (word_count / duration) * 60
    return 0.0

def evaluate_audio_files(metadata_path: Path, output_dir: Path):
    """Evaluate all generated audio files."""
    
    # Load generation metadata
    print(f"Loading metadata from {metadata_path}...")
    metadata = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)
    
    print(f"✓ Loaded {len(metadata)} audio file entries\n")
    
    # Evaluate each file
    quality_metrics = []
    
    print("Evaluating audio quality...")
    for entry in tqdm(metadata, desc="Processing files"):
        audio_path = Path(entry['audio_path'])
        
        # Skip if file doesn't exist
        if not audio_path.exists():
            continue
        
        # Extract features
        features = extract_basic_features(str(audio_path))
        
        # Calculate RTF (Real-Time Factor)
        generation_time = float(entry['generation_time'])
        rtf = generation_time / features['duration'] if features['duration'] > 0 else 0.0
        
        # Calculate speech rate
        word_count = int(entry['word_count'])
        speech_rate_wpm = calculate_speech_rate(word_count, features['duration'])
        
        # Compile metrics
        metrics_entry = {
            'response_id': entry['response_id'],
            'voice': entry['voice'],
            'speaking_rate': entry['speaking_rate'],
            'category': entry['category'],
            'text': entry['text'],
            'word_count': word_count,
            'generation_time': generation_time,
            'audio_duration': features['duration'],
            'rtf': rtf,
            'speech_rate_wpm': speech_rate_wpm,
            'mean_f0': features['mean_f0'],
            'std_f0': features['std_f0'],
            'f0_range': features['f0_range'],
            'mean_rms': features['mean_rms'],
            'mean_zcr': features['mean_zcr'],
            'mean_spectral_centroid': features['mean_spectral_centroid'],
        }
        
        quality_metrics.append(metrics_entry)
    
    return quality_metrics

def save_quality_metrics(metrics: list, output_dir: Path):
    """Save quality metrics to CSV and generate statistics."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    csv_path = output_dir / "quality_metrics.csv"
    if metrics:
        fieldnames = metrics[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)
        print(f"\n✓ Saved quality metrics to {csv_path}")
    
    # Calculate statistics
    if len(metrics) > 0:
        stats = {
            'total_files': len(metrics),
            'mean_duration': np.mean([m['audio_duration'] for m in metrics]),
            'mean_generation_time': np.mean([m['generation_time'] for m in metrics]),
            'mean_rtf': np.mean([m['rtf'] for m in metrics]),
            'mean_speech_rate': np.mean([m['speech_rate_wpm'] for m in metrics]),
            'mean_f0': np.mean([m['mean_f0'] for m in metrics if m['mean_f0'] > 0]),
            'mean_f0_range': np.mean([m['f0_range'] for m in metrics if m['f0_range'] > 0]),
        }
        
        # By voice
        voices = set(m['voice'] for m in metrics)
        stats['by_voice'] = {}
        for voice in voices:
            voice_metrics = [m for m in metrics if m['voice'] == voice]
            stats['by_voice'][voice] = {
                'count': len(voice_metrics),
                'mean_duration': np.mean([m['audio_duration'] for m in voice_metrics]),
                'mean_rtf': np.mean([m['rtf'] for m in voice_metrics]),
                'mean_speech_rate': np.mean([m['speech_rate_wpm'] for m in voice_metrics]),
                'mean_f0': np.mean([m['mean_f0'] for m in voice_metrics if m['mean_f0'] > 0]),
            }
        
        # By speaking rate
        rates = set(m['speaking_rate'] for m in metrics)
        stats['by_rate'] = {}
        for rate in rates:
            rate_metrics = [m for m in metrics if m['speaking_rate'] == rate]
            stats['by_rate'][rate] = {
                'count': len(rate_metrics),
                'mean_duration': np.mean([m['audio_duration'] for m in rate_metrics]),
                'mean_speech_rate': np.mean([m['speech_rate_wpm'] for m in rate_metrics]),
                'mean_rtf': np.mean([m['rtf'] for m in rate_metrics]),
            }
        
        # Save statistics
        stats_path = output_dir / "quality_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics to {stats_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("QUALITY EVALUATION SUMMARY")
        print("="*70)
        print(f"Total Files Evaluated: {stats['total_files']:,}")
        print(f"\nOverall Metrics:")
        print(f"  Mean Duration:        {stats['mean_duration']:.2f}s")
        print(f"  Mean Generation Time: {stats['mean_generation_time']:.3f}s")
        print(f"  Mean RTF:             {stats['mean_rtf']:.3f}")
        print(f"  Mean Speech Rate:     {stats['mean_speech_rate']:.1f} WPM")
        print(f"  Mean Pitch (F0):      {stats['mean_f0']:.1f} Hz")
        
        print(f"\nBy Voice Profile:")
        for voice, voice_stats in stats['by_voice'].items():
            print(f"  {voice}:")
            print(f"    Speech Rate: {voice_stats['mean_speech_rate']:.1f} WPM")
            print(f"    RTF:         {voice_stats['mean_rtf']:.3f}")
            print(f"    Pitch:       {voice_stats['mean_f0']:.1f} Hz")
        
        print(f"\nBy Speaking Rate:")
        for rate, rate_stats in stats['by_rate'].items():
            print(f"  {rate}:")
            print(f"    Speech Rate: {rate_stats['mean_speech_rate']:.1f} WPM")
            print(f"    RTF:         {rate_stats['mean_rtf']:.3f}")
        
        print("="*70 + "\n")

def main():
    """Main execution."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    metadata_path = project_dir / "results" / "generation_metadata.csv"
    output_dir = project_dir / "results"
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        print("Please run 2_generate_tts_audio.py first.")
        return
    
    print("Starting TTS Quality Evaluation...")
    print("="*70 + "\n")
    
    # Evaluate audio files
    metrics = evaluate_audio_files(metadata_path, output_dir)
    
    # Save metrics and statistics
    save_quality_metrics(metrics, output_dir)
    
    print("✓ Quality evaluation complete!")

if __name__ == "__main__":
    main()
