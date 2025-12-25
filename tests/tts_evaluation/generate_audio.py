"""
Script 2: Generate TTS Audio Files

Generates 12,000 WAV files using Edge-TTS:
- 4 voice profiles (GuyNeural, JennyNeural, RyanNeural, NeerjaNeural)
- 3 speaking rates (slow +0%, medium +10%, fast +25%)
- 1000 response texts

Output: generated_audio/ folders with WAV files
"""

import asyncio
import csv
import time
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import edge_tts

# Voice configurations
VOICE_CONFIGS = [
    {"name": "en-US-GuyNeural", "gender": "Male", "accent": "US"},
    {"name": "en-US-JennyNeural", "gender": "Female", "accent": "US"},
    {"name": "en-GB-RyanNeural", "gender": "Male", "accent": "UK"},
    {"name": "en-IN-NeerjaNeural", "gender": "Female", "accent": "Indian"},
]

# Speaking rate configurations (percentage adjustment)
SPEAKING_RATES = [
    {"name": "slow", "rate": "+0%", "wpm_target": 175},
    {"name": "medium", "rate": "+10%", "wpm_target": 192},
    {"name": "fast", "rate": "+25%", "wpm_target": 219},
]

async def generate_single_audio(
    text: str,
    voice_name: str,
    rate: str,
    output_path: Path
) -> Dict:
    """Generate a single TTS audio file."""
    
    start_time = time.time()
    
    try:
        # Create TTS communicator
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice_name,
            rate=rate
        )
        
        # Save audio file
        await communicate.save(str(output_path))
        
        generation_time = time.time() - start_time
        
        # Get audio duration (requires reading the file)
        import wave
        try:
            with wave.open(str(output_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
        except:
            duration = None
        
        return {
            "success": True,
            "generation_time": generation_time,
            "audio_duration": duration,
            "error": None
        }
    
    except Exception as e:
        generation_time = time.time() - start_time
        return {
            "success": False,
            "generation_time": generation_time,
            "audio_duration": None,
            "error": str(e)
        }

async def generate_audio_batch(
    corpus: List[Dict],
    voice_config: Dict,
    rate_config: Dict,
    output_dir: Path,
    metadata_list: List[Dict]
) -> int:
    """Generate audio files for one voice-rate combination."""
    
    voice_name = voice_config["name"]
    rate_name = rate_config["name"]
    rate_value = rate_config["rate"]
    
    # Create output directory
    folder_name = f"{voice_name}_{rate_name}"
    voice_output_dir = output_dir / folder_name
    voice_output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    # Progress bar for this batch
    pbar = tqdm(corpus, desc=f"{voice_name} ({rate_name})", leave=False)
    
    for response in pbar:
        response_id = response["response_id"]
        text = response["text"]
        
        # Output filename
        output_filename = f"{response_id}.wav"
        output_path = voice_output_dir / output_filename
        
        # Skip if already exists
        if output_path.exists():
            success_count += 1
            continue
        
        # Generate audio
        result = await generate_single_audio(
            text=text,
            voice_name=voice_name,
            rate=rate_value,
            output_path=output_path
        )
        
        # Record metadata
        metadata_entry = {
            "response_id": response_id,
            "voice": voice_name,
            "speaking_rate": rate_name,
            "rate_value": rate_value,
            "text": text,
            "word_count": response["word_count"],
            "category": response["category"],
            "generation_time": result["generation_time"],
            "audio_duration": result["audio_duration"],
            "audio_path": str(output_path.relative_to(output_dir.parent)),
            "success": result["success"],
            "error": result["error"]
        }
        metadata_list.append(metadata_entry)
        
        if result["success"]:
            success_count += 1
        
        # Small delay to avoid rate limiting
        await asyncio.sleep(0.05)  # 50ms delay
    
    return success_count

async def generate_all_audio(corpus_path: Path, output_dir: Path) -> List[Dict]:
    """Generate all TTS audio files."""
    
    # Load corpus
    print(f"Loading corpus from {corpus_path}...")
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['word_count'] = int(row['word_count'])
            corpus.append(row)
    
    # Limit to 200 samples for faster testing
    corpus = corpus[:200]
    
    print(f"✓ Loaded {len(corpus)} responses (limited to 200 for testing)\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata tracking
    all_metadata = []
    total_combinations = len(VOICE_CONFIGS) * len(SPEAKING_RATES)
    total_files = len(corpus) * total_combinations
    
    print(f"Generating {total_files:,} audio files...")
    print(f"  Voices: {len(VOICE_CONFIGS)}")
    print(f"  Speaking Rates: {len(SPEAKING_RATES)}")
    print(f"  Responses: {len(corpus)}")
    print("="*70 + "\n")
    
    # Overall progress
    overall_pbar = tqdm(total=total_combinations, desc="Overall Progress", position=0)
    
    total_success = 0
    total_failed = 0
    
    # Generate for each voice-rate combination
    for voice_config in VOICE_CONFIGS:
        for rate_config in SPEAKING_RATES:
            success_count = await generate_audio_batch(
                corpus=corpus,
                voice_config=voice_config,
                rate_config=rate_config,
                output_dir=output_dir,
                metadata_list=all_metadata
            )
            
            failed_count = len(corpus) - success_count
            total_success += success_count
            total_failed += failed_count
            
            overall_pbar.update(1)
            overall_pbar.set_postfix({
                "Success": total_success,
                "Failed": total_failed
            })
    
    overall_pbar.close()
    
    return all_metadata

def save_metadata(metadata: List[Dict], output_dir: Path):
    """Save generation metadata to CSV and JSON."""
    
    results_dir = output_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = results_dir / "generation_metadata.csv"
    if metadata:
        fieldnames = metadata[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata)
        print(f"\n✓ Saved metadata to {csv_path}")
    
    # Calculate statistics
    total_files = len(metadata)
    successful = sum(1 for m in metadata if m["success"])
    failed = total_files - successful
    
    if successful > 0:
        avg_gen_time = sum(m["generation_time"] for m in metadata if m["success"]) / successful
        
        # Calculate RTF for successful generations
        rtf_values = []
        for m in metadata:
            if m["success"] and m["audio_duration"]:
                rtf = m["generation_time"] / m["audio_duration"]
                rtf_values.append(rtf)
        
        avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else None
    else:
        avg_gen_time = None
        avg_rtf = None
    
    stats = {
        "total_files": total_files,
        "successful": successful,
        "failed": failed,
        "success_rate": (successful / total_files * 100) if total_files > 0 else 0,
        "avg_generation_time": avg_gen_time,
        "avg_rtf": avg_rtf,
        "voices": len(VOICE_CONFIGS),
        "speaking_rates": len(SPEAKING_RATES),
        "responses_per_combo": total_files // (len(VOICE_CONFIGS) * len(SPEAKING_RATES))
    }
    
    stats_path = results_dir / "generation_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("AUDIO GENERATION SUMMARY")
    print("="*70)
    print(f"Total Files:          {stats['total_files']:,}")
    print(f"Successful:           {stats['successful']:,} ({stats['success_rate']:.1f}%)")
    print(f"Failed:               {stats['failed']:,}")
    if avg_gen_time:
        print(f"Avg Generation Time:  {avg_gen_time:.3f}s")
    if avg_rtf:
        print(f"Avg RTF:              {avg_rtf:.3f}")
    print("="*70 + "\n")

async def main():
    """Main execution function."""
    
    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    corpus_path = project_dir / "data" / "response_corpus.csv"
    output_dir = project_dir / "generated_audio"
    
    # Check if corpus exists
    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        print("Please run 1_generate_corpus.py first.")
        return
    
    # Generate audio files
    print("Starting TTS Audio Generation...")
    print("="*70)
    print("This will take approximately 2-3 hours.\n")
    
    start_time = time.time()
    
    metadata = await generate_all_audio(corpus_path, output_dir)
    
    elapsed_time = time.time() - start_time
    
    # Save metadata
    save_metadata(metadata, output_dir)
    
    print(f"✓ Total time: {elapsed_time/3600:.2f} hours")
    print(f"✓ Audio files saved to: {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
