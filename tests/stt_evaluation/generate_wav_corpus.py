"""
CSV to WAV Conversion Script for STT Testing
Reads queries from stt_test_queries.csv and generates WAV files using edge-tts.
"""

import asyncio
import csv
import os
from pathlib import Path
import edge_tts
from tqdm import tqdm


# Voice mapping for different accents
VOICE_MAP = {
    'us_neutral': 'en-US-GuyNeural',  # Male neutral US voice
    'uk_english': 'en-GB-RyanNeural',  # Male UK voice
    'indian_english': 'en-IN-PrabhatNeural',  # Male Indian voice
    'australian_english': 'en-AU-WilliamNeural',  # Male Australian voice
}

# Rate adjustment for natural speech
RATE = '+0%'  # Normal speaking rate
VOLUME = '+0%'  # Normal volume


async def generate_single_wav(query_text: str, output_path: str, voice: str, rate: str = RATE, volume: str = VOLUME):
    """
    Generate a single WAV file from text using edge-tts.
    
    Args:
        query_text: The text to convert to speech
        output_path: Path where the WAV file will be saved
        voice: The voice to use (from VOICE_MAP)
        rate: Speech rate adjustment
        volume: Volume adjustment
    """
    communicate = edge_tts.Communicate(query_text, voice, rate=rate, volume=volume)
    await communicate.save(output_path)


async def generate_all_wavs(csv_path: str, output_dir: str):
    """
    Read CSV file and generate WAV files for all queries.
    
    Args:
        csv_path: Path to the CSV file containing queries
        output_dir: Directory where WAV files will be saved
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read CSV file
    queries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        queries = list(reader)
    
    print(f"Found {len(queries)} queries to process")
    print(f"Output directory: {output_dir}")
    
    # Generate WAV files with progress bar
    tasks = []
    for query in queries:
        query_id = query['query_id']
        query_text = query['query_text']
        accent = query['accent']
        
        # Get voice for accent
        voice = VOICE_MAP.get(accent, VOICE_MAP['us_neutral'])
        
        # Create output filename: query_001_us_neutral.wav
        output_filename = f"query_{int(query_id):03d}_{accent}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        tasks.append({
            'query_id': query_id,
            'text': query_text,
            'output_path': output_path,
            'voice': voice,
            'accent': accent
        })
    
    # Process all tasks with progress bar
    print("\nGenerating WAV files...")
    for task in tqdm(tasks, desc="Converting to WAV"):
        await generate_single_wav(
            task['text'],
            task['output_path'],
            task['voice']
        )
    
    print(f"\n✅ Successfully generated {len(tasks)} WAV files")
    print(f"📁 Files saved to: {output_dir}")
    
    # Generate summary
    accent_counts = {}
    for task in tasks:
        accent = task['accent']
        accent_counts[accent] = accent_counts.get(accent, 0) + 1
    
    print("\n📊 Generation Summary:")
    print(f"Total WAV files: {len(tasks)}")
    for accent, count in sorted(accent_counts.items()):
        print(f"  - {accent}: {count} files")


def main():
    """Main function to run the WAV generation."""
    # Paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'stt_test_queries.csv'
    output_dir = script_dir / 'wav_outputs'
    
    # Check if CSV exists
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        return
    
    print("=" * 60)
    print("STT Testing: CSV to WAV Conversion")
    print("=" * 60)
    
    # Run async generation
    asyncio.run(generate_all_wavs(str(csv_path), str(output_dir)))
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
