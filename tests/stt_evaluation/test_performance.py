"""
STT Performance Testing Script
Tests Whisper Medium STT model on generated WAV files and calculates performance metrics.
"""

import os
import csv
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import whisper
from jiwer import wer, cer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class STTPerformanceTester:
    """Class to test STT performance on WAV files."""
    
    def __init__(self, model_name: str = "medium"):
        """
        Initialize the tester with Whisper model and semantic similarity model.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        print(f"Loading Whisper model: {model_name}...")
        self.whisper_model = whisper.load_model(model_name)
        
        print("Loading Sentence Transformer for semantic similarity...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ Models loaded successfully\n")
    
    def transcribe_wav(self, wav_path: str) -> Tuple[str, float]:
        """
        Transcribe a WAV file using Whisper.
        
        Args:
            wav_path: Path to the WAV file
            
        Returns:
            Tuple of (transcribed_text, latency_ms)
        """
        start_time = time.time()
        
        # Try to transcribe, handle ffmpeg issues by loading audio manually
        try:
            result = self.whisper_model.transcribe(wav_path, language='en')
        except FileNotFoundError as e:
            # If ffmpeg is not found, load audio using librosa or soundfile
            print(f"Warning: ffmpeg not found, trying alternative audio loading...")
            import librosa
            audio, sr = librosa.load(wav_path, sr=16000)
            result = self.whisper_model.transcribe(audio, language='en')
        
        latency_ms = (time.time() - start_time) * 1000
        
        return result['text'].strip(), latency_ms
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate."""
        return wer(reference, hypothesis)
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate."""
        return cer(reference, hypothesis)
    
    def calculate_semantic_similarity(self, reference: str, hypothesis: str) -> float:
        """
        Calculate semantic similarity between reference and hypothesis.
        
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self.semantic_model.encode([reference, hypothesis])
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return float(similarity[0][0])
    
    def test_single_query(self, query_data: Dict, wav_path: str) -> Dict:
        """
        Test a single query and return metrics.
        
        Args:
            query_data: Dictionary with query information from CSV
            wav_path: Path to the WAV file
            
        Returns:
            Dictionary with test results
        """
        reference_text = query_data['query_text']
        
        # Transcribe
        hypothesis_text, latency_ms = self.transcribe_wav(wav_path)
        
        # Calculate metrics
        wer_score = self.calculate_wer(reference_text, hypothesis_text)
        cer_score = self.calculate_cer(reference_text, hypothesis_text)
        semantic_sim = self.calculate_semantic_similarity(reference_text, hypothesis_text)
        
        return {
            'query_id': query_data['query_id'],
            'reference': reference_text,
            'hypothesis': hypothesis_text,
            'category': query_data['category'],
            'accent': query_data['accent'],
            'noise_level': query_data['noise_level'],
            'difficulty': query_data['expected_difficulty'],
            'wer': wer_score,
            'cer': cer_score,
            'semantic_similarity': semantic_sim,
            'latency_ms': latency_ms,
            'match': reference_text.lower() == hypothesis_text.lower()
        }
    
    def test_all_queries(self, csv_path: str, wav_dir: str, output_dir: str):
        """
        Test all queries and generate comprehensive results.
        
        Args:
            csv_path: Path to CSV file with queries
            wav_dir: Directory containing WAV files
            output_dir: Directory to save results
        """
        # Read queries
        queries = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            queries = list(reader)
        
        print(f"Testing {len(queries)} queries...\n")
        
        # Test each query
        results = []
        for query in tqdm(queries, desc="Processing queries"):
            query_id = int(query['query_id'])
            accent = query['accent']
            wav_filename = f"query_{query_id:03d}_{accent}.wav"
            wav_path = os.path.join(wav_dir, wav_filename)
            
            if not os.path.exists(wav_path):
                print(f"⚠️  Warning: WAV file not found: {wav_filename}")
                continue
            
            result = self.test_single_query(query, wav_path)
            results.append(result)
        
        # Save detailed results
        self.save_results(results, output_dir)
        
        # Calculate and display summary statistics
        self.print_summary(results)
        
        return results
    
    def save_results(self, results: List[Dict], output_dir: str):
        """Save results to JSON and CSV files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save detailed JSON
        json_path = os.path.join(output_dir, 'stt_detailed_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Detailed results saved to: {json_path}")
        
        # Save summary CSV
        csv_path = os.path.join(output_dir, 'stt_results_summary.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['query_id', 'category', 'accent', 'noise_level', 'difficulty',
                         'wer', 'cer', 'semantic_similarity', 'latency_ms', 'match']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow({k: result[k] for k in fieldnames})
        print(f"💾 Summary CSV saved to: {csv_path}")
    
    def print_summary(self, results: List[Dict]):
        """Print comprehensive summary statistics."""
        print("\n" + "=" * 70)
        print("STT PERFORMANCE SUMMARY")
        print("=" * 70)
        
        # Overall metrics
        wer_scores = [r['wer'] for r in results]
        cer_scores = [r['cer'] for r in results]
        semantic_scores = [r['semantic_similarity'] for r in results]
        latency_scores = [r['latency_ms'] for r in results]
        
        print(f"\n📊 Overall Performance (n={len(results)}):")
        print(f"  WER (Word Error Rate):")
        print(f"    Mean: {np.mean(wer_scores):.4f}")
        print(f"    Std:  {np.std(wer_scores):.4f}")
        print(f"    Min:  {np.min(wer_scores):.4f}")
        print(f"    Max:  {np.max(wer_scores):.4f}")
        
        print(f"\n  CER (Character Error Rate):")
        print(f"    Mean: {np.mean(cer_scores):.4f}")
        print(f"    Std:  {np.std(cer_scores):.4f}")
        
        print(f"\n  Semantic Similarity:")
        print(f"    Mean: {np.mean(semantic_scores):.4f}")
        print(f"    Std:  {np.std(semantic_scores):.4f}")
        
        print(f"\n  Latency (ms):")
        print(f"    Mean:   {np.mean(latency_scores):.2f} ms")
        print(f"    Median: {np.median(latency_scores):.2f} ms")
        print(f"    P95:    {np.percentile(latency_scores, 95):.2f} ms")
        print(f"    P99:    {np.percentile(latency_scores, 99):.2f} ms")
        print(f"    Min:    {np.min(latency_scores):.2f} ms")
        print(f"    Max:    {np.max(latency_scores):.2f} ms")
        
        # Perfect matches
        perfect_matches = sum(1 for r in results if r['match'])
        print(f"\n  Perfect Match Rate: {perfect_matches}/{len(results)} ({perfect_matches/len(results)*100:.1f}%)")
        
        # By category
        print(f"\n📋 Performance by Category:")
        categories = {}
        for r in results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)
        
        for cat, cat_results in sorted(categories.items()):
            cat_wer = np.mean([r['wer'] for r in cat_results])
            cat_latency = np.mean([r['latency_ms'] for r in cat_results])
            cat_semantic = np.mean([r['semantic_similarity'] for r in cat_results])
            print(f"  {cat} (n={len(cat_results)}):")
            print(f"    WER: {cat_wer:.4f} | Latency: {cat_latency:.1f}ms | Semantic: {cat_semantic:.4f}")
        
        # By accent
        print(f"\n🌍 Performance by Accent:")
        accents = {}
        for r in results:
            acc = r['accent']
            if acc not in accents:
                accents[acc] = []
            accents[acc].append(r)
        
        for acc, acc_results in sorted(accents.items()):
            acc_wer = np.mean([r['wer'] for r in acc_results])
            acc_latency = np.mean([r['latency_ms'] for r in acc_results])
            acc_semantic = np.mean([r['semantic_similarity'] for r in acc_results])
            print(f"  {acc} (n={len(acc_results)}):")
            print(f"    WER: {acc_wer:.4f} | Latency: {acc_latency:.1f}ms | Semantic: {acc_semantic:.4f}")
        
        # By noise level
        print(f"\n🔊 Performance by Noise Level:")
        noises = {}
        for r in results:
            noise = r['noise_level']
            if noise not in noises:
                noises[noise] = []
            noises[noise].append(r)
        
        for noise, noise_results in sorted(noises.items()):
            noise_wer = np.mean([r['wer'] for r in noise_results])
            noise_latency = np.mean([r['latency_ms'] for r in noise_results])
            noise_semantic = np.mean([r['semantic_similarity'] for r in noise_results])
            print(f"  {noise} (n={len(noise_results)}):")
            print(f"    WER: {noise_wer:.4f} | Latency: {noise_latency:.1f}ms | Semantic: {noise_semantic:.4f}")
        
        print("\n" + "=" * 70)


def main():
    """Main function to run STT testing."""
    # Paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'stt_test_queries.csv'
    wav_dir = script_dir / 'wav_outputs'
    output_dir = script_dir / 'results'
    
    # Check paths
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        return
    
    if not wav_dir.exists() or not list(wav_dir.glob('*.wav')):
        print(f"❌ Error: No WAV files found in {wav_dir}")
        print("Please run generate_wav_from_csv.py first!")
        return
    
    print("=" * 70)
    print("STT PERFORMANCE TESTING")
    print("=" * 70)
    print(f"CSV:    {csv_path}")
    print(f"WAV:    {wav_dir}")
    print(f"Output: {output_dir}")
    print("=" * 70 + "\n")
    
    # Initialize tester
    tester = STTPerformanceTester(model_name="medium")
    
    # Run tests
    results = tester.test_all_queries(str(csv_path), str(wav_dir), str(output_dir))
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()
