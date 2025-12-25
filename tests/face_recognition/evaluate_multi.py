"""
Multi-Person Face Recognition Evaluation System
Comprehensive evaluation with 10+ diverse persons and unknown person testing.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Facenet"))
from facenet_multi import init_models, embed_faces_bgr

class MultiPersonFaceRecognitionEvaluator:
    def __init__(self, dataset_dir="test_dataset", output_dir="evaluation_results_multi"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "latex").mkdir(exist_ok=True)
        (self.output_dir / "raw_data").mkdir(exist_ok=True)
        
        # Force CPU mode
        self.device = "cpu"
        print(f"\n{'='*70}")
        print(f"🔧 Multi-Person Face Recognition Evaluator")
        print(f"{'='*70}")
        print(f"Device: {self.device.upper()}")
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Initialize models
        print("\nLoading models...")
        self.mtcnn, self.net = init_models(self.device)
        print("✓ MTCNN and InceptionResnetV1 loaded")
        
        # Storage
        self.enrolled_persons = {}  # {name: {'embedding': np.array, 'count': int}}
        self.results = []
        
    def enroll_all_persons(self, enrollment_count=25):
        """Enroll all persons from dataset"""
        print("\n" + "="*70)
        print("📝 ENROLLMENT PHASE")
        print("="*70)
        
        person_dirs = sorted([d for d in self.dataset_dir.iterdir() 
                             if d.is_dir() and d.name.startswith("person_")])
        
        if len(person_dirs) == 0:
            raise ValueError(f"No persons found in {self.dataset_dir}")
        
        print(f"Found {len(person_dirs)} persons to enroll")
        print(f"Using {enrollment_count} photos per person for enrollment")
        
        for person_dir in person_dirs:
            person_name = person_dir.name
            
            # Get all photos
            photos = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
                photos.extend(list(person_dir.glob(ext)))
            photos = sorted(photos)
            
            if len(photos) < enrollment_count:
                print(f"\n⚠ {person_name}: Only {len(photos)} photos (need {enrollment_count}), skipping")
                continue
            
            # Use first N photos for enrollment
            enrollment_photos = photos[:enrollment_count]
            embeddings = []
            
            print(f"\n📸 Enrolling {person_name}...")
            
            for photo_path in tqdm(enrollment_photos, desc="  Processing", leave=False):
                img = cv2.imread(str(photo_path))
                if img is None:
                    continue
                
                embs, boxes = embed_faces_bgr(img, self.mtcnn, self.net, self.device)
                
                if embs is not None and len(embs) > 0:
                    # Take largest face
                    if len(boxes) > 1:
                        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                        largest_idx = int(np.argmax(areas))
                        embeddings.append(embs[largest_idx])
                    else:
                        embeddings.append(embs[0])
            
            if len(embeddings) > 0:
                # Average all embeddings and normalize
                mean_embedding = np.mean(embeddings, axis=0)
                mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-9)
                
                self.enrolled_persons[person_name] = {
                    'embedding': mean_embedding,
                    'enrollment_count': len(embeddings),
                    'total_photos': len(photos),
                    'available_test': len(photos) - enrollment_count
                }
                
                print(f"  ✓ {person_name}: {len(embeddings)}/{len(enrollment_photos)} faces detected")
            else:
                print(f"  ✗ {person_name}: No faces detected, skipping")
        
        print(f"\n{'='*70}")
        print(f"✓ Successfully enrolled {len(self.enrolled_persons)} persons")
        print(f"{'='*70}")
        
        return len(self.enrolled_persons)
    
    def recognize_face(self, embedding, threshold=0.82):
        """Recognize face against gallery"""
        if embedding is None:
            return "NO_DETECTION", 0.0
        
        # Build gallery matrix
        enrolled_names = list(self.enrolled_persons.keys())
        gallery_matrix = np.vstack([self.enrolled_persons[name]['embedding'] 
                                    for name in enrolled_names])
        
        # Normalize test embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        # Cosine similarity
        similarities = gallery_matrix @ embedding
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        best_name = enrolled_names[best_idx]
        
        if best_sim >= threshold:
            return best_name, best_sim
        else:
            return "Unknown", best_sim
    
    def test_known_persons(self, threshold=0.82, enrollment_count=25):
        """Test recognition on enrolled persons"""
        print("\n" + "="*70)
        print("🧪 TESTING PHASE - KNOWN PERSONS")
        print("="*70)
        
        person_dirs = sorted([d for d in self.dataset_dir.iterdir() 
                             if d.is_dir() and d.name.startswith("person_")])
        
        total_tested = 0
        
        for person_dir in person_dirs:
            person_name = person_dir.name
            
            if person_name not in self.enrolled_persons:
                continue
            
            # Get all photos
            photos = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
                photos.extend(list(person_dir.glob(ext)))
            photos = sorted(photos)
            
            # Use photos after enrollment set for testing
            test_photos = photos[enrollment_count:]
            
            if len(test_photos) == 0:
                print(f"\n⚠ {person_name}: No test photos available")
                continue
            
            print(f"\n🔍 Testing {person_name} ({len(test_photos)} photos)...")
            
            for photo_path in tqdm(test_photos, desc="  Processing", leave=False):
                start_time = time.perf_counter()
                
                img = cv2.imread(str(photo_path))
                if img is None:
                    continue
                
                # Detect and embed
                detect_start = time.perf_counter()
                embs, boxes = embed_faces_bgr(img, self.mtcnn, self.net, self.device)
                detect_time = time.perf_counter() - detect_start
                
                if embs is None or len(embs) == 0:
                    self.results.append({
                        'true_label': person_name,
                        'predicted_label': 'NO_DETECTION',
                        'confidence': 0.0,
                        'correct': False,
                        'latency_total_ms': (time.perf_counter() - start_time) * 1000,
                        'latency_detection_ms': detect_time * 1000,
                        'latency_recognition_ms': 0,
                        'photo_path': str(photo_path),
                        'is_unknown': False
                    })
                    continue
                
                # Take largest face
                if len(boxes) > 1:
                    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                    largest_idx = int(np.argmax(areas))
                    test_embedding = embs[largest_idx]
                else:
                    test_embedding = embs[0]
                
                # Recognize
                recog_start = time.perf_counter()
                predicted_name, confidence = self.recognize_face(test_embedding, threshold)
                recog_time = time.perf_counter() - recog_start
                
                total_time = time.perf_counter() - start_time
                
                self.results.append({
                    'true_label': person_name,
                    'predicted_label': predicted_name,
                    'confidence': confidence,
                    'correct': predicted_name == person_name,
                    'latency_total_ms': total_time * 1000,
                    'latency_detection_ms': detect_time * 1000,
                    'latency_recognition_ms': recog_time * 1000,
                    'photo_path': str(photo_path),
                    'is_unknown': False
                })
                
                total_tested += 1
            
            correct = sum(1 for r in self.results 
                         if r['true_label'] == person_name and r['correct'])
            total = sum(1 for r in self.results if r['true_label'] == person_name)
            
            print(f"  ✓ Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
        
        print(f"\n{'='*70}")
        print(f"✓ Tested {total_tested} known person images")
        print(f"{'='*70}")
    
    def test_unknown_persons(self, threshold=0.82):
        """Test false positive rate on unknown persons"""
        print("\n" + "="*70)
        print("🔍 TESTING PHASE - UNKNOWN PERSONS (FPR)")
        print("="*70)
        
        unknown_dir = self.dataset_dir / "unknown_persons"
        
        if not unknown_dir.exists():
            print("⚠ No unknown persons directory found, skipping FPR test")
            return
        
        unknown_persons = sorted([d for d in unknown_dir.iterdir() if d.is_dir()])
        
        if len(unknown_persons) == 0:
            print("⚠ No unknown persons found")
            return
        
        print(f"Found {len(unknown_persons)} unknown persons")
        
        total_tested = 0
        
        for unknown_person_dir in unknown_persons:
            # Get all photos
            photos = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
                photos.extend(list(unknown_person_dir.glob(ext)))
            
            if len(photos) == 0:
                continue
            
            print(f"\n🔍 Testing {unknown_person_dir.name} ({len(photos)} photos)...")
            
            for photo_path in tqdm(photos, desc="  Processing", leave=False):
                start_time = time.perf_counter()
                
                img = cv2.imread(str(photo_path))
                if img is None:
                    continue
                
                # Detect and embed
                detect_start = time.perf_counter()
                embs, boxes = embed_faces_bgr(img, self.mtcnn, self.net, self.device)
                detect_time = time.perf_counter() - detect_start
                
                if embs is None or len(embs) == 0:
                    self.results.append({
                        'true_label': 'Unknown',
                        'predicted_label': 'NO_DETECTION',
                        'confidence': 0.0,
                        'correct': True,  # Correct to not detect
                        'latency_total_ms': (time.perf_counter() - start_time) * 1000,
                        'latency_detection_ms': detect_time * 1000,
                        'latency_recognition_ms': 0,
                        'photo_path': str(photo_path),
                        'is_unknown': True
                    })
                    continue
                
                # Take largest face
                if len(boxes) > 1:
                    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                    largest_idx = int(np.argmax(areas))
                    test_embedding = embs[largest_idx]
                else:
                    test_embedding = embs[0]
                
                # Recognize
                recog_start = time.perf_counter()
                predicted_name, confidence = self.recognize_face(test_embedding, threshold)
                recog_time = time.perf_counter() - recog_start
                
                total_time = time.perf_counter() - start_time
                
                self.results.append({
                    'true_label': 'Unknown',
                    'predicted_label': predicted_name,
                    'confidence': confidence,
                    'correct': predicted_name == "Unknown",
                    'latency_total_ms': total_time * 1000,
                    'latency_detection_ms': detect_time * 1000,
                    'latency_recognition_ms': recog_time * 1000,
                    'photo_path': str(photo_path),
                    'is_unknown': True
                })
                
                total_tested += 1
            
            correct = sum(1 for r in self.results 
                         if r['is_unknown'] and r['correct'])
            total = sum(1 for r in self.results if r['is_unknown'])
            
            if total > 0:
                print(f"  ✓ Results: {correct}/{total} correctly rejected ({correct/total*100:.1f}%)")
        
        print(f"\n{'='*70}")
        print(f"✓ Tested {total_tested} unknown person images")
        print(f"{'='*70}")
    
    def calculate_metrics(self):
        """Calculate comprehensive metrics"""
        df = pd.DataFrame(self.results)
        
        # Filter valid detections
        valid_df = df[df['predicted_label'] != 'NO_DETECTION'].copy()
        
        # Separate known and unknown
        known_df = valid_df[valid_df['is_unknown'] == False]
        unknown_df = valid_df[valid_df['is_unknown'] == True]
        
        # Calculate confusion matrix elements
        tp = len(known_df[known_df['correct'] == True])  # Known correctly identified
        fn = len(known_df[known_df['correct'] == False])  # Known misidentified as Unknown
        
        tn = len(unknown_df[unknown_df['correct'] == True])  # Unknown correctly rejected
        fp = len(unknown_df[unknown_df['correct'] == False])  # Unknown misidentified as known
        
        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Latency stats
        latencies = valid_df['latency_total_ms'].values
        detect_latencies = valid_df['latency_detection_ms'].values
        recog_latencies = valid_df['latency_recognition_ms'].values
        
        metrics = {
            'total_images': len(df),
            'valid_detections': len(valid_df),
            'detection_rate': len(valid_df) / len(df) if len(df) > 0 else 0,
            'enrolled_persons': len(self.enrolled_persons),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'tnr': float(tnr),
            'fnr': float(fnr),
            'mean_latency_ms': float(np.mean(latencies)) if len(latencies) > 0 else 0,
            'median_latency_ms': float(np.median(latencies)) if len(latencies) > 0 else 0,
            'std_latency_ms': float(np.std(latencies)) if len(latencies) > 0 else 0,
            'p95_latency_ms': float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0,
            'mean_detection_ms': float(np.mean(detect_latencies)) if len(detect_latencies) > 0 else 0,
            'mean_recognition_ms': float(np.mean(recog_latencies)) if len(recog_latencies) > 0 else 0,
        }
        
        return metrics
    
    def run_full_evaluation(self, enrollment_count=25, threshold=0.82):
        """Run complete evaluation pipeline"""
        print("\n" + "="*70)
        print("🎯 MULTI-PERSON FACE RECOGNITION EVALUATION")
        print("="*70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Enrollment: {enrollment_count} images per person")
        print(f"Threshold: {threshold}")
        print(f"Device: {self.device.upper()}")
        
        start_time = time.time()
        
        # Phase 1: Enrollment
        num_enrolled = self.enroll_all_persons(enrollment_count=enrollment_count)
        
        if num_enrolled == 0:
            raise ValueError("No persons successfully enrolled!")
        
        # Phase 2: Test known persons
        self.test_known_persons(threshold=threshold, enrollment_count=enrollment_count)
        
        # Phase 3: Test unknown persons
        self.test_unknown_persons(threshold=threshold)
        
        # Phase 4: Calculate metrics
        print("\n" + "="*70)
        print("📊 CALCULATING METRICS")
        print("="*70)
        
        metrics = self.calculate_metrics()
        
        # Save results
        print("\n💾 Saving results...")
        
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.output_dir / "raw_data" / "all_results.csv", index=False)
        print(f"  ✓ Raw results: {self.output_dir / 'raw_data' / 'all_results.csv'}")
        
        with open(self.output_dir / "raw_data" / "overall_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Metrics: {self.output_dir / 'raw_data' / 'overall_metrics.json'}")
        
        # Print summary
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE")
        print("="*70)
        print(f"\nDuration: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📊 Summary:")
        print(f"  Enrolled persons: {metrics['enrolled_persons']}")
        print(f"  Total images tested: {metrics['total_images']}")
        print(f"  Valid detections: {metrics['valid_detections']} ({metrics['detection_rate']*100:.1f}%)")
        
        print(f"\n🎯 Performance @ Threshold {threshold}:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        print(f"  TPR:       {metrics['tpr']*100:.2f}%")
        print(f"  FPR:       {metrics['fpr']*100:.2f}%")
        
        print(f"\n📦 Confusion Matrix:")
        print(f"  TP: {metrics['tp']}  FP: {metrics['fp']}")
        print(f"  FN: {metrics['fn']}  TN: {metrics['tn']}")
        
        print(f"\n⏱ Latency (CPU):")
        print(f"  Mean:   {metrics['mean_latency_ms']:.1f} ms")
        print(f"  Median: {metrics['median_latency_ms']:.1f} ms")
        print(f"  95th:   {metrics['p95_latency_ms']:.1f} ms")
        print(f"  Detection:   {metrics['mean_detection_ms']:.1f} ms")
        print(f"  Recognition: {metrics['mean_recognition_ms']:.1f} ms")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print("="*70)
        
        return metrics, results_df

def main():
    """Main execution"""
    evaluator = MultiPersonFaceRecognitionEvaluator(
        dataset_dir="test_dataset",
        output_dir="evaluation_results_multi"
    )
    
    try:
        metrics, results_df = evaluator.run_full_evaluation(
            enrollment_count=25,
            threshold=0.82
        )
        
        print("\n🎉 Evaluation completed successfully!")
        print("\nNext steps:")
        print("  1. Review results in: evaluation_results_multi/")
        print("  2. Generate figures: py generate_ieee_figures_multi.py")
        
    except KeyboardInterrupt:
        print("\n\n❌ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
