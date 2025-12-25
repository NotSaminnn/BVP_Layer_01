"""
Automated Face Recognition Evaluation System
Generates accuracy, latency, and performance metrics for CVPR paper.

CPU-only performance testing (no GPU).
"""

import os
import sys
import time
import pickle
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import cv2
import torch

# Add parent directory to path to import facenet_multi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Facenet'))
from facenet_multi import init_models, embed_faces_bgr, bgr2rgb

class AutomatedFaceRecognitionEvaluator:
    """
    Fully automated evaluation of face recognition system.
    Focuses on CPU performance only.
    """
    
    def __init__(self, base_dir=".", output_dir="evaluation_results"):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / output_dir
        self.enrolled_dir = self.base_dir / "enrolled_persons"
        self.unknown_dir = self.base_dir / "unknown_persons"
        
        # Create output directories
        (self.output_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "raw_data").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "latex").mkdir(parents=True, exist_ok=True)
        
        # Force CPU-only mode
        self.device = "cpu"
        print(f"\n{'='*60}")
        print(f"🔧 Initializing Face Recognition Evaluator (CPU ONLY)")
        print(f"{'='*60}")
        print(f"Device: {self.device.upper()}")
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Initialize models
        print("\nLoading models...")
        self.mtcnn, self.net = init_models(self.device)
        print("✓ MTCNN and InceptionResnetV1 loaded")
        
        # Build gallery from enrolled persons
        print("\nBuilding gallery from enrolled persons...")
        self.gallery = self.build_gallery_from_folders()
        
        if not self.gallery:
            raise ValueError("❌ No enrolled persons found! Check enrolled_persons/ folder")
        
        print(f"✓ Gallery built with {len(self.gallery)} persons: {list(self.gallery.keys())}")
        
        # Results storage
        self.all_results = []
        
    def build_gallery_from_folders(self, enrollment_limit=25):
        """
        Build gallery by enrolling all persons from enrolled_persons/ folder.
        Uses first N images for enrollment.
        """
        gallery = {}
        
        person_folders = [d for d in self.enrolled_dir.iterdir() if d.is_dir()]
        
        for person_folder in person_folders:
            person_name = person_folder.name.replace("_photos", "")
            
            # Get all image files
            image_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
                image_paths.extend(person_folder.glob(ext))
            
            if not image_paths:
                print(f"  ⚠️  No images found for {person_name}")
                continue
            
            # Use first enrollment_limit images
            enrollment_images = sorted(image_paths)[:enrollment_limit]
            
            print(f"  Enrolling {person_name} with {len(enrollment_images)} images...")
            
            embeddings = []
            for img_path in enrollment_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                embs, boxes = embed_faces_bgr(img, self.mtcnn, self.net, self.device)
                if embs is not None and len(embs) > 0:
                    embeddings.append(embs[0])
            
            if len(embeddings) == 0:
                print(f"    ⚠️  No faces detected for {person_name}")
                continue
            
            # Average embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-9)
            
            gallery[person_name] = {
                "emb": avg_embedding,
                "n": len(embeddings)
            }
            
            print(f"    ✓ {person_name} enrolled ({len(embeddings)} faces)")
        
        return gallery
    
    def recognize_face(self, embedding, threshold=0.82):
        """
        Recognize a face embedding against gallery.
        Returns: (name, similarity_score)
        """
        if embedding is None:
            return None, 0.0
        
        names = list(self.gallery.keys())
        G = np.stack([self.gallery[n]["emb"] for n in names], axis=0)
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        # Cosine similarity
        sims = G @ embedding
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_name = names[best_idx]
        
        if best_sim >= threshold:
            return best_name, best_sim
        else:
            return "Unknown", best_sim
    
    def evaluate_single_image(self, image_path, ground_truth_name, threshold=0.82):
        """
        Evaluate a single image and return detailed metrics.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Time detection
        t_detect_start = time.perf_counter()
        embs, boxes = embed_faces_bgr(img, self.mtcnn, self.net, self.device)
        t_detect = time.perf_counter() - t_detect_start
        
        if embs is None or len(embs) == 0:
            return {
                'image': str(image_path),
                'ground_truth': ground_truth_name,
                'detected': False,
                'prediction': None,
                'similarity': 0.0,
                'correct': False,
                'latency_detection_ms': t_detect * 1000,
                'latency_recognition_ms': 0,
                'latency_total_ms': t_detect * 1000
            }
        
        # Take largest face
        if len(boxes) > 1:
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            idx = int(np.argmax(areas))
            embedding = embs[idx]
        else:
            embedding = embs[0]
        
        # Time recognition
        t_recog_start = time.perf_counter()
        pred_name, similarity = self.recognize_face(embedding, threshold)
        t_recog = time.perf_counter() - t_recog_start
        
        # Check correctness
        if ground_truth_name == "Unknown":
            correct = (pred_name == "Unknown")
        else:
            correct = (pred_name == ground_truth_name)
        
        return {
            'image': str(image_path),
            'ground_truth': ground_truth_name,
            'detected': True,
            'prediction': pred_name,
            'similarity': similarity,
            'correct': correct,
            'latency_detection_ms': t_detect * 1000,
            'latency_recognition_ms': t_recog * 1000,
            'latency_total_ms': (t_detect + t_recog) * 1000,
            'embedding': embedding  # Store for threshold ablation
        }
    
    def evaluate_person_folder(self, person_name, folder_path, threshold=0.82, skip_first_n=25):
        """
        Evaluate all images in a person's folder.
        skip_first_n: Skip first N images (used for enrollment)
        """
        print(f"\n📁 Evaluating {person_name}...")
        
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(folder_path.glob(ext))
        
        # Sort and skip enrollment images
        image_paths = sorted(image_paths)[skip_first_n:]
        
        if not image_paths:
            print(f"  ⚠️  No test images (after skipping enrollment images)")
            return []
        
        print(f"  Testing with {len(image_paths)} images...")
        
        results = []
        for i, img_path in enumerate(image_paths, 1):
            result = self.evaluate_single_image(img_path, person_name, threshold)
            if result:
                results.append(result)
            
            if i % 10 == 0:
                print(f"    Progress: {i}/{len(image_paths)}")
        
        detected_count = sum(1 for r in results if r['detected'])
        correct_count = sum(1 for r in results if r['correct'])
        
        print(f"  ✓ Results: {detected_count}/{len(results)} detected, {correct_count}/{detected_count if detected_count > 0 else 1} correct")
        
        return results
    
    def evaluate_all_enrolled_persons(self, threshold=0.82, enrollment_size=25):
        """
        Evaluate all enrolled persons using their remaining images.
        """
        print("\n" + "="*60)
        print("🚀 EVALUATING ALL ENROLLED PERSONS")
        print("="*60)
        
        all_results = []
        
        for person_name in self.gallery.keys():
            # Find corresponding folder
            folder_name = f"{person_name}_photos"
            folder_path = self.enrolled_dir / folder_name
            
            if not folder_path.exists():
                # Try without _photos suffix
                folder_path = self.enrolled_dir / person_name
            
            if folder_path.exists() and folder_path.is_dir():
                results = self.evaluate_person_folder(
                    person_name, 
                    folder_path, 
                    threshold=threshold,
                    skip_first_n=enrollment_size
                )
                all_results.extend(results)
            else:
                print(f"⚠️  Folder not found for {person_name}")
        
        return all_results
    
    def evaluate_unknown_persons(self, threshold=0.82):
        """
        Evaluate unknown persons (for FPR calculation).
        """
        print("\n" + "="*60)
        print("🚀 EVALUATING UNKNOWN PERSONS (FPR Test)")
        print("="*60)
        
        if not self.unknown_dir.exists() or not any(self.unknown_dir.iterdir()):
            print("  ⚠️  No unknown person images found")
            print("  ℹ️  FPR cannot be calculated without unknown person data")
            return []
        
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(self.unknown_dir.glob(ext))
        
        print(f"  Testing with {len(image_paths)} unknown images...")
        
        results = []
        for i, img_path in enumerate(image_paths, 1):
            result = self.evaluate_single_image(img_path, "Unknown", threshold)
            if result:
                results.append(result)
            
            if i % 10 == 0:
                print(f"    Progress: {i}/{len(image_paths)}")
        
        return results
    
    def calculate_metrics(self, results_df):
        """
        Calculate comprehensive metrics from results.
        """
        detected = results_df[results_df['detected'] == True]
        
        if len(detected) == 0:
            return None
        
        # Separate known and unknown
        known = detected[detected['ground_truth'] != "Unknown"]
        unknown = detected[detected['ground_truth'] == "Unknown"]
        
        # True/False Positives/Negatives
        tp = len(known[known['correct'] == True])
        fn = len(known[known['correct'] == False])
        
        tn = len(unknown[unknown['prediction'] == "Unknown"]) if len(unknown) > 0 else 0
        fp = len(unknown[unknown['prediction'] != "Unknown"]) if len(unknown) > 0 else 0
        
        # Calculate metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Latency stats
        latencies = detected['latency_total_ms'].values
        latency_detect = detected['latency_detection_ms'].values
        latency_recog = detected['latency_recognition_ms'].values
        
        return {
            'total_images': len(results_df),
            'detected_faces': len(detected),
            'detection_rate': len(detected) / len(results_df) if len(results_df) > 0 else 0,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'tpr': tpr, 'fpr': fpr, 'tnr': tnr, 'fnr': fnr,
            'precision': precision,
            'recall': tpr,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'mean_detection_ms': float(np.mean(latency_detect)),
            'mean_recognition_ms': float(np.mean(latency_recog)),
        }
    
    def threshold_ablation_study(self, test_results, thresholds=None):
        """
        Test multiple thresholds on the same embedding data.
        """
        if thresholds is None:
            thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95]
        
        print("\n" + "="*60)
        print("🔍 THRESHOLD ABLATION STUDY")
        print("="*60)
        
        # Filter results with embeddings
        results_with_emb = [r for r in test_results if 'embedding' in r and r['embedding'] is not None]
        
        if not results_with_emb:
            print("  ⚠️  No embeddings stored. Skipping threshold ablation.")
            return None
        
        threshold_metrics = []
        
        for threshold in thresholds:
            print(f"\n  Threshold: {threshold:.2f}")
            
            # Re-classify with new threshold
            modified_results = []
            for res in results_with_emb:
                pred_name, similarity = self.recognize_face(res['embedding'], threshold)
                
                # Update result
                new_res = res.copy()
                new_res['prediction'] = pred_name
                new_res['similarity'] = similarity
                
                # Recalculate correctness
                if new_res['ground_truth'] == "Unknown":
                    new_res['correct'] = (pred_name == "Unknown")
                else:
                    new_res['correct'] = (pred_name == new_res['ground_truth'])
                
                modified_results.append(new_res)
            
            # Calculate metrics
            df = pd.DataFrame(modified_results)
            metrics = self.calculate_metrics(df)
            
            if metrics:
                metrics['threshold'] = threshold
                threshold_metrics.append(metrics)
                print(f"    TPR: {metrics['tpr']:.3f}, FPR: {metrics['fpr']:.3f}, "
                      f"F1: {metrics['f1_score']:.3f}, Acc: {metrics['accuracy']:.3f}")
        
        return pd.DataFrame(threshold_metrics)
    
    def run_full_evaluation(self, enrollment_size=25, threshold=0.82):
        """
        Run complete evaluation pipeline.
        """
        print("\n" + "="*70)
        print("🎯 STARTING FULL AUTOMATED EVALUATION")
        print("="*70)
        print(f"Enrollment size: {enrollment_size} images per person")
        print(f"Recognition threshold: {threshold}")
        print(f"Device: {self.device.upper()}")
        print(f"Gallery: {list(self.gallery.keys())}")
        
        start_time = time.time()
        
        # 1. Evaluate enrolled persons
        known_results = self.evaluate_all_enrolled_persons(threshold, enrollment_size)
        
        # 2. Evaluate unknown persons
        unknown_results = self.evaluate_unknown_persons(threshold)
        
        # 3. Combine results
        all_results = known_results + unknown_results
        
        if not all_results:
            print("\n❌ No results generated!")
            return
        
        # 4. Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # 5. Calculate overall metrics
        print("\n" + "="*60)
        print("📊 OVERALL METRICS")
        print("="*60)
        
        overall_metrics = self.calculate_metrics(results_df)
        
        if overall_metrics:
            print(f"\nDetection Rate: {overall_metrics['detection_rate']:.2%}")
            print(f"Accuracy: {overall_metrics['accuracy']:.2%}")
            print(f"Precision: {overall_metrics['precision']:.2%}")
            print(f"Recall (TPR): {overall_metrics['recall']:.2%}")
            print(f"F1-Score: {overall_metrics['f1_score']:.3f}")
            print(f"False Positive Rate: {overall_metrics['fpr']:.2%}")
            print(f"False Negative Rate: {overall_metrics['fnr']:.2%}")
            print(f"\nLatency (CPU):")
            print(f"  Mean: {overall_metrics['mean_latency_ms']:.1f} ms")
            print(f"  Median: {overall_metrics['median_latency_ms']:.1f} ms")
            print(f"  95th percentile: {overall_metrics['p95_latency_ms']:.1f} ms")
            print(f"  Detection: {overall_metrics['mean_detection_ms']:.1f} ms")
            print(f"  Recognition: {overall_metrics['mean_recognition_ms']:.1f} ms")
        
        # 6. Threshold ablation
        threshold_df = self.threshold_ablation_study(all_results)
        
        # 7. Save raw data
        print("\n" + "="*60)
        print("💾 SAVING RESULTS")
        print("="*60)
        
        results_df.to_csv(self.output_dir / "raw_data" / "all_results.csv", index=False)
        print(f"✓ Raw results: {self.output_dir / 'raw_data' / 'all_results.csv'}")
        
        if threshold_df is not None:
            threshold_df.to_csv(self.output_dir / "raw_data" / "threshold_ablation.csv", index=False)
            print(f"✓ Threshold ablation: {self.output_dir / 'raw_data' / 'threshold_ablation.csv'}")
        
        # Save metrics
        with open(self.output_dir / "raw_data" / "overall_metrics.json", "w") as f:
            json.dump(overall_metrics, f, indent=2)
        print(f"✓ Metrics JSON: {self.output_dir / 'raw_data' / 'overall_metrics.json'}")
        
        # 8. Generate visualizations
        self.generate_figures(results_df, threshold_df, overall_metrics)
        
        # 9. Generate LaTeX tables
        self.generate_latex_tables(overall_metrics, threshold_df)
        
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f"✅ EVALUATION COMPLETE in {elapsed:.1f}s")
        print("="*60)
        print(f"\n📁 Results saved to: {self.output_dir}")
        
        return results_df, threshold_df, overall_metrics
    
    def generate_figures(self, results_df, threshold_df, metrics):
        """
        Generate all visualization figures.
        """
        print("\n📈 Generating figures...")
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 150
        
        # Figure 1: Latency distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        detected = results_df[results_df['detected'] == True]
        
        axes[0].hist(detected['latency_total_ms'], bins=30, color='steelblue', edgecolor='black')
        axes[0].axvline(metrics['mean_latency_ms'], color='red', linestyle='--', linewidth=2, label=f"Mean: {metrics['mean_latency_ms']:.1f}ms")
        axes[0].axvline(metrics['median_latency_ms'], color='green', linestyle='--', linewidth=2, label=f"Median: {metrics['median_latency_ms']:.1f}ms")
        axes[0].set_xlabel('Total Latency (ms)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('CPU Latency Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Latency breakdown
        latency_components = pd.DataFrame({
            'Detection (MTCNN)': [metrics['mean_detection_ms']],
            'Recognition': [metrics['mean_recognition_ms']]
        })
        latency_components.T.plot(kind='barh', ax=axes[1], legend=False, color=['#ff7f0e', '#2ca02c'])
        axes[1].set_xlabel('Mean Latency (ms)', fontsize=12)
        axes[1].set_title('Latency Breakdown', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "latency_analysis.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ latency_analysis.png")
        plt.close()
        
        # Figure 2: Threshold sensitivity (if available)
        if threshold_df is not None and len(threshold_df) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # TPR and FPR
            axes[0, 0].plot(threshold_df['threshold'], threshold_df['tpr'] * 100, 'b-o', linewidth=2, label='TPR (Sensitivity)')
            axes[0, 0].plot(threshold_df['threshold'], threshold_df['fpr'] * 100, 'r-o', linewidth=2, label='FPR')
            axes[0, 0].axvline(x=0.82, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Default (0.82)')
            axes[0, 0].set_xlabel('Threshold', fontsize=12)
            axes[0, 0].set_ylabel('Rate (%)', fontsize=12)
            axes[0, 0].set_title('TPR and FPR vs Threshold', fontsize=14, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            
            # Precision and Recall
            axes[0, 1].plot(threshold_df['threshold'], threshold_df['precision'] * 100, 'b-o', linewidth=2, label='Precision')
            axes[0, 1].plot(threshold_df['threshold'], threshold_df['recall'] * 100, 'r-o', linewidth=2, label='Recall')
            axes[0, 1].axvline(x=0.82, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Default (0.82)')
            axes[0, 1].set_xlabel('Threshold', fontsize=12)
            axes[0, 1].set_ylabel('Score (%)', fontsize=12)
            axes[0, 1].set_title('Precision and Recall vs Threshold', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
            
            # F1-Score
            axes[1, 0].plot(threshold_df['threshold'], threshold_df['f1_score'], 'purple', linewidth=3, marker='o', markersize=6)
            axes[1, 0].axvline(x=0.82, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Default (0.82)')
            best_f1_idx = threshold_df['f1_score'].idxmax()
            best_f1_thresh = threshold_df.loc[best_f1_idx, 'threshold']
            best_f1_val = threshold_df.loc[best_f1_idx, 'f1_score']
            axes[1, 0].plot(best_f1_thresh, best_f1_val, 'r*', markersize=15, label=f'Optimal: {best_f1_thresh:.2f}')
            axes[1, 0].set_xlabel('Threshold', fontsize=12)
            axes[1, 0].set_ylabel('F1-Score', fontsize=12)
            axes[1, 0].set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
            
            # Accuracy
            axes[1, 1].plot(threshold_df['threshold'], threshold_df['accuracy'] * 100, 'green', linewidth=3, marker='o', markersize=6)
            axes[1, 1].axvline(x=0.82, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Default (0.82)')
            axes[1, 1].set_xlabel('Threshold', fontsize=12)
            axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
            axes[1, 1].set_title('Accuracy vs Threshold', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "threshold_sensitivity.png", dpi=150, bbox_inches='tight')
            print(f"  ✓ threshold_sensitivity.png")
            plt.close()
            
            # Figure 3: ROC-like curve (FPR vs TPR)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(threshold_df['fpr'] * 100, threshold_df['tpr'] * 100, 'b-o', linewidth=3, markersize=8)
            ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Random Classifier')
            
            # Mark default threshold
            default_row = threshold_df[threshold_df['threshold'] == 0.82]
            if len(default_row) > 0:
                ax.plot(default_row['fpr'].values[0] * 100, default_row['tpr'].values[0] * 100, 
                       'g*', markersize=20, label='Default (0.82)')
            
            ax.set_xlabel('False Positive Rate (%)', fontsize=13)
            ax.set_ylabel('True Positive Rate (%)', fontsize=13)
            ax.set_title('ROC-Style Curve: TPR vs FPR', fontsize=15, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            ax.set_xlim([-5, 105])
            ax.set_ylim([-5, 105])
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "roc_curve.png", dpi=150, bbox_inches='tight')
            print(f"  ✓ roc_curve.png")
            plt.close()
        
        # Figure 4: Confusion matrix style
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm_data = np.array([
            [metrics['tp'], metrics['fp']],
            [metrics['fn'], metrics['tn']]
        ])
        
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Positive', 'Negative'],
                   yticklabels=['True', 'False'],
                   ax=ax, annot_kws={'size': 16, 'weight': 'bold'})
        
        ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold')
        ax.set_ylabel('Ground Truth', fontsize=13)
        ax.set_xlabel('Prediction', fontsize=13)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "confusion_matrix.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ confusion_matrix.png")
        plt.close()
        
        print("✓ All figures generated")
    
    def generate_latex_tables(self, metrics, threshold_df):
        """
        Generate LaTeX tables for CVPR paper.
        """
        print("\n📝 Generating LaTeX tables...")
        
        # Table 1: Overall Performance Metrics
        table1 = f"""\\begin{{table}}[!t]
\\centering
\\caption{{Face Recognition Performance Metrics (CPU, Gallery Size={len(self.gallery)}, Threshold=0.82)}}
\\label{{tab:face_overall_metrics}}
\\small
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Detection Rate & {metrics['detection_rate']:.2%} \\\\
Accuracy & {metrics['accuracy']:.2%} \\\\
Precision & {metrics['precision']:.2%} \\\\
Recall (TPR) & {metrics['recall']:.2%} \\\\
F1-Score & {metrics['f1_score']:.3f} \\\\
False Positive Rate & {metrics['fpr']:.2%} \\\\
False Negative Rate & {metrics['fnr']:.2%} \\\\
Specificity (TNR) & {metrics['tnr']:.2%} \\\\
\\midrule
\\multicolumn{{2}}{{l}}{{\\textit{{Latency (CPU)}}}} \\\\
Mean Latency & {metrics['mean_latency_ms']:.1f} ms \\\\
Median Latency & {metrics['median_latency_ms']:.1f} ms \\\\
95th Percentile & {metrics['p95_latency_ms']:.1f} ms \\\\
Detection (MTCNN) & {metrics['mean_detection_ms']:.1f} ms \\\\
Recognition & {metrics['mean_recognition_ms']:.1f} ms \\\\
\\midrule
Throughput & {1000/metrics['mean_latency_ms']:.1f} faces/sec \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        
        with open(self.output_dir / "latex" / "overall_metrics_table.tex", "w") as f:
            f.write(table1)
        print(f"  ✓ overall_metrics_table.tex")
        
        # Table 2: Threshold Ablation
        if threshold_df is not None and len(threshold_df) > 0:
            table2 = """\\begin{table}[!t]
\\centering
\\caption{Face Recognition Threshold Ablation Study (CPU Performance)}
\\label{tab:face_threshold_ablation}
\\small
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Threshold} & \\textbf{TPR (\\%)} & \\textbf{FPR (\\%)} & \\textbf{Precision (\\%)} & \\textbf{F1-Score} & \\textbf{Accuracy (\\%)} \\\\
\\midrule
"""
            
            for _, row in threshold_df.iterrows():
                thresh = row['threshold']
                line = f"{thresh:.2f} & {row['tpr']*100:.1f} & {row['fpr']*100:.1f} & {row['precision']*100:.1f} & {row['f1_score']:.3f} & {row['accuracy']*100:.1f} \\\\\n"
                
                # Highlight default threshold
                if thresh == 0.82:
                    line = "\\rowcolor{yellow!20}\n" + line
                
                table2 += line
            
            table2 += """\\bottomrule
\\end{tabular}
\\end{table}
"""
            
            with open(self.output_dir / "latex" / "threshold_ablation_table.tex", "w") as f:
                f.write(table2)
            print(f"  ✓ threshold_ablation_table.tex")
        
        print("✓ LaTeX tables generated")

def main():
    """
    Main execution function.
    """
    print("\n" + "="*70)
    print("🎯 AUTOMATED FACE RECOGNITION EVALUATION SYSTEM")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize evaluator
    evaluator = AutomatedFaceRecognitionEvaluator(
        base_dir=".",
        output_dir="evaluation_results"
    )
    
    # Run full evaluation
    results_df, threshold_df, metrics = evaluator.run_full_evaluation(
        enrollment_size=25,
        threshold=0.82
    )
    
    print("\n" + "="*70)
    print("✅ ALL EVALUATIONS COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"  • Enrolled persons: {len(evaluator.gallery)}")
    print(f"  • Total images tested: {len(results_df)}")
    print(f"  • Overall accuracy: {metrics['accuracy']:.2%}")
    print(f"  • F1-Score: {metrics['f1_score']:.3f}")
    print(f"  • CPU latency (mean): {metrics['mean_latency_ms']:.1f} ms")
    print(f"\n📁 Results location: {evaluator.output_dir}")
    print(f"  • Figures: {evaluator.output_dir / 'figures'}")
    print(f"  • Tables: {evaluator.output_dir / 'latex'}")
    print(f"  • Raw data: {evaluator.output_dir / 'raw_data'}")
    
    print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return evaluator, results_df, threshold_df, metrics

if __name__ == "__main__":
    evaluator, results_df, threshold_df, metrics = main()
