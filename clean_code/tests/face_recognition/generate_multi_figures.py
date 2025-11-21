"""
Generate IEEE-style figures for multi-person face recognition evaluation.
Publication-quality figures at 300 DPI.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Try to use scienceplots, fallback to IEEE style
try:
    plt.style.use(['science', 'ieee', 'no-latex'])
    print("✓ Using scienceplots IEEE style")
except:
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 10,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })
    print("✓ Using fallback IEEE style")

class IEEEFigureGenerator:
    def __init__(self, results_dir="evaluation_results_multi"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load data
        self.metrics = self._load_metrics()
        self.results_df = self._load_results()
        
        print(f"\n{'='*70}")
        print("📊 IEEE Figure Generator - Multi-Person Evaluation")
        print(f"{'='*70}")
        print(f"Results directory: {self.results_dir}")
        print(f"Figures directory: {self.figures_dir}")
        
    def _load_metrics(self):
        with open(self.results_dir / "raw_data" / "overall_metrics.json") as f:
            return json.load(f)
    
    def _load_results(self):
        return pd.read_csv(self.results_dir / "raw_data" / "all_results.csv")
    
    def _save_figure(self, fig, name):
        """Save figure in both PNG and PDF formats"""
        png_path = self.figures_dir / f"{name}.png"
        pdf_path = self.figures_dir / f"{name}.pdf"
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        
        print(f"  ✓ {name}.png")
        print(f"  ✓ {name}.pdf")
    
    def figure1_latency_distribution(self):
        """Latency distribution histogram"""
        print("\n📈 Generating Figure 1: Latency Distribution...")
        
        valid_df = self.results_df[self.results_df['predicted_label'] != 'NO_DETECTION']
        latencies = valid_df['latency_total_ms'].values
        
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        
        # Histogram
        n, bins, patches = ax.hist(latencies, bins=30, edgecolor='black', 
                                    linewidth=0.5, alpha=0.7, color='steelblue')
        
        # Statistics lines
        mean_lat = np.mean(latencies)
        median_lat = np.median(latencies)
        p95_lat = np.percentile(latencies, 95)
        
        ax.axvline(mean_lat, color='red', linestyle='--', linewidth=1.2, 
                   label=f'Mean: {mean_lat:.1f} ms')
        ax.axvline(median_lat, color='green', linestyle='--', linewidth=1.2,
                   label=f'Median: {median_lat:.1f} ms')
        ax.axvline(p95_lat, color='orange', linestyle='--', linewidth=1.2,
                   label=f'95th: {p95_lat:.1f} ms')
        
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Face Recognition Latency Distribution (CPU)')
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        self._save_figure(fig, "figure1_latency_distribution")
        plt.close()
    
    def figure2_confusion_matrix(self):
        """Confusion matrix"""
        print("\n📈 Generating Figure 2: Confusion Matrix...")
        
        tp = self.metrics['tp']
        fp = self.metrics['fp']
        fn = self.metrics['fn']
        tn = self.metrics['tn']
        
        cm = np.array([[tp, fp], [fn, tn]])
        
        fig, ax = plt.subplots(figsize=(3.0, 2.5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Known', 'Unknown'],
                    yticklabels=['Known', 'Unknown'],
                    cbar_kws={'label': 'Count'},
                    linewidths=0.5, linecolor='gray',
                    ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        self._save_figure(fig, "figure2_confusion_matrix")
        plt.close()
    
    def figure3_performance_metrics(self):
        """Performance metrics bar chart"""
        print("\n📈 Generating Figure 3: Performance Metrics...")
        
        metrics_data = {
            'Accuracy': self.metrics['accuracy'] * 100,
            'Precision': self.metrics['precision'] * 100,
            'Recall': self.metrics['recall'] * 100,
            'F1-Score': self.metrics['f1_score'] * 100,
        }
        
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        
        bars = ax.bar(metrics_data.keys(), metrics_data.values(), 
                      color=['steelblue', 'seagreen', 'coral', 'mediumpurple'],
                      edgecolor='black', linewidth=0.8, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Performance Metrics @ Threshold 0.82')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.xticks(rotation=15, ha='right')
        
        self._save_figure(fig, "figure3_performance_metrics")
        plt.close()
    
    def figure4_roc_components(self):
        """TPR and FPR visualization"""
        print("\n📈 Generating Figure 4: ROC Components...")
        
        tpr = self.metrics['tpr'] * 100
        fpr = self.metrics['fpr'] * 100
        
        fig, ax = plt.subplots(figsize=(3.0, 2.5))
        
        categories = ['TPR\n(Sensitivity)', 'FPR\n(Fall-out)']
        values = [tpr, fpr]
        colors = ['seagreen', 'coral']
        
        bars = ax.bar(categories, values, color=colors, 
                      edgecolor='black', linewidth=0.8, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Rate (%)')
        ax.set_title('True Positive Rate vs False Positive Rate')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        self._save_figure(fig, "figure4_roc_components")
        plt.close()
    
    def figure5_latency_breakdown(self):
        """Latency component breakdown"""
        print("\n📈 Generating Figure 5: Latency Breakdown...")
        
        detection_ms = self.metrics['mean_detection_ms']
        recognition_ms = self.metrics['mean_recognition_ms']
        
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        
        components = ['Detection\n(MTCNN)', 'Recognition\n(InceptionResnetV1)']
        times = [detection_ms, recognition_ms]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(components, times, color=colors, 
                      edgecolor='black', linewidth=0.8, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                   f'{height:.1f} ms', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Face Recognition Pipeline Latency (CPU)')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add percentage annotations
        total = detection_ms + recognition_ms
        det_pct = (detection_ms / total) * 100
        rec_pct = (recognition_ms / total) * 100
        
        ax.text(0, detection_ms/2, f'{det_pct:.1f}%', 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        ax.text(1, recognition_ms/2, f'{rec_pct:.1f}%', 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        self._save_figure(fig, "figure5_latency_breakdown")
        plt.close()
    
    def figure6_per_person_accuracy(self):
        """Per-person accuracy comparison"""
        print("\n📈 Generating Figure 6: Per-Subject Accuracy...")
        
        # Calculate per-person accuracy (only known persons)
        known_df = self.results_df[self.results_df['is_unknown'] == False]
        known_df = known_df[known_df['predicted_label'] != 'NO_DETECTION']
        
        persons = []
        accuracies = []
        counts = []
        
        for person in known_df['true_label'].unique():
            person_df = known_df[known_df['true_label'] == person]
            correct = (person_df['correct'] == True).sum()
            total = len(person_df)
            accuracy = (correct / total * 100) if total > 0 else 0
            
            # Use generic labels
            persons.append(f'Subject {len(persons)+1}')
            accuracies.append(accuracy)
            counts.append(total)
        
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        
        bars = ax.bar(persons, accuracies, 
                      color='steelblue', edgecolor='black', 
                      linewidth=0.8, alpha=0.8)
        
        # Add value labels with counts
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%\n(n={count})', 
                   ha='center', va='bottom', fontsize=7)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Recognition Accuracy by Subject')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.xticks(rotation=0)
        
        # Add mean line
        mean_acc = np.mean(accuracies)
        ax.axhline(mean_acc, color='red', linestyle='--', 
                   linewidth=1.0, label=f'Mean: {mean_acc:.1f}%')
        ax.legend(loc='upper right')
        
        self._save_figure(fig, "figure6_per_subject_accuracy")
        plt.close()
    
    def generate_all_figures(self):
        """Generate all figures"""
        print(f"\n{'='*70}")
        print("🎨 Generating IEEE-Style Figures")
        print(f"{'='*70}")
        
        self.figure1_latency_distribution()
        self.figure2_confusion_matrix()
        self.figure3_performance_metrics()
        self.figure4_roc_components()
        self.figure5_latency_breakdown()
        self.figure6_per_person_accuracy()
        
        print(f"\n{'='*70}")
        print("✅ All figures generated successfully!")
        print(f"{'='*70}")
        print(f"\n📁 Figures saved to: {self.figures_dir}")
        print("\nGenerated files:")
        print("  • figure1_latency_distribution (PNG + PDF)")
        print("  • figure2_confusion_matrix (PNG + PDF)")
        print("  • figure3_performance_metrics (PNG + PDF)")
        print("  • figure4_roc_components (PNG + PDF)")
        print("  • figure5_latency_breakdown (PNG + PDF)")
        print("  • figure6_per_subject_accuracy (PNG + PDF)")
        print(f"\n{'='*70}")

def main():
    generator = IEEEFigureGenerator(results_dir="evaluation_results_multi")
    generator.generate_all_figures()

if __name__ == "__main__":
    main()
