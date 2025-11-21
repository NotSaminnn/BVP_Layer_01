"""
Generate IEEE-style figures for face recognition evaluation.
Uses scienceplots package with IEEE style (no LaTeX rendering).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import scienceplots

# Configure matplotlib for IEEE style without LaTeX
try:
    plt.style.use(['science', 'ieee', 'no-latex'])
except:
    # Fallback: Manual IEEE-style configuration
    plt.style.use('default')
    
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.fontsize': 8,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'figure.titlesize': 12,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'axes.grid': False,
    'figure.figsize': (3.5, 2.625),  # IEEE column width
    'figure.autolayout': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01
})

def load_data():
    """Load all evaluation data."""
    base_dir = Path("evaluation_results/raw_data")
    
    # Load results
    results_df = pd.read_csv(base_dir / "all_results.csv")
    threshold_df = pd.read_csv(base_dir / "threshold_ablation.csv")
    
    with open(base_dir / "overall_metrics.json", "r") as f:
        metrics = json.load(f)
    
    return results_df, threshold_df, metrics

def figure1_latency_distribution(results_df, metrics):
    """
    Figure 1: CPU Latency Analysis
    Two-panel figure showing latency distribution and breakdown.
    """
    detected = results_df[results_df['detected'] == True]
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
    
    # Panel A: Latency histogram
    axes[0].hist(detected['latency_total_ms'], bins=30, 
                color='#2E86AB', edgecolor='black', linewidth=0.5, alpha=0.8)
    axes[0].axvline(metrics['mean_latency_ms'], color='#A23B72', 
                   linestyle='--', linewidth=1.5, label=f"Mean: {metrics['mean_latency_ms']:.1f} ms")
    axes[0].axvline(metrics['median_latency_ms'], color='#F18F01', 
                   linestyle='-.', linewidth=1.5, label=f"Median: {metrics['median_latency_ms']:.1f} ms")
    axes[0].set_xlabel('Total Latency (ms)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('(a) Latency Distribution')
    axes[0].legend(frameon=True, loc='upper right')
    axes[0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel B: Latency breakdown
    components = ['Detection\n(MTCNN)', 'Recognition\n(InceptionResnet)']
    times = [metrics['mean_detection_ms'], metrics['mean_recognition_ms']]
    colors = ['#2E86AB', '#F18F01']
    
    bars = axes[1].barh(components, times, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Mean Latency (ms)')
    axes[1].set_title('(b) Latency Breakdown')
    axes[1].grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        axes[1].text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                    f'{time:.1f} ms', va='center', ha='left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/figures/ieee_latency_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('evaluation_results/figures/ieee_latency_analysis.pdf', bbox_inches='tight')
    print("  ✓ ieee_latency_analysis.png/pdf")
    plt.close()

def figure2_threshold_sensitivity(threshold_df):
    """
    Figure 2: Threshold Ablation Study (4-panel)
    Shows TPR/FPR, Precision/Recall, F1-Score, and Accuracy vs Threshold.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    
    # Panel A: TPR and FPR
    ax1 = axes[0, 0]
    ax1.plot(threshold_df['threshold'], threshold_df['tpr'] * 100, 
            'o-', color='#2E86AB', linewidth=1.5, markersize=4, label='TPR (Sensitivity)')
    ax1.plot(threshold_df['threshold'], threshold_df['fpr'] * 100, 
            's-', color='#A23B72', linewidth=1.5, markersize=4, label='FPR')
    ax1.axvline(x=0.82, color='#666666', linestyle='--', linewidth=1, alpha=0.7, label='Default (0.82)')
    ax1.set_xlabel('Similarity Threshold')
    ax1.set_ylabel('Rate (%)')
    ax1.set_title('(a) True/False Positive Rates')
    ax1.legend(frameon=True, loc='best')
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.set_xlim([0.58, 0.97])
    
    # Panel B: Precision and Recall
    ax2 = axes[0, 1]
    ax2.plot(threshold_df['threshold'], threshold_df['precision'] * 100, 
            '^-', color='#F18F01', linewidth=1.5, markersize=4, label='Precision')
    ax2.plot(threshold_df['threshold'], threshold_df['recall'] * 100, 
            'v-', color='#2E86AB', linewidth=1.5, markersize=4, label='Recall')
    ax2.axvline(x=0.82, color='#666666', linestyle='--', linewidth=1, alpha=0.7, label='Default (0.82)')
    ax2.set_xlabel('Similarity Threshold')
    ax2.set_ylabel('Score (%)')
    ax2.set_title('(b) Precision and Recall')
    ax2.legend(frameon=True, loc='best')
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.set_xlim([0.58, 0.97])
    
    # Panel C: F1-Score
    ax3 = axes[1, 0]
    ax3.plot(threshold_df['threshold'], threshold_df['f1_score'], 
            'D-', color='#6A4C93', linewidth=2, markersize=5, label='F1-Score')
    ax3.axvline(x=0.82, color='#666666', linestyle='--', linewidth=1, alpha=0.7, label='Default (0.82)')
    
    # Find and mark optimal F1
    best_f1_idx = threshold_df['f1_score'].idxmax()
    best_f1_thresh = threshold_df.loc[best_f1_idx, 'threshold']
    best_f1_val = threshold_df.loc[best_f1_idx, 'f1_score']
    ax3.plot(best_f1_thresh, best_f1_val, '*', color='#E63946', 
            markersize=12, label=f'Optimal: {best_f1_thresh:.2f}')
    
    ax3.set_xlabel('Similarity Threshold')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('(c) F1-Score vs Threshold')
    ax3.legend(frameon=True, loc='best')
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax3.set_xlim([0.58, 0.97])
    ax3.set_ylim([0, 1.05])
    
    # Panel D: Accuracy
    ax4 = axes[1, 1]
    ax4.plot(threshold_df['threshold'], threshold_df['accuracy'] * 100, 
            'o-', color='#06A77D', linewidth=2, markersize=5, label='Accuracy')
    ax4.axvline(x=0.82, color='#666666', linestyle='--', linewidth=1, alpha=0.7, label='Default (0.82)')
    ax4.set_xlabel('Similarity Threshold')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('(d) Accuracy vs Threshold')
    ax4.legend(frameon=True, loc='best')
    ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax4.set_xlim([0.58, 0.97])
    ax4.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('evaluation_results/figures/ieee_threshold_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.savefig('evaluation_results/figures/ieee_threshold_sensitivity.pdf', bbox_inches='tight')
    print("  ✓ ieee_threshold_sensitivity.png/pdf")
    plt.close()

def figure3_roc_curve(threshold_df):
    """
    Figure 3: ROC-Style Curve (TPR vs FPR)
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Plot ROC curve
    ax.plot(threshold_df['fpr'] * 100, threshold_df['tpr'] * 100, 
           'o-', color='#2E86AB', linewidth=2, markersize=5, label='System Performance')
    
    # Diagonal line (random classifier)
    ax.plot([0, 100], [0, 100], '--', color='#999999', linewidth=1.5, label='Random Classifier')
    
    # Mark default threshold
    default_row = threshold_df[threshold_df['threshold'] == 0.82]
    if len(default_row) > 0:
        ax.plot(default_row['fpr'].values[0] * 100, default_row['tpr'].values[0] * 100, 
               '*', color='#E63946', markersize=15, label='Default (0.82)', zorder=5)
    
    # Mark optimal point (highest TPR with FPR = 0)
    optimal_row = threshold_df[threshold_df['fpr'] == 0].sort_values('tpr', ascending=False).iloc[0]
    ax.plot(optimal_row['fpr'] * 100, optimal_row['tpr'] * 100, 
           's', color='#06A77D', markersize=10, label=f"Optimal (t={optimal_row['threshold']:.2f})", zorder=5)
    
    ax.set_xlabel('False Positive Rate (%)')
    ax.set_ylabel('True Positive Rate (%)')
    ax.set_title('ROC Curve: Face Recognition Performance')
    ax.legend(frameon=True, loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_xlim([-5, 105])
    ax.set_ylim([-5, 105])
    ax.set_aspect('equal')
    
    # Add text annotation for AUC (approximate)
    auc_approx = np.trapz(threshold_df['tpr'], threshold_df['fpr'])
    ax.text(0.98, 0.02, f'AUC ≈ {abs(auc_approx):.3f}', 
           transform=ax.transAxes, fontsize=8, 
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5))
    
    plt.tight_layout()
    plt.savefig('evaluation_results/figures/ieee_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig('evaluation_results/figures/ieee_roc_curve.pdf', bbox_inches='tight')
    print("  ✓ ieee_roc_curve.png/pdf")
    plt.close()

def figure4_confusion_matrix(metrics):
    """
    Figure 4: Confusion Matrix
    """
    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    # Confusion matrix data
    cm_data = np.array([
        [metrics['tp'], metrics['fn']],
        [metrics['fp'], metrics['tn']]
    ])
    
    # Create heatmap
    im = ax.imshow(cm_data, cmap='Blues', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Positive', 'Negative'])
    ax.set_yticklabels(['Positive', 'Negative'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            value = cm_data[i, j]
            color = 'white' if value > cm_data.max() / 2 else 'black'
            ax.text(j, i, f'{int(value)}', ha='center', va='center', 
                   color=color, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/figures/ieee_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig('evaluation_results/figures/ieee_confusion_matrix.pdf', bbox_inches='tight')
    print("  ✓ ieee_confusion_matrix.png/pdf")
    plt.close()

def figure5_performance_summary(threshold_df, metrics):
    """
    Figure 5: Performance Summary Bar Chart
    Compares key metrics at different thresholds.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    
    # Select key thresholds
    key_thresholds = [0.65, 0.70, 0.75, 0.82]
    data = threshold_df[threshold_df['threshold'].isin(key_thresholds)].copy()
    
    # Metrics to plot
    x = np.arange(len(key_thresholds))
    width = 0.2
    
    metrics_to_plot = {
        'Accuracy': data['accuracy'].values * 100,
        'Precision': data['precision'].values * 100,
        'Recall': data['recall'].values * 100,
        'F1-Score': data['f1_score'].values * 100
    }
    
    colors = ['#2E86AB', '#F18F01', '#06A77D', '#6A4C93']
    
    for i, (metric_name, values) in enumerate(metrics_to_plot.items()):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=metric_name, 
                     color=colors[i], edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=6)
    
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Comparison Across Thresholds')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.2f}' for t in key_thresholds])
    ax.legend(frameon=True, loc='lower left', ncol=2)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_ylim([0, 110])
    
    # Highlight default threshold
    default_idx = key_thresholds.index(0.82)
    ax.axvspan(default_idx - 0.5, default_idx + 0.5, alpha=0.1, color='red', zorder=-1)
    ax.text(default_idx, 105, 'Default', ha='center', fontsize=7, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red', linewidth=0.5))
    
    plt.tight_layout()
    plt.savefig('evaluation_results/figures/ieee_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('evaluation_results/figures/ieee_performance_comparison.pdf', bbox_inches='tight')
    print("  ✓ ieee_performance_comparison.png/pdf")
    plt.close()

def figure6_latency_vs_accuracy(threshold_df, metrics):
    """
    Figure 6: Accuracy vs Latency Trade-off
    Shows relationship between threshold choice and system performance.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    
    # Create scatter plot with threshold as color
    scatter = ax.scatter(threshold_df['accuracy'] * 100, 
                        [metrics['mean_latency_ms']] * len(threshold_df),
                        c=threshold_df['threshold'], cmap='viridis', 
                        s=100, edgecolors='black', linewidth=0.5, alpha=0.8)
    
    # Mark default threshold
    default_row = threshold_df[threshold_df['threshold'] == 0.82]
    if len(default_row) > 0:
        ax.scatter(default_row['accuracy'].values[0] * 100, 
                  metrics['mean_latency_ms'],
                  marker='*', s=300, c='red', edgecolors='black', 
                  linewidth=1, label='Default (0.82)', zorder=5)
    
    ax.set_xlabel('Accuracy (%)')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('Accuracy vs Processing Latency')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Threshold')
    
    ax.legend(frameon=True, loc='best')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add annotation
    ax.text(0.02, 0.98, f'Latency constant at {metrics["mean_latency_ms"]:.1f} ms\n(threshold affects accuracy only)', 
           transform=ax.transAxes, fontsize=7, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5))
    
    plt.tight_layout()
    plt.savefig('evaluation_results/figures/ieee_accuracy_latency.png', dpi=300, bbox_inches='tight')
    plt.savefig('evaluation_results/figures/ieee_accuracy_latency.pdf', bbox_inches='tight')
    print("  ✓ ieee_accuracy_latency.png/pdf")
    plt.close()

def main():
    """Generate all IEEE-style figures."""
    print("\n" + "="*60)
    print("📊 GENERATING IEEE-STYLE FIGURES")
    print("="*60)
    print("Style: science + ieee + no-latex")
    print("Resolution: 300 DPI")
    print("Formats: PNG + PDF")
    
    # Load data
    print("\n📂 Loading evaluation data...")
    results_df, threshold_df, metrics = load_data()
    print("  ✓ Data loaded")
    
    # Generate figures
    print("\n🎨 Generating figures...")
    figure1_latency_distribution(results_df, metrics)
    figure2_threshold_sensitivity(threshold_df)
    figure3_roc_curve(threshold_df)
    figure4_confusion_matrix(metrics)
    figure5_performance_summary(threshold_df, metrics)
    figure6_latency_vs_accuracy(threshold_df, metrics)
    
    print("\n" + "="*60)
    print("✅ ALL FIGURES GENERATED")
    print("="*60)
    print("\n📁 Location: evaluation_results/figures/")
    print("  • ieee_latency_analysis.png/pdf")
    print("  • ieee_threshold_sensitivity.png/pdf")
    print("  • ieee_roc_curve.png/pdf")
    print("  • ieee_confusion_matrix.png/pdf")
    print("  • ieee_performance_comparison.png/pdf")
    print("  • ieee_accuracy_latency.png/pdf")

if __name__ == "__main__":
    main()
