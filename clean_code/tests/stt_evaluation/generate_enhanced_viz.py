"""
Generate additional publication-quality STT performance visualizations
Using scienceplots style for CVPR supplementary material
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Try to use scienceplots style
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
    print("✅ Using scienceplots style")
except ImportError:
    print("⚠️  scienceplots not installed, using default style")
    plt.style.use('seaborn-v0_8-darkgrid')

# Set publication-quality parameters
plt.rcParams.update({
    'font.size': 13,
    'font.weight': 'heavy',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


class EnhancedSTTVisualizer:
    """Generate enhanced publication-quality STT visualizations."""
    
    def __init__(self, results_path: str, output_dir: str):
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loaded {len(self.results)} test results")
    
    def plot_combined_metrics_bars(self):
        """Combined bar chart showing WER, CER, and Semantic Similarity by category."""
        categories = defaultdict(lambda: {'wer': [], 'cer': [], 'semantic': []})
        for r in self.results:
            cat = r['category']
            categories[cat]['wer'].append(r['wer'])
            categories[cat]['cer'].append(r['cer'])
            categories[cat]['semantic'].append(r['semantic_similarity'])
        
        cat_names = []
        wer_means = []
        cer_means = []
        sem_means = []
        
        for cat in sorted(categories.keys()):
            cat_names.append(cat.replace('_', '\n'))
            wer_means.append(np.mean(categories[cat]['wer']))
            cer_means.append(np.mean(categories[cat]['cer']))
            sem_means.append(np.mean(categories[cat]['semantic']))
        
        x = np.arange(len(cat_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        bars1 = ax.bar(x - width, wer_means, width, label='WER', 
                       color='tab:red', edgecolor='black', linewidth=1.2, alpha=0.85)
        bars2 = ax.bar(x, cer_means, width, label='CER',
                       color='tab:orange', edgecolor='black', linewidth=1.2, alpha=0.85)
        bars3 = ax.bar(x + width, sem_means, width, label='Semantic Sim.',
                       color='tab:green', edgecolor='black', linewidth=1.2, alpha=0.85)
        
        ax.set_xlabel('Query Category', fontweight='bold')
        ax.set_ylabel('Error Rate / Similarity', fontweight='bold')
        ax.set_title('STT Performance Metrics by Category', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_names, fontsize=10)
        ax.legend(fontsize=11, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'combined_metrics_by_category.png'
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_accent_performance_radar(self):
        """Radar chart showing performance across different accents."""
        accents = defaultdict(lambda: {'wer': [], 'cer': [], 'semantic': [], 'latency': []})
        for r in self.results:
            acc = r['accent']
            accents[acc]['wer'].append(r['wer'])
            accents[acc]['cer'].append(r['cer'])
            accents[acc]['semantic'].append(r['semantic_similarity'])
            accents[acc]['latency'].append(r['latency_ms'])
        
        # Normalize metrics for radar chart (0-1 scale)
        acc_names = sorted(accents.keys())
        wer_norm = []
        cer_norm = []
        sem_norm = []
        lat_norm = []
        
        all_wer = [np.mean(accents[a]['wer']) for a in acc_names]
        all_cer = [np.mean(accents[a]['cer']) for a in acc_names]
        all_sem = [np.mean(accents[a]['semantic']) for a in acc_names]
        all_lat = [np.mean(accents[a]['latency']) for a in acc_names]
        
        max_wer = max(all_wer)
        max_cer = max(all_cer)
        min_sem = min(all_sem)
        max_sem = max(all_sem)
        max_lat = max(all_lat)
        
        fig, ax = plt.subplots(figsize=(10, 5), ncols=2)
        
        # Bar chart version for clarity
        x = np.arange(len(acc_names))
        
        # Subplot 1: Error rates
        ax[0].bar(x - 0.2, all_wer, 0.4, label='WER', color='tab:red', 
                  edgecolor='black', linewidth=1.2, alpha=0.85)
        ax[0].bar(x + 0.2, all_cer, 0.4, label='CER', color='tab:orange',
                  edgecolor='black', linewidth=1.2, alpha=0.85)
        ax[0].set_xticks(x)
        ax[0].set_xticklabels([a.replace('_', '\n') for a in acc_names], fontsize=10)
        ax[0].set_ylabel('Error Rate', fontweight='bold')
        ax[0].set_title('Error Rates by Accent', fontweight='bold')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: Semantic similarity
        bars = ax[1].bar(x, all_sem, color='tab:green', edgecolor='black', 
                         linewidth=1.2, alpha=0.85)
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([a.replace('_', '\n') for a in acc_names], fontsize=10)
        ax[1].set_ylabel('Semantic Similarity', fontweight='bold')
        ax[1].set_title('Semantic Similarity by Accent', fontweight='bold')
        ax[1].set_ylim([0.94, 1.0])
        ax[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, all_sem):
            height = bar.get_height()
            ax[1].text(bar.get_x() + bar.get_width()/2, height - 0.005,
                      f'{val:.3f}', ha='center', va='top', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'accent_performance_comparison.png'
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_latency_breakdown_box(self):
        """Box plot showing latency distribution by category and accent."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # By category
        categories = defaultdict(list)
        for r in self.results:
            categories[r['category']].append(r['latency_ms'] / 1000)  # Convert to seconds
        
        cat_names = [c.replace('_', '\n') for c in sorted(categories.keys())]
        cat_data = [categories[c] for c in sorted(categories.keys())]
        
        bp1 = ax1.boxplot(cat_data, labels=cat_names, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.2),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(color='black', linewidth=1.2),
                          capprops=dict(color='black', linewidth=1.2))
        
        ax1.set_ylabel('Latency (seconds)', fontweight='bold')
        ax1.set_title('Latency Distribution by Category', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # By accent
        accents = defaultdict(list)
        for r in self.results:
            accents[r['accent']].append(r['latency_ms'] / 1000)
        
        acc_names = [a.replace('_', '\n') for a in sorted(accents.keys())]
        acc_data = [accents[a] for a in sorted(accents.keys())]
        
        bp2 = ax2.boxplot(acc_data, labels=acc_names, patch_artist=True,
                          boxprops=dict(facecolor='lightgreen', edgecolor='black', linewidth=1.2),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(color='black', linewidth=1.2),
                          capprops=dict(color='black', linewidth=1.2))
        
        ax2.set_ylabel('Latency (seconds)', fontweight='bold')
        ax2.set_title('Latency Distribution by Accent', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'latency_boxplot_comparison.png'
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_difficulty_analysis(self):
        """Analysis of performance by expected difficulty level."""
        difficulties = defaultdict(lambda: {'wer': [], 'latency': [], 'semantic': []})
        for r in self.results:
            diff = r['difficulty']
            difficulties[diff]['wer'].append(r['wer'])
            difficulties[diff]['latency'].append(r['latency_ms'] / 1000)
            difficulties[diff]['semantic'].append(r['semantic_similarity'])
        
        diff_order = ['easy', 'medium', 'hard']
        diff_names = [d.capitalize() for d in diff_order]
        
        wer_means = [np.mean(difficulties[d]['wer']) for d in diff_order]
        wer_stds = [np.std(difficulties[d]['wer']) for d in diff_order]
        lat_means = [np.mean(difficulties[d]['latency']) for d in diff_order]
        lat_stds = [np.std(difficulties[d]['latency']) for d in diff_order]
        
        x = np.arange(len(diff_names))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        # WER bars
        bars1 = ax1.bar(x - width/2, wer_means, width, yerr=wer_stds, capsize=5,
                        label='WER', color='tab:red', edgecolor='black', 
                        linewidth=1.2, alpha=0.85)
        
        ax1.set_xlabel('Query Difficulty', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Word Error Rate', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(diff_names, fontsize=11)
        ax1.set_title('STT Performance vs Query Difficulty', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Latency on secondary axis
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, lat_means, width, yerr=lat_stds, capsize=5,
                        label='Latency', color='tab:blue', edgecolor='black',
                        linewidth=1.2, alpha=0.85)
        
        ax2.set_ylabel('Latency (seconds)', fontweight='bold', fontsize=12)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
        
        # Add value labels
        for i, (wer, lat) in enumerate(zip(wer_means, lat_means)):
            ax1.text(i - width/2, wer + wer_stds[i] + 0.01, f'{wer:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax2.text(i + width/2, lat + lat_stds[i] + 0.2, f'{lat:.1f}s',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'difficulty_performance_analysis.png'
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_noise_impact_comparison(self):
        """Compare performance between clean and noisy conditions."""
        noise_levels = defaultdict(lambda: {'wer': [], 'cer': [], 'semantic': [], 'latency': []})
        for r in self.results:
            noise = r['noise_level']
            noise_levels[noise]['wer'].append(r['wer'])
            noise_levels[noise]['cer'].append(r['cer'])
            noise_levels[noise]['semantic'].append(r['semantic_similarity'])
            noise_levels[noise]['latency'].append(r['latency_ms'] / 1000)
        
        noise_order = ['clean', 'low_noise']
        noise_names = ['Clean', 'Low Noise']
        
        metrics = ['WER', 'CER', 'Semantic Sim.', 'Latency (s)']
        clean_vals = [
            np.mean(noise_levels['clean']['wer']),
            np.mean(noise_levels['clean']['cer']),
            np.mean(noise_levels['clean']['semantic']),
            np.mean(noise_levels['clean']['latency'])
        ]
        noisy_vals = [
            np.mean(noise_levels['low_noise']['wer']),
            np.mean(noise_levels['low_noise']['cer']),
            np.mean(noise_levels['low_noise']['semantic']),
            np.mean(noise_levels['low_noise']['latency'])
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        bars1 = ax.bar(x - width/2, clean_vals, width, label='Clean',
                       color='tab:green', edgecolor='black', linewidth=1.2, alpha=0.85)
        bars2 = ax.bar(x + width/2, noisy_vals, width, label='Low Noise',
                       color='tab:orange', edgecolor='black', linewidth=1.2, alpha=0.85)
        
        ax.set_xlabel('Performance Metric', fontweight='bold', fontsize=12)
        ax.set_ylabel('Value', fontweight='bold', fontsize=12)
        ax.set_title('Clean vs Noisy Audio Performance', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=11, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (c_val, n_val) in enumerate(zip(clean_vals, noisy_vals)):
            ax.text(i - width/2, c_val + 0.01, f'{c_val:.3f}' if i < 3 else f'{c_val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(i + width/2, n_val + 0.01, f'{n_val:.3f}' if i < 3 else f'{n_val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'noise_impact_comparison.png'
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def plot_comprehensive_heatmap(self):
        """Heatmap showing WER across category and accent combinations."""
        # Create matrix of WER values
        categories = sorted(set(r['category'] for r in self.results))
        accents = sorted(set(r['accent'] for r in self.results))
        
        matrix = np.zeros((len(categories), len(accents)))
        counts = np.zeros((len(categories), len(accents)))
        
        for r in self.results:
            cat_idx = categories.index(r['category'])
            acc_idx = accents.index(r['accent'])
            matrix[cat_idx, acc_idx] += r['wer']
            counts[cat_idx, acc_idx] += 1
        
        # Average WER
        matrix = np.divide(matrix, counts, where=counts > 0)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.3)
        
        # Set ticks
        ax.set_xticks(np.arange(len(accents)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels([a.replace('_', ' ').title() for a in accents], rotation=45, ha='right')
        ax.set_yticklabels([c.replace('_', ' ').title() for c in categories])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Word Error Rate', fontweight='bold', fontsize=12)
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(accents)):
                if counts[i, j] > 0:
                    text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                                 ha='center', va='center', color='black',
                                 fontsize=9, fontweight='bold')
        
        ax.set_title('WER Heatmap: Category × Accent', fontweight='bold', fontsize=14)
        ax.set_xlabel('Accent', fontweight='bold', fontsize=12)
        ax.set_ylabel('Query Category', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        output_path = self.output_dir / 'wer_heatmap_category_accent.png'
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
    
    def generate_all_enhanced_visualizations(self):
        """Generate all enhanced visualizations."""
        print("\n" + "=" * 70)
        print("GENERATING ENHANCED PUBLICATION-QUALITY VISUALIZATIONS")
        print("=" * 70 + "\n")
        
        self.plot_combined_metrics_bars()
        self.plot_accent_performance_radar()
        self.plot_latency_breakdown_box()
        self.plot_difficulty_analysis()
        self.plot_noise_impact_comparison()
        self.plot_comprehensive_heatmap()
        
        print("\n" + "=" * 70)
        print("✅ ALL ENHANCED VISUALIZATIONS GENERATED")
        print("=" * 70)


def main():
    script_dir = Path(__file__).parent
    results_path = script_dir / 'results' / 'stt_detailed_results.json'
    output_dir = script_dir / 'results' / 'visualizations'
    
    if not results_path.exists():
        print(f"❌ Error: Results file not found at {results_path}")
        return
    
    visualizer = EnhancedSTTVisualizer(str(results_path), str(output_dir))
    visualizer.generate_all_enhanced_visualizations()


if __name__ == "__main__":
    main()
