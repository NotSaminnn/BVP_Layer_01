"""
Generate Performance Curves and LaTeX Tables for STT Testing Results
Creates visualizations and LaTeX-formatted tables for the CVPR supplementary material.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


class PerformanceVisualizer:
    """Generate visualizations and LaTeX tables from STT test results."""
    
    def __init__(self, results_path: str, output_dir: str):
        """
        Initialize visualizer with results data.
        
        Args:
            results_path: Path to JSON results file
            output_dir: Directory to save visualizations
        """
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loaded {len(self.results)} test results")
        print(f"Output directory: {self.output_dir}")
    
    def plot_latency_distribution(self):
        """Generate latency distribution histogram."""
        latencies = [r['latency_ms'] for r in self.results]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot histogram
        n, bins, patches = ax.hist(latencies, bins=30, edgecolor='black', alpha=0.7)
        
        # Add percentile lines
        percentiles = [50, 95, 99]
        colors = ['green', 'orange', 'red']
        for p, color in zip(percentiles, colors):
            value = np.percentile(latencies, p)
            ax.axvline(value, color=color, linestyle='--', linewidth=2, 
                      label=f'P{p}: {value:.1f}ms')
        
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('STT Latency Distribution (Whisper Medium)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'latency_distribution.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
    
    def plot_wer_by_category(self):
        """Generate WER comparison by query category."""
        categories = defaultdict(list)
        for r in self.results:
            categories[r['category']].append(r['wer'])
        
        # Calculate statistics
        cat_names = []
        cat_means = []
        cat_stds = []
        cat_counts = []
        
        for cat in sorted(categories.keys()):
            cat_names.append(cat.replace('_', ' ').title())
            cat_means.append(np.mean(categories[cat]))
            cat_stds.append(np.std(categories[cat]))
            cat_counts.append(len(categories[cat]))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(cat_names))
        bars = ax.bar(x_pos, cat_means, yerr=cat_stds, capsize=5, 
                      alpha=0.7, edgecolor='black')
        
        # Color bars based on WER
        colors = plt.cm.RdYlGn_r(np.array(cat_means) / max(cat_means))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Query Category', fontsize=12)
        ax.set_ylabel('Word Error Rate (WER)', fontsize=12)
        ax.set_title('STT WER by Query Category', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cat_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, (mean, count) in enumerate(zip(cat_means, cat_counts)):
            ax.text(i, mean + cat_stds[i] + 0.01, f'n={count}', 
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_path = self.output_dir / 'wer_by_category.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
    
    def plot_wer_by_accent(self):
        """Generate WER comparison by accent."""
        accents = defaultdict(list)
        for r in self.results:
            accents[r['accent']].append(r['wer'])
        
        # Calculate statistics
        acc_names = []
        acc_means = []
        acc_stds = []
        acc_counts = []
        
        for acc in sorted(accents.keys()):
            acc_names.append(acc.replace('_', ' ').title())
            acc_means.append(np.mean(accents[acc]))
            acc_stds.append(np.std(accents[acc]))
            acc_counts.append(len(accents[acc]))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_pos = np.arange(len(acc_names))
        bars = ax.bar(x_pos, acc_means, yerr=acc_stds, capsize=5,
                      alpha=0.7, edgecolor='black', color='steelblue')
        
        ax.set_xlabel('Accent', fontsize=12)
        ax.set_ylabel('Word Error Rate (WER)', fontsize=12)
        ax.set_title('STT WER by Accent', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(acc_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, (mean, count) in enumerate(zip(acc_means, acc_counts)):
            ax.text(i, mean + acc_stds[i] + 0.01, f'n={count}', 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / 'wer_by_accent.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
    
    def plot_semantic_similarity_histogram(self):
        """Generate semantic similarity distribution."""
        similarities = [r['semantic_similarity'] for r in self.results]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        n, bins, patches = ax.hist(similarities, bins=30, edgecolor='black', alpha=0.7,
                                    color='mediumseagreen')
        
        # Add mean line
        mean_sim = np.mean(similarities)
        ax.axvline(mean_sim, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_sim:.3f}')
        
        ax.set_xlabel('Semantic Similarity', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('STT Semantic Similarity Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'semantic_similarity_distribution.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
    
    def plot_latency_vs_accuracy(self):
        """Generate scatter plot of latency vs WER."""
        latencies = [r['latency_ms'] for r in self.results]
        wers = [r['wer'] for r in self.results]
        categories = [r['category'] for r in self.results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create color map for categories
        unique_cats = sorted(set(categories))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
        cat_color_map = {cat: colors[i] for i, cat in enumerate(unique_cats)}
        
        for cat in unique_cats:
            cat_indices = [i for i, c in enumerate(categories) if c == cat]
            cat_latencies = [latencies[i] for i in cat_indices]
            cat_wers = [wers[i] for i in cat_indices]
            ax.scatter(cat_latencies, cat_wers, alpha=0.6, s=50,
                      label=cat.replace('_', ' ').title(),
                      color=cat_color_map[cat])
        
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Word Error Rate (WER)', fontsize=12)
        ax.set_title('STT Latency vs Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'latency_vs_wer.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
    
    def plot_error_rate_comparison(self):
        """Generate comparison of WER and CER."""
        categories = defaultdict(lambda: {'wer': [], 'cer': []})
        for r in self.results:
            categories[r['category']]['wer'].append(r['wer'])
            categories[r['category']]['cer'].append(r['cer'])
        
        cat_names = []
        wer_means = []
        cer_means = []
        
        for cat in sorted(categories.keys()):
            cat_names.append(cat.replace('_', ' ').title())
            wer_means.append(np.mean(categories[cat]['wer']))
            cer_means.append(np.mean(categories[cat]['cer']))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(cat_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, wer_means, width, label='WER', alpha=0.8,
                       edgecolor='black', color='coral')
        bars2 = ax.bar(x + width/2, cer_means, width, label='CER', alpha=0.8,
                       edgecolor='black', color='skyblue')
        
        ax.set_xlabel('Query Category', fontsize=12)
        ax.set_ylabel('Error Rate', fontsize=12)
        ax.set_title('WER vs CER by Category', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cat_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'wer_cer_comparison.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
    
    def generate_latex_table_overall(self):
        """Generate LaTeX table for overall performance."""
        wers = [r['wer'] for r in self.results]
        cers = [r['cer'] for r in self.results]
        sims = [r['semantic_similarity'] for r in self.results]
        lats = [r['latency_ms'] for r in self.results]
        
        latex = r"""\begin{table}[h]
\centering
\caption{Overall STT Performance (Whisper Medium)}
\label{tab:stt_overall}
\begin{tabular}{lcccc}
\toprule
\textbf{Metric} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\
\midrule
"""
        latex += f"WER & {np.mean(wers):.4f} & {np.std(wers):.4f} & {np.min(wers):.4f} & {np.max(wers):.4f} \\\\\n"
        latex += f"CER & {np.mean(cers):.4f} & {np.std(cers):.4f} & {np.min(cers):.4f} & {np.max(cers):.4f} \\\\\n"
        latex += f"Semantic Sim. & {np.mean(sims):.4f} & {np.std(sims):.4f} & {np.min(sims):.4f} & {np.max(sims):.4f} \\\\\n"
        latex += f"Latency (ms) & {np.mean(lats):.1f} & {np.std(lats):.1f} & {np.min(lats):.1f} & {np.max(lats):.1f} \\\\\n"
        
        latex += r"""\midrule
\multicolumn{5}{l}{\textit{Latency Percentiles:}} \\
"""
        latex += f"P50 (Median) & \\multicolumn{{4}}{{l}}{{{np.percentile(lats, 50):.1f} ms}} \\\\\n"
        latex += f"P95 & \\multicolumn{{4}}{{l}}{{{np.percentile(lats, 95):.1f} ms}} \\\\\n"
        latex += f"P99 & \\multicolumn{{4}}{{l}}{{{np.percentile(lats, 99):.1f} ms}} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        output_path = self.output_dir / 'latex_table_overall.tex'
        with open(output_path, 'w') as f:
            f.write(latex)
        
        print(f"✅ Saved: {output_path}")
        return latex
    
    def generate_latex_table_by_category(self):
        """Generate LaTeX table for performance by category."""
        categories = defaultdict(lambda: {'wer': [], 'cer': [], 'semantic': [], 'latency': []})
        for r in self.results:
            cat = r['category']
            categories[cat]['wer'].append(r['wer'])
            categories[cat]['cer'].append(r['cer'])
            categories[cat]['semantic'].append(r['semantic_similarity'])
            categories[cat]['latency'].append(r['latency_ms'])
        
        latex = r"""\begin{table}[h]
\centering
\caption{STT Performance by Query Category}
\label{tab:stt_by_category}
\begin{tabular}{lccccc}
\toprule
\textbf{Category} & \textbf{n} & \textbf{WER} & \textbf{CER} & \textbf{Sem. Sim.} & \textbf{Latency (ms)} \\
\midrule
"""
        
        for cat in sorted(categories.keys()):
            cat_name = cat.replace('_', '\\_')
            n = len(categories[cat]['wer'])
            wer_mean = np.mean(categories[cat]['wer'])
            cer_mean = np.mean(categories[cat]['cer'])
            sem_mean = np.mean(categories[cat]['semantic'])
            lat_mean = np.mean(categories[cat]['latency'])
            
            latex += f"{cat_name} & {n} & {wer_mean:.4f} & {cer_mean:.4f} & {sem_mean:.3f} & {lat_mean:.1f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        output_path = self.output_dir / 'latex_table_by_category.tex'
        with open(output_path, 'w') as f:
            f.write(latex)
        
        print(f"✅ Saved: {output_path}")
        return latex
    
    def generate_latex_table_by_accent(self):
        """Generate LaTeX table for performance by accent."""
        accents = defaultdict(lambda: {'wer': [], 'cer': [], 'semantic': [], 'latency': []})
        for r in self.results:
            acc = r['accent']
            accents[acc]['wer'].append(r['wer'])
            accents[acc]['cer'].append(r['cer'])
            accents[acc]['semantic'].append(r['semantic_similarity'])
            accents[acc]['latency'].append(r['latency_ms'])
        
        latex = r"""\begin{table}[h]
\centering
\caption{STT Performance by Accent}
\label{tab:stt_by_accent}
\begin{tabular}{lccccc}
\toprule
\textbf{Accent} & \textbf{n} & \textbf{WER} & \textbf{CER} & \textbf{Sem. Sim.} & \textbf{Latency (ms)} \\
\midrule
"""
        
        for acc in sorted(accents.keys()):
            acc_name = acc.replace('_', ' ').title()
            n = len(accents[acc]['wer'])
            wer_mean = np.mean(accents[acc]['wer'])
            cer_mean = np.mean(accents[acc]['cer'])
            sem_mean = np.mean(accents[acc]['semantic'])
            lat_mean = np.mean(accents[acc]['latency'])
            
            latex += f"{acc_name} & {n} & {wer_mean:.4f} & {cer_mean:.4f} & {sem_mean:.3f} & {lat_mean:.1f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        output_path = self.output_dir / 'latex_table_by_accent.tex'
        with open(output_path, 'w') as f:
            f.write(latex)
        
        print(f"✅ Saved: {output_path}")
        return latex
    
    def generate_all_visualizations(self):
        """Generate all visualizations and tables."""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS AND TABLES")
        print("=" * 70 + "\n")
        
        print("📊 Generating plots...")
        self.plot_latency_distribution()
        self.plot_wer_by_category()
        self.plot_wer_by_accent()
        self.plot_semantic_similarity_histogram()
        self.plot_latency_vs_accuracy()
        self.plot_error_rate_comparison()
        
        print("\n📄 Generating LaTeX tables...")
        self.generate_latex_table_overall()
        self.generate_latex_table_by_category()
        self.generate_latex_table_by_accent()
        
        print("\n" + "=" * 70)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print("=" * 70)
        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"   - 6 PNG plots")
        print(f"   - 3 LaTeX tables")


def main():
    """Main function to generate visualizations."""
    script_dir = Path(__file__).parent
    results_path = script_dir / 'results' / 'stt_detailed_results.json'
    output_dir = script_dir / 'results' / 'visualizations'
    
    if not results_path.exists():
        print(f"❌ Error: Results file not found at {results_path}")
        print("Please run test_stt_performance.py first!")
        return
    
    visualizer = PerformanceVisualizer(str(results_path), str(output_dir))
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
