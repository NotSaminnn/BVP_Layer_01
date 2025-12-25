"""
Script 6: Generate TTS Visualizations

Creates publication-quality visualizations for CVPR submission:
- MOS distribution by voice
- Latency vs text length
- Speaking rate comparison
- Voice comparison radar chart
- Overall performance dashboard

Requires: completed quality evaluation (results/quality_metrics.csv)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Use scienceplots IEEE style without LaTeX
try:
    import scienceplots
    plt.style.use(['science', 'ieee', 'no-latex'])
    print("✓ Using SciencePlots IEEE style (no LaTeX)")
except ImportError:
    print("⚠️  scienceplots not installed. Using default style")
    print("   Install with: pip install scienceplots")
    plt.style.use('seaborn-v0_8-whitegrid')

# Disable LaTeX rendering explicitly
plt.rcParams['text.usetex'] = False
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'en-US-GuyNeural': '#2E86AB',       # Blue
    'en-US-JennyNeural': '#A23B72',     # Purple
    'en-GB-RyanNeural': '#F18F01',      # Orange
    'en-IN-NeerjaNeural': '#C73E1D',    # Red
}

def load_data(results_dir: Path) -> pd.DataFrame:
    """Load quality metrics data."""
    metrics_path = results_dir / "quality_metrics.csv"
    
    if not metrics_path.exists():
        print(f"Error: Metrics file not found at {metrics_path}")
        print("Please run 3_evaluate_quality.py first.")
        return None
    
    df = pd.read_csv(metrics_path)
    return df

def plot_mos_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot speech rate distribution by voice."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    voices = df['voice'].unique()
    positions = range(len(voices))
    
    speech_data = [df[df['voice'] == voice]['speech_rate_wpm'].dropna() for voice in voices]
    
    bp = ax.boxplot(speech_data, positions=positions, labels=voices,
                    patch_artist=True, widths=0.6)
    
    # Color boxes
    for patch, voice in zip(bp['boxes'], voices):
        patch.set_facecolor(COLORS.get(voice, '#888888'))
        patch.set_alpha(0.7)
    
    ax.axhline(y=150, color='green', linestyle='--', alpha=0.5, label='Target Speech Rate')
    ax.set_ylabel('Speech Rate (WPM)')
    ax.set_title('TTS Speech Rate Distribution by Voice Profile\n(Words Per Minute)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')
    
    output_path = output_dir / "speech_rate_distribution_by_voice.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved: {output_path.name}")

def plot_latency_vs_length(df: pd.DataFrame, output_dir: Path):
    """Plot latency vs text length scatter."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for voice in df['voice'].unique():
        voice_df = df[df['voice'] == voice]
        ax.scatter(voice_df['word_count'], voice_df['generation_time'],
                  label=voice, color=COLORS.get(voice, '#888888'),
                  alpha=0.6, s=30)
    
    ax.set_xlabel('Response Word Count')
    ax.set_ylabel('Generation Time (seconds)')
    ax.set_title('TTS Generation Latency vs Response Length')
    ax.legend(title='Voice Profile')
    ax.grid(alpha=0.3)
    
    output_path = output_dir / "latency_vs_text_length.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved: {output_path.name}")

def plot_speaking_rate_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot metrics by speaking rate."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Average speech rate by rate setting
    rate_speech = df.groupby('speaking_rate')['speech_rate_wpm'].mean()
    axes[0].bar(rate_speech.index, rate_speech.values, color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[0].set_ylabel('Mean Speech Rate (WPM)')
    axes[0].set_title('Speech Rate by Setting')
    axes[0].axhline(y=150, color='green', linestyle='--', alpha=0.5, label='Target')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Average latency by rate
    rate_latency = df.groupby('speaking_rate')['generation_time'].mean()
    axes[1].bar(rate_latency.index, rate_latency.values, color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[1].set_ylabel('Mean Generation Time (s)')
    axes[1].set_title('Latency by Speaking Rate')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Average RTF by rate
    rate_rtf = df.groupby('speaking_rate')['rtf'].mean()
    axes[2].bar(rate_rtf.index, rate_rtf.values, color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[2].set_ylabel('Mean RTF')
    axes[2].set_title('Real-Time Factor by Speaking Rate')
    axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Real-time')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "speaking_rate_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved: {output_path.name}")

def plot_voice_comparison_radar(df: pd.DataFrame, output_dir: Path):
    """Plot radar chart comparing voice profiles."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Metrics to compare (normalized 0-1)
    metrics = ['speech_rate_wpm', 'rtf', 'mean_f0', 'std_f0', 'mean_rms']
    metric_labels = ['Speech Rate\n(WPM)', 'RTF\n(Speed)', 'Pitch\n(F0)',
                    'Pitch Var\n(Expressiveness)', 'Energy\n(RMS)']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for voice in df['voice'].unique():
        voice_df = df[df['voice'] == voice]
        
        # Calculate normalized metrics (higher is better)
        values = [
            voice_df['speech_rate_wpm'].mean() / df['speech_rate_wpm'].max(),
            1 - (voice_df['rtf'].mean() / df['rtf'].max()),  # Invert RTF (lower is better)
            voice_df['mean_f0'].mean() / df['mean_f0'].max(),
            voice_df['std_f0'].mean() / df['std_f0'].max(),
            voice_df['mean_rms'].mean() / df['mean_rms'].max(),
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=voice.split('-')[-1],
                color=COLORS.get(voice, '#888888'))
        ax.fill(angles, values, alpha=0.15, color=COLORS.get(voice, '#888888'))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title('Voice Profile Comparison\n(Normalized Metrics)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    output_path = output_dir / "voice_comparison_radar.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path.name}")

def plot_performance_dashboard(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive performance dashboard."""
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)
    
    # Overall metrics - SAME SIZE AS OTHER GRAPHS
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_summary = {
        'Speech Rate\n(WPM)': df['speech_rate_wpm'].mean(),
        'Latency\n(ms)': df['generation_time'].mean() * 1000,
        'RTF': df['rtf'].mean(),
        'Pitch\n(Hz)': df['mean_f0'].mean(),
    }
    
    # Create properly spaced bars
    x_pos = np.arange(len(metrics_summary))
    bars = ax1.bar(x_pos, metrics_summary.values(), 
                   color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
                   alpha=0.8, width=0.6)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics_summary.keys(), fontsize=8)
    ax1.set_title('Overall TTS Performance Metrics', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Speech rate by voice - CHANGED TO VERTICAL BARS
    ax2 = fig.add_subplot(gs[0, 1])
    voice_speech = df.groupby('voice')['speech_rate_wpm'].mean().sort_values(ascending=False)
    voice_labels = [v.split('-')[-1] for v in voice_speech.index]
    bars2 = ax2.bar(range(len(voice_speech)), voice_speech.values, 
                    color=[COLORS.get(v, '#888') for v in voice_speech.index],
                    alpha=0.8)
    ax2.set_xticks(range(len(voice_speech)))
    ax2.set_xticklabels(voice_labels, fontsize=8, rotation=45, ha='right')
    ax2.set_ylabel('Mean Speech Rate (WPM)', fontsize=9)
    ax2.set_title('Speech Rate by Voice', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # RTF by category - CHANGED TO VERTICAL BARS
    ax3 = fig.add_subplot(gs[0, 2])
    category_rtf = df.groupby('category')['rtf'].mean().sort_values(ascending=False)
    bars3 = ax3.bar(range(len(category_rtf)), category_rtf.values, color='#A23B72', alpha=0.8)
    ax3.set_xticks(range(len(category_rtf)))
    ax3.set_xticklabels(category_rtf.index, fontsize=7, rotation=45, ha='right')
    ax3.set_ylabel('Mean RTF', fontsize=9)
    ax3.set_title('Real-Time Factor by Content Type', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Latency distribution - KEPT AS IS
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(df['generation_time'] * 1000, bins=25, color='#F18F01', alpha=0.7, edgecolor='black')
    ax4.axvline(df['generation_time'].mean() * 1000, color='red', linestyle='--', label=f'Mean: {df["generation_time"].mean()*1000:.0f}ms')
    ax4.set_xlabel('Generation Time (ms)', fontsize=9)
    ax4.set_ylabel('Frequency', fontsize=9)
    ax4.set_title('Latency Distribution', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    
    # Bottom row - additional metrics
    # Real-Time Factor by Voice
    ax5 = fig.add_subplot(gs[1, 0])
    voice_rtf = df.groupby('voice')['rtf'].mean().sort_values(ascending=False)
    voice_labels_rtf = [v.split('-')[-1] for v in voice_rtf.index]
    bars5 = ax5.bar(range(len(voice_rtf)), voice_rtf.values,
                    color=[COLORS.get(v, '#888') for v in voice_rtf.index],
                    alpha=0.8)
    ax5.set_xticks(range(len(voice_rtf)))
    ax5.set_xticklabels(voice_labels_rtf, fontsize=8, rotation=45, ha='right')
    ax5.set_ylabel('Mean RTF', fontsize=9)
    ax5.set_title('RTF by Voice', fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    
    # Speaking Rate by Category
    ax6 = fig.add_subplot(gs[1, 1])
    category_speech = df.groupby('category')['speech_rate_wpm'].mean().sort_values(ascending=False)
    bars6 = ax6.bar(range(len(category_speech)), category_speech.values, color='#2E86AB', alpha=0.8)
    ax6.set_xticks(range(len(category_speech)))
    ax6.set_xticklabels(category_speech.index, fontsize=7, rotation=45, ha='right')
    ax6.set_ylabel('Mean WPM', fontsize=9)
    ax6.set_title('Speaking Rate by Category', fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    
    # Pitch Distribution
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.hist(df['mean_f0'].dropna(), bins=20, color='#C73E1D', alpha=0.7, edgecolor='black')
    ax7.axvline(df['mean_f0'].mean(), color='darkred', linestyle='--', 
                label=f'Mean: {df["mean_f0"].mean():.1f}Hz')
    ax7.set_xlabel('Pitch (Hz)', fontsize=9)
    ax7.set_ylabel('Frequency', fontsize=9)
    ax7.set_title('Pitch Distribution', fontsize=10)
    ax7.legend(fontsize=8)
    ax7.grid(axis='y', alpha=0.3)
    
    # Audio Duration Distribution
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.hist(df['audio_duration'], bins=20, color='#F18F01', alpha=0.7, edgecolor='black')
    ax8.axvline(df['audio_duration'].mean(), color='darkorange', linestyle='--',
                label=f'Mean: {df["audio_duration"].mean():.1f}s')
    ax8.set_xlabel('Audio Duration (s)', fontsize=9)
    ax8.set_ylabel('Frequency', fontsize=9)
    ax8.set_title('Audio Duration Distribution', fontsize=10)
    ax8.legend(fontsize=8)
    ax8.grid(axis='y', alpha=0.3)
    
    plt.suptitle('LUMENAA TTS Evaluation Dashboard', fontsize=14, fontweight='bold', y=0.98)
    
    output_path = output_dir / "overall_performance_dashboard.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved: {output_path.name}")

def generate_latex_tables(df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX tables for paper."""
    
    # Table 1: Overall metrics by voice
    voice_metrics = df.groupby('voice').agg({
        'speech_rate_wpm': 'mean',
        'rtf': 'mean',
        'generation_time': 'mean',
    }).round(3)
    
    latex_table1 = r"""\begin{table}[t]
\centering
\caption{TTS Performance by Voice Profile. Speech rate measured in words per minute (WPM). RTF is Real-Time Factor (generation time / audio duration). Results averaged over 200 responses per voice.}
\label{tab:tts_voice_comparison}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Voice Profile} & \textbf{Speech Rate (WPM)} & \textbf{RTF} & \textbf{Latency (ms)} \\
\midrule
"""
    
    for voice, row in voice_metrics.iterrows():
        latex_table1 += f"{voice} & {row['speech_rate_wpm']:.1f} & {row['rtf']:.3f} & {row['generation_time']*1000:.0f} \\\\\n"
    
    latex_table1 += r"""\midrule
\textit{Mean (All Voices)} & """ + f"{df['speech_rate_wpm'].mean():.1f} & {df['rtf'].mean():.3f} & {df['generation_time'].mean()*1000:.0f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    table1_path = output_dir / "latex_table_voice_comparison.tex"
    with open(table1_path, 'w') as f:
        f.write(latex_table1)
    print(f"✓ Saved: {table1_path.name}")

def main():
    """Main execution."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    results_dir = project_dir / "results"
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating TTS Visualizations...")
    print("="*70)
    
    # Load data
    df = load_data(results_dir)
    if df is None:
        return
    
    print(f"Loaded {len(df):,} metric entries\n")
    
    # Generate plots
    plot_mos_distribution(df, viz_dir)
    plot_latency_vs_length(df, viz_dir)
    plot_speaking_rate_comparison(df, viz_dir)
    plot_voice_comparison_radar(df, viz_dir)
    plot_performance_dashboard(df, viz_dir)
    
    # Generate LaTeX tables
    generate_latex_tables(df, viz_dir)
    
    print("\n" + "="*70)
    print("✓ All visualizations generated successfully!")
    print(f"✓ Output directory: {viz_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
