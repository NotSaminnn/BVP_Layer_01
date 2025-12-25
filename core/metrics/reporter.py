#!/usr/bin/env python3
"""
Experimental Metrics Reporter and Visualizer

This module generates comprehensive reports and visualizations from the experimental
metrics data collected by the LUMENAA agent. It can be run independently to analyze
data from completed sessions or used for real-time monitoring.
"""

import os
import sys
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path
import numpy as np

class ExperimentalMetricsReporter:
    """
    Comprehensive reporter for experimental metrics with visualization capabilities
    """
    
    def __init__(self, metrics_dir: str = None):
        if metrics_dir is None:
            # Default to the data directory within this module
            module_dir = os.path.dirname(__file__)
            metrics_dir = os.path.join(module_dir, 'data')
        
        self.metrics_dir = Path(metrics_dir)
        self.output_dir = self.metrics_dir / "reports"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def find_sessions(self) -> List[str]:
        """Find all available session IDs"""
        sessions = set()
        if not self.metrics_dir.exists():
            return []
        
        for file in self.metrics_dir.glob("*_*.csv"):
            # Extract session ID from filename
            parts = file.stem.split('_')
            if len(parts) >= 3:  # e.g., query_metrics_20231201_143022.csv
                session_id = f"{parts[-2]}_{parts[-1]}"
                sessions.add(session_id)
        
        return sorted(list(sessions))
    
    def load_session_data(self, session_id: str) -> Dict[str, pd.DataFrame]:
        """Load all data for a specific session"""
        data = {}
        
        # Load query metrics
        query_file = self.metrics_dir / f"query_metrics_{session_id}.csv"
        if query_file.exists():
            data['queries'] = pd.read_csv(query_file)
            data['queries']['timestamp'] = pd.to_datetime(data['queries']['timestamp'], unit='s')
        
        # Load frame metrics
        frame_file = self.metrics_dir / f"frame_metrics_{session_id}.csv"
        if frame_file.exists():
            data['frames'] = pd.read_csv(frame_file)
            data['frames']['timestamp'] = pd.to_datetime(data['frames']['timestamp'], unit='s')
        
        # Load API metrics
        api_file = self.metrics_dir / f"api_metrics_{session_id}.csv"
        if api_file.exists():
            data['api_calls'] = pd.read_csv(api_file)
            data['api_calls']['timestamp'] = pd.to_datetime(data['api_calls']['timestamp'], unit='s')
        
        # Load summary if available
        summary_file = self.metrics_dir / f"summary_{session_id}.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data['summary'] = json.load(f)
        
        return data
    
    def generate_comprehensive_report(self, session_id: str) -> str:
        """Generate a comprehensive report for a session"""
        data = self.load_session_data(session_id)
        
        if not data:
            return f"No data found for session {session_id}"
        
        report_file = self.output_dir / f"comprehensive_report_{session_id}.html"
        
        # Generate HTML report
        html_content = self._generate_html_report(session_id, data)
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        # Generate visualizations
        self._generate_visualizations(session_id, data)
        
        return str(report_file)
    
    def _generate_html_report(self, session_id: str, data: Dict[str, Any]) -> str:
        """Generate HTML report content"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Experimental Metrics Report - {session_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 30px 0; }}
        .metric-box {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        .warning {{ color: #ff6600; }}
        .success {{ color: #00aa00; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Experimental Metrics Report</h1>
        <p><strong>Session ID:</strong> {session_id}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
        
        # Query Performance Analysis
        if 'queries' in data and not data['queries'].empty:
            html += self._generate_query_analysis_html(data['queries'])
        
        # Frame Processing Analysis
        if 'frames' in data and not data['frames'].empty:
            html += self._generate_frame_analysis_html(data['frames'])
        
        # API Call Analysis
        if 'api_calls' in data and not data['api_calls'].empty:
            html += self._generate_api_analysis_html(data['api_calls'])
        
        # Cost Analysis
        if 'api_calls' in data and not data['api_calls'].empty:
            html += self._generate_cost_analysis_html(data['api_calls'])
        
        html += f"""
    <div class="section">
        <h2>Visualizations</h2>
        <p>Check the following generated charts:</p>
        <ul>
            <li>query_latency_analysis_{session_id}.png</li>
            <li>api_call_frequency_{session_id}.png</li>
            <li>cost_analysis_{session_id}.png</li>
            <li>performance_trends_{session_id}.png</li>
        </ul>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_query_analysis_html(self, queries_df: pd.DataFrame) -> str:
        """Generate query analysis section of HTML report"""
        # Calculate statistics by query type
        stats_by_type = queries_df.groupby('query_type').agg({
            'end_to_end_latency': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'cache_hit': 'mean',
            'llm_calls_made': 'mean'
        }).round(4)
        
        html = f"""
    <div class="section">
        <h2>Query Performance Analysis</h2>
        <div class="metric-box">
            <h3>Overall Statistics</h3>
            <p><strong>Total Queries:</strong> {len(queries_df)}</p>
            <p><strong>Average Latency:</strong> {queries_df['end_to_end_latency'].mean():.3f}s</p>
            <p><strong>Cache Hit Rate:</strong> {queries_df['cache_hit'].mean():.1%}</p>
            <p><strong>Average LLM Calls per Query:</strong> {queries_df['llm_calls_made'].mean():.2f}</p>
        </div>
        
        <h3>Performance by Query Type</h3>
        <table>
            <tr>
                <th>Query Type</th>
                <th>Count</th>
                <th>Avg Latency (s)</th>
                <th>Median Latency (s)</th>
                <th>Cache Hit Rate</th>
                <th>Avg LLM Calls</th>
            </tr>
"""
        
        # Add rows for each query type
        for query_type in stats_by_type.index:
            row_data = stats_by_type.loc[query_type]
            html += f"""
            <tr>
                <td>{query_type}</td>
                <td>{int(row_data[('end_to_end_latency', 'count')])}</td>
                <td>{row_data[('end_to_end_latency', 'mean')]:.3f}</td>
                <td>{row_data[('end_to_end_latency', 'median')]:.3f}</td>
                <td>{row_data[('cache_hit', 'mean')]:.1%}</td>
                <td>{row_data[('llm_calls_made', 'mean')]:.2f}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
"""
        return html
    
    def _generate_frame_analysis_html(self, frames_df: pd.DataFrame) -> str:
        """Generate frame analysis section"""
        avg_fps = frames_df['fps'].mean()
        avg_objects = frames_df['objects_count'].mean()
        total_vlm_calls = frames_df['vlm_calls_made'].sum()
        
        html = f"""
    <div class="section">
        <h2>Frame Processing Analysis</h2>
        <div class="metric-box">
            <h3>Performance Metrics</h3>
            <p><strong>Average FPS:</strong> {avg_fps:.1f}</p>
            <p><strong>Average Objects per Frame:</strong> {avg_objects:.1f}</p>
            <p><strong>Total VLM Calls:</strong> {total_vlm_calls}</p>
        </div>
    </div>
"""
        return html
    
    def _generate_api_analysis_html(self, api_df: pd.DataFrame) -> str:
        """Generate API analysis section"""
        api_counts = api_df.groupby('api_type')['call_count'].sum()
        
        html = """
    <div class="section">
        <h2>API Call Analysis</h2>
        <div class="metric-box">
            <h3>API Calls by Type</h3>
"""
        
        for api_type, count in api_counts.items():
            html += f"            <p><strong>{api_type}:</strong> {count} calls</p>\n"
        
        html += """
        </div>
    </div>
"""
        return html
    
    def _generate_cost_analysis_html(self, api_df: pd.DataFrame) -> str:
        """Generate cost analysis section"""
        total_cost = api_df['cost_estimate'].sum()
        cost_by_type = api_df.groupby('api_type')['cost_estimate'].sum()
        
        html = f"""
    <div class="section">
        <h2>Cost Analysis</h2>
        <div class="metric-box">
            <h3>Estimated Costs (USD)</h3>
            <p><strong>Total Estimated Cost:</strong> ${total_cost:.4f}</p>
"""
        
        for api_type, cost in cost_by_type.items():
            html += f"            <p><strong>{api_type} API:</strong> ${cost:.4f}</p>\n"
        
        html += """
        </div>
    </div>
"""
        return html
    
    def _generate_visualizations(self, session_id: str, data: Dict[str, Any]) -> None:
        """Generate visualization charts"""
        
        # Query Latency Analysis
        if 'queries' in data and not data['queries'].empty:
            self._plot_query_latency_analysis(session_id, data['queries'])
        
        # API Call Frequency
        if 'api_calls' in data and not data['api_calls'].empty:
            self._plot_api_frequency(session_id, data['api_calls'])
        
        # Cost Analysis
        if 'api_calls' in data and not data['api_calls'].empty:
            self._plot_cost_analysis(session_id, data['api_calls'])
        
        # Performance Trends
        if 'queries' in data and not data['queries'].empty:
            self._plot_performance_trends(session_id, data['queries'])
    
    def _plot_query_latency_analysis(self, session_id: str, queries_df: pd.DataFrame) -> None:
        """Plot query latency analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Query Latency Analysis - {session_id}', fontsize=16)
        
        # Latency by query type (box plot)
        queries_df.boxplot(column='end_to_end_latency', by='query_type', ax=ax1)
        ax1.set_title('Latency Distribution by Query Type')
        ax1.set_ylabel('Latency (seconds)')
        
        # Latency over time
        ax2.scatter(queries_df['timestamp'], queries_df['end_to_end_latency'], alpha=0.6)
        ax2.set_title('Latency Over Time')
        ax2.set_ylabel('Latency (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Cache hit rate by query type
        cache_rates = queries_df.groupby('query_type')['cache_hit'].mean()
        cache_rates.plot(kind='bar', ax=ax3)
        ax3.set_title('Cache Hit Rate by Query Type')
        ax3.set_ylabel('Cache Hit Rate')
        ax3.tick_params(axis='x', rotation=45)
        
        # LLM calls distribution
        queries_df['llm_calls_made'].hist(bins=20, ax=ax4)
        ax4.set_title('Distribution of LLM Calls per Query')
        ax4.set_xlabel('LLM Calls')
        ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'query_latency_analysis_{session_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_api_frequency(self, session_id: str, api_df: pd.DataFrame) -> None:
        """Plot API call frequency over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'API Call Frequency Analysis - {session_id}', fontsize=16)
        
        # API calls over time
        api_df.set_index('timestamp').resample('1min')['call_count'].sum().plot(ax=ax1)
        ax1.set_title('API Calls per Minute')
        ax1.set_ylabel('Calls per Minute')
        
        # API calls by type
        api_counts = api_df.groupby('api_type')['call_count'].sum()
        api_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('API Calls Distribution by Type')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'api_call_frequency_{session_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cost_analysis(self, session_id: str, api_df: pd.DataFrame) -> None:
        """Plot cost analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Cost Analysis - {session_id}', fontsize=16)
        
        # Cumulative cost over time
        api_df_sorted = api_df.sort_values('timestamp')
        api_df_sorted['cumulative_cost'] = api_df_sorted['cost_estimate'].cumsum()
        ax1.plot(api_df_sorted['timestamp'], api_df_sorted['cumulative_cost'])
        ax1.set_title('Cumulative Cost Over Time')
        ax1.set_ylabel('Cost (USD)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Cost by API type
        cost_by_type = api_df.groupby('api_type')['cost_estimate'].sum()
        cost_by_type.plot(kind='bar', ax=ax2)
        ax2.set_title('Total Cost by API Type')
        ax2.set_ylabel('Cost (USD)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'cost_analysis_{session_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_trends(self, session_id: str, queries_df: pd.DataFrame) -> None:
        """Plot performance trends"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Performance Trends - {session_id}', fontsize=16)
        
        # Rolling average latency
        queries_df_sorted = queries_df.sort_values('timestamp')
        queries_df_sorted['rolling_latency'] = queries_df_sorted['end_to_end_latency'].rolling(window=10).mean()
        ax1.plot(queries_df_sorted['timestamp'], queries_df_sorted['rolling_latency'])
        ax1.set_title('Rolling Average Latency (10-query window)')
        ax1.set_ylabel('Latency (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Planning mode distribution over time
        planning_counts = queries_df.groupby(['timestamp', 'planning_mode']).size().unstack(fill_value=0)
        if not planning_counts.empty:
            planning_counts.plot(kind='area', stacked=True, ax=ax2)
            ax2.set_title('Planning Mode Usage Over Time')
            ax2.set_ylabel('Query Count')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'performance_trends_{session_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_report(self, session_ids: List[str]) -> str:
        """Generate a comparison report across multiple sessions"""
        comparison_data = {}
        
        for session_id in session_ids:
            data = self.load_session_data(session_id)
            if 'queries' in data:
                comparison_data[session_id] = {
                    'total_queries': len(data['queries']),
                    'avg_latency': data['queries']['end_to_end_latency'].mean(),
                    'cache_hit_rate': data['queries']['cache_hit'].mean(),
                    'total_cost': data.get('api_calls', pd.DataFrame())['cost_estimate'].sum() if 'api_calls' in data else 0
                }
        
        # Create comparison visualization
        if comparison_data:
            self._plot_session_comparison(comparison_data)
        
        return f"Comparison report generated for {len(session_ids)} sessions"
    
    def _plot_session_comparison(self, comparison_data: Dict[str, Dict]) -> None:
        """Plot comparison across sessions"""
        df = pd.DataFrame(comparison_data).T
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Session Comparison', fontsize=16)
        
        # Total queries
        df['total_queries'].plot(kind='bar', ax=ax1)
        ax1.set_title('Total Queries per Session')
        ax1.set_ylabel('Query Count')
        
        # Average latency
        df['avg_latency'].plot(kind='bar', ax=ax2)
        ax2.set_title('Average Latency per Session')
        ax2.set_ylabel('Latency (seconds)')
        
        # Cache hit rate
        df['cache_hit_rate'].plot(kind='bar', ax=ax3)
        ax3.set_title('Cache Hit Rate per Session')
        ax3.set_ylabel('Hit Rate')
        
        # Total cost
        df['total_cost'].plot(kind='bar', ax=ax4)
        ax4.set_title('Total Cost per Session')
        ax4.set_ylabel('Cost (USD)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'session_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
