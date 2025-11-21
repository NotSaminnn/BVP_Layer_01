"""
Experimental Metrics Collector

Core metrics collection functionality for the experimental metrics system.
Handles real-time data capture, storage, and background processing.
"""

import os
import time
import json
import csv
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from collections import defaultdict, deque
import statistics

from core.data_structures import QueryMetrics, FrameMetrics, APICallMetrics


class ExperimentalMetricsCollector:
    """
    Comprehensive metrics collector for experimental analysis
    Runs continuously in background, automatically detecting and logging metrics
    """
    
    def __init__(self, output_dir: str = None):
        # Set output directory within the module
        if output_dir is None:
            module_dir = os.path.dirname(__file__)
            output_dir = os.path.join(module_dir, 'data')
        
        self.output_dir = output_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Data storage
        self.query_metrics: List[QueryMetrics] = []
        self.frame_metrics: List[FrameMetrics] = []
        self.api_call_metrics: List[APICallMetrics] = []
        
        # Real-time tracking
        self.current_queries: Dict[str, Dict] = {}  # Track ongoing queries
        self.api_call_buffer = deque(maxlen=1000)  # Recent API calls for rate calculation
        self.frame_buffer = deque(maxlen=100)  # Recent frames for FPS calculation
        
        # Cache tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Cost tracking (approximate USD)
        self.cost_per_vlm_call = 0.01  # Approximate cost per VLM API call
        self.cost_per_llm_call = 0.002  # Approximate cost per LLM API call
        
        # Background thread for periodic tasks
        self.running = True
        self.background_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.background_thread.start()
        
        # File paths
        self.query_csv = os.path.join(output_dir, f"query_metrics_{self.session_id}.csv")
        self.frame_csv = os.path.join(output_dir, f"frame_metrics_{self.session_id}.csv")
        self.api_csv = os.path.join(output_dir, f"api_metrics_{self.session_id}.csv")
        self.summary_json = os.path.join(output_dir, f"summary_{self.session_id}.json")
        
        print(f"+ Experimental metrics collector initialized")
        print(f"+ Session ID: {self.session_id}")
        print(f"+ Output directory: {output_dir}")
    
    def start_query(self, query_id: str, query_type: str, query_text: str = "") -> None:
        """Start tracking a new query"""
        self.current_queries[query_id] = {
            'start_time': time.time(),
            'query_type': query_type,
            'query_text': query_text,
            'fast_path_start': None,
            'llm_start': None,
            'llm_calls': 0,
            'cache_checked': False
        }
    
    def mark_fast_path_start(self, query_id: str) -> None:
        """Mark the start of fast-path execution"""
        if query_id in self.current_queries:
            self.current_queries[query_id]['fast_path_start'] = time.time()
    
    def mark_llm_start(self, query_id: str, is_cached: bool = False) -> None:
        """Mark the start of LLM execution"""
        if query_id in self.current_queries:
            self.current_queries[query_id]['llm_start'] = time.time()
            self.current_queries[query_id]['is_cached'] = is_cached
            if is_cached:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def record_llm_call(self, query_id: str) -> None:
        """Record an LLM API call for this query"""
        if query_id in self.current_queries:
            self.current_queries[query_id]['llm_calls'] += 1
        
        # Record API call
        api_metric = APICallMetrics(
            timestamp=time.time(),
            api_type="LLM",
            call_count=1,
            cost_estimate=self.cost_per_llm_call,
            session_id=self.session_id
        )
        self.api_call_metrics.append(api_metric)
        self.api_call_buffer.append(time.time())
        self._write_api_to_csv(api_metric)
    
    def record_vlm_call(self, with_tracking: bool = False) -> None:
        """Record a VLM API call"""
        api_metric = APICallMetrics(
            timestamp=time.time(),
            api_type="VLM",
            call_count=1,
            cost_estimate=self.cost_per_vlm_call,
            session_id=self.session_id
        )
        self.api_call_metrics.append(api_metric)
        self.api_call_buffer.append(time.time())
        self._write_api_to_csv(api_metric)
    
    def end_query(self, query_id: str, accuracy_score: Optional[float] = None) -> None:
        """End query tracking and record metrics"""
        if query_id not in self.current_queries:
            return
        
        query_data = self.current_queries[query_id]
        end_time = time.time()
        
        # Calculate timings
        end_to_end_latency = end_time - query_data['start_time']
        fast_path_time = None
        llm_time = None
        
        if query_data['fast_path_start']:
            if query_data['llm_start']:
                fast_path_time = query_data['llm_start'] - query_data['fast_path_start']
                llm_time = end_time - query_data['llm_start']
            else:
                fast_path_time = end_time - query_data['fast_path_start']
        elif query_data['llm_start']:
            llm_time = end_time - query_data['llm_start']
        
        # Determine planning mode
        planning_mode = "Unknown"
        if fast_path_time and llm_time:
            planning_mode = "Hybrid"
        elif fast_path_time:
            planning_mode = "Fast-Path Only"
        elif llm_time:
            planning_mode = "LLM Only"
        
        # Create metrics record
        metrics = QueryMetrics(
            timestamp=query_data['start_time'],
            query_type=query_data['query_type'],
            end_to_end_latency=end_to_end_latency,
            fast_path_time=fast_path_time,
            llm_first_pass_time=llm_time if not query_data.get('is_cached', False) else None,
            llm_cached_time=llm_time if query_data.get('is_cached', False) else None,
            planning_mode=planning_mode,
            cache_hit=query_data.get('is_cached', False),
            llm_calls_made=query_data['llm_calls'],
            accuracy_score=accuracy_score,
            session_id=self.session_id
        )
        
        self.query_metrics.append(metrics)
        del self.current_queries[query_id]
        
        # Write to CSV immediately for real-time analysis
        self._write_query_to_csv(metrics)
    
    def record_frame_metrics(self, scene_type: str, objects_count: int, fps: float, 
                           vlm_calls: int = 0, vlm_with_tracking: int = 0) -> None:
        """Record frame processing metrics"""
        vlm_without_tracking = vlm_calls - vlm_with_tracking
        cost_estimate = vlm_calls * self.cost_per_vlm_call
        
        metrics = FrameMetrics(
            timestamp=time.time(),
            scene_type=scene_type,
            objects_count=objects_count,
            fps=fps,
            vlm_calls_made=vlm_calls,
            vlm_calls_with_tracking=vlm_with_tracking,
            vlm_calls_without_tracking=vlm_without_tracking,
            cost_estimate=cost_estimate,
            session_id=self.session_id
        )
        
        self.frame_metrics.append(metrics)
        self.frame_buffer.append(time.time())
        
        # Write to CSV immediately
        self._write_frame_to_csv(metrics)
    
    def _write_query_to_csv(self, metrics: QueryMetrics) -> None:
        """Write query metrics to CSV file"""
        file_exists = os.path.exists(self.query_csv)
        with open(self.query_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(metrics).keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(metrics))
    
    def _write_frame_to_csv(self, metrics: FrameMetrics) -> None:
        """Write frame metrics to CSV file"""
        file_exists = os.path.exists(self.frame_csv)
        with open(self.frame_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(metrics).keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(metrics))
    
    def _write_api_to_csv(self, metrics: APICallMetrics) -> None:
        """Write API metrics to CSV file"""
        file_exists = os.path.exists(self.api_csv)
        with open(self.api_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(metrics).keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(metrics))
    
    def _background_worker(self) -> None:
        """Background thread for periodic tasks"""
        while self.running:
            try:
                # Generate summary every 5 minutes
                self.generate_summary()
                time.sleep(300)  # 5 minutes
            except Exception as e:
                print(f"Background worker error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of all metrics"""
        now = time.time()
        
        # Query metrics summary
        query_summary = self._analyze_query_metrics()
        
        # Frame metrics summary
        frame_summary = self._analyze_frame_metrics()
        
        # API call summary
        api_summary = self._analyze_api_metrics()
        
        # Cost analysis
        cost_summary = self._analyze_costs()
        
        summary = {
            'session_id': self.session_id,
            'generated_at': datetime.now().isoformat(),
            'total_runtime_hours': (now - time.mktime(datetime.strptime(self.session_id, "%Y%m%d_%H%M%S").timetuple())) / 3600,
            'query_metrics': query_summary,
            'frame_metrics': frame_summary,
            'api_metrics': api_summary,
            'cost_analysis': cost_summary
        }
        
        # Write summary to JSON
        with open(self.summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _analyze_query_metrics(self) -> Dict[str, Any]:
        """Analyze query metrics for summary"""
        if not self.query_metrics:
            return {}
        
        # Group by query type
        by_type = defaultdict(list)
        for metric in self.query_metrics:
            by_type[metric.query_type].append(metric)
        
        analysis = {}
        for query_type, metrics in by_type.items():
            latencies = [m.end_to_end_latency for m in metrics]
            cache_hits = sum(1 for m in metrics if m.cache_hit)
            llm_calls = sum(m.llm_calls_made for m in metrics)
            
            analysis[query_type] = {
                'count': len(metrics),
                'avg_latency': statistics.mean(latencies),
                'median_latency': statistics.median(latencies),
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'cache_hit_rate': cache_hits / len(metrics) if metrics else 0,
                'avg_llm_calls': llm_calls / len(metrics) if metrics else 0,
                'total_llm_calls': llm_calls
            }
        
        # Overall statistics
        all_latencies = [m.end_to_end_latency for m in self.query_metrics]
        total_cache_hits = sum(1 for m in self.query_metrics if m.cache_hit)
        
        analysis['overall'] = {
            'total_queries': len(self.query_metrics),
            'avg_latency': statistics.mean(all_latencies) if all_latencies else 0,
            'cache_hit_rate': total_cache_hits / len(self.query_metrics) if self.query_metrics else 0,
            'llm_calls_per_100_queries': sum(m.llm_calls_made for m in self.query_metrics) / len(self.query_metrics) * 100 if self.query_metrics else 0
        }
        
        return analysis
    
    def _analyze_frame_metrics(self) -> Dict[str, Any]:
        """Analyze frame metrics for summary"""
        if not self.frame_metrics:
            return {}
        
        # Recent metrics (last hour)
        recent_cutoff = time.time() - 3600
        recent_metrics = [m for m in self.frame_metrics if m.timestamp > recent_cutoff]
        
        if not recent_metrics:
            return {}
        
        avg_fps = statistics.mean([m.fps for m in recent_metrics])
        avg_objects = statistics.mean([m.objects_count for m in recent_metrics])
        total_vlm_calls = sum(m.vlm_calls_made for m in recent_metrics)
        total_vlm_with_tracking = sum(m.vlm_calls_with_tracking for m in recent_metrics)
        
        reduction_percent = 0
        if total_vlm_calls > 0:
            reduction_percent = (total_vlm_with_tracking / total_vlm_calls) * 100
        
        return {
            'recent_hour': {
                'avg_fps': avg_fps,
                'avg_objects_per_frame': avg_objects,
                'total_vlm_calls': total_vlm_calls,
                'vlm_calls_with_tracking': total_vlm_with_tracking,
                'tracking_optimization_percent': reduction_percent
            }
        }
    
    def _analyze_api_metrics(self) -> Dict[str, Any]:
        """Analyze API call metrics"""
        if not self.api_call_metrics:
            return {}
        
        # Recent API calls (last hour)
        recent_cutoff = time.time() - 3600
        recent_calls = [m for m in self.api_call_metrics if m.timestamp > recent_cutoff]
        
        # Group by API type
        by_type = defaultdict(int)
        for call in recent_calls:
            by_type[call.api_type] += call.call_count
        
        # Calculate calls per second
        calls_per_second = {}
        if recent_calls:
            time_span = max(m.timestamp for m in recent_calls) - min(m.timestamp for m in recent_calls)
            if time_span > 0:
                for api_type, count in by_type.items():
                    calls_per_second[api_type] = count / time_span
        
        return {
            'recent_hour_calls': dict(by_type),
            'calls_per_second': calls_per_second,
            'total_api_calls': len(self.api_call_metrics)
        }
    
    def _analyze_costs(self) -> Dict[str, Any]:
        """Analyze cost metrics"""
        total_cost = sum(m.cost_estimate for m in self.api_call_metrics)
        
        # Cost by API type
        cost_by_type = defaultdict(float)
        for metric in self.api_call_metrics:
            cost_by_type[metric.api_type] += metric.cost_estimate
        
        return {
            'total_estimated_cost_usd': total_cost,
            'cost_by_api_type': dict(cost_by_type),
            'avg_cost_per_query': total_cost / len(self.query_metrics) if self.query_metrics else 0
        }
    
    def stop(self) -> None:
        """Stop the metrics collector and generate final summary"""
        self.running = False
        if self.background_thread.is_alive():
            self.background_thread.join(timeout=5)
        
        # Generate final summary
        final_summary = self.generate_summary()
        print(f"\n+ Experimental metrics collection stopped")
        print(f"+ Final summary saved to: {self.summary_json}")
        print(f"+ Total queries processed: {len(self.query_metrics)}")
        print(f"+ Total estimated cost: ${final_summary.get('cost_analysis', {}).get('total_estimated_cost_usd', 0):.4f}")
        
        return final_summary
