"""
Experimental Metrics System

Main system class that coordinates all experimental metrics functionality.
Provides a unified interface for metrics collection, reporting, and analysis.
"""

import os
from typing import Optional, Dict, Any
from core.collector import ExperimentalMetricsCollector
from core.reporter import ExperimentalMetricsReporter
from core.integration import QueryTracker, detect_query_type, detect_scene_type


class ExperimentalMetricsSystem:
    """
    Main experimental metrics system that coordinates collection and reporting
    """
    
    def __init__(self, output_dir: str = None, enable_collection: bool = True):
        """
        Initialize the experimental metrics system
        
        Args:
            output_dir: Directory for metrics output (defaults to module data directory)
            enable_collection: Whether to enable metrics collection
        """
        # Set default output directory within the module
        if output_dir is None:
            module_dir = os.path.dirname(__file__)
            output_dir = os.path.join(module_dir, 'data')
        
        self.output_dir = output_dir
        self.enable_collection = enable_collection
        
        # Initialize components
        self.collector: Optional[ExperimentalMetricsCollector] = None
        self.reporter: Optional[ExperimentalMetricsReporter] = None
        
        if enable_collection:
            self.collector = ExperimentalMetricsCollector(output_dir)
        
        self.reporter = ExperimentalMetricsReporter(output_dir)
        
        print(f"+ Experimental Metrics System initialized")
        print(f"+ Output directory: {output_dir}")
        print(f"+ Collection enabled: {enable_collection}")
    
    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID"""
        return self.collector.session_id if self.collector else None
    
    def track_query(self, query_text: str, query_type: str = None) -> QueryTracker:
        """
        Create a query tracker for measuring query performance
        
        Args:
            query_text: The query text
            query_type: Optional query type (auto-detected if not provided)
        
        Returns:
            QueryTracker instance for use as context manager
        """
        if not query_type:
            query_type = detect_query_type(query_text)
        
        return QueryTracker(query_text, query_type)
    
    def record_vlm_call(self, with_tracking: bool = False) -> None:
        """Record a VLM API call"""
        if self.collector:
            self.collector.record_vlm_call(with_tracking)
    
    def record_frame_metrics(self, scene_type: str, objects_count: int, fps: float,
                           vlm_calls: int = 0, vlm_with_tracking: int = 0) -> None:
        """Record frame processing metrics"""
        if self.collector:
            self.collector.record_frame_metrics(scene_type, objects_count, fps, vlm_calls, vlm_with_tracking)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of current metrics"""
        if self.collector:
            return self.collector.generate_summary()
        return {}
    
    def generate_report(self, session_id: str = None) -> str:
        """
        Generate a comprehensive report
        
        Args:
            session_id: Session to report on (defaults to current session)
        
        Returns:
            Path to generated report file
        """
        if not session_id:
            session_id = self.session_id
        
        if not session_id:
            raise ValueError("No session ID available for reporting")
        
        return self.reporter.generate_comprehensive_report(session_id)
    
    def find_sessions(self) -> list:
        """Find all available sessions"""
        return self.reporter.find_sessions()
    
    def compare_sessions(self, session_ids: list) -> str:
        """Generate a comparison report for multiple sessions"""
        return self.reporter.generate_comparison_report(session_ids)
    
    def get_data_directory(self) -> str:
        """Get the data directory path"""
        return self.output_dir
    
    def get_reports_directory(self) -> str:
        """Get the reports directory path"""
        return os.path.join(self.output_dir, 'reports')
    
    def stop(self) -> Dict[str, Any]:
        """Stop the metrics system and generate final summary"""
        if self.collector:
            return self.collector.stop()
        return {}
    
    def is_collecting(self) -> bool:
        """Check if metrics collection is active"""
        return self.collector is not None and self.collector.running
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        if not self.collector:
            return {}
        
        return {
            'session_id': self.collector.session_id,
            'total_queries': len(self.collector.query_metrics),
            'total_api_calls': len(self.collector.api_call_metrics),
            'total_frames': len(self.collector.frame_metrics),
            'cache_hits': self.collector.cache_hits,
            'cache_misses': self.collector.cache_misses,
            'cache_hit_rate': self.collector.cache_hits / (self.collector.cache_hits + self.collector.cache_misses) if (self.collector.cache_hits + self.collector.cache_misses) > 0 else 0,
            'estimated_cost': sum(m.cost_estimate for m in self.collector.api_call_metrics),
            'output_directory': self.output_dir
        }
    
    def print_stats(self) -> None:
        """Print current statistics to console"""
        stats = self.get_stats()
        if not stats:
            print("No metrics collection active")
            return
        
        print("\n" + "="*50)
        print("EXPERIMENTAL METRICS STATISTICS")
        print("="*50)
        print(f"Session ID: {stats['session_id']}")
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Total API Calls: {stats['total_api_calls']}")
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        print(f"Estimated Cost: ${stats['estimated_cost']:.4f}")
        print(f"Output Directory: {stats['output_directory']}")
        print("="*50)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop collection"""
        self.stop()


# Convenience functions for module-level usage
def create_system(output_dir: str = None, enable_collection: bool = True) -> ExperimentalMetricsSystem:
    """Create and return an ExperimentalMetricsSystem instance"""
    return ExperimentalMetricsSystem(output_dir, enable_collection)

def get_default_data_dir() -> str:
    """Get the default data directory for the module"""
    module_dir = os.path.dirname(__file__)
    return os.path.join(module_dir, 'data')
