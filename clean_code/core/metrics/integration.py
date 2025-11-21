"""
Experimental Metrics Integration Helper

This module provides helper functions and decorators to automatically integrate
experimental metrics collection into existing agent code with minimal changes.
"""

import time
import uuid
import functools
from typing import Optional, Callable, Any, Dict
import inspect

def get_experimental_metrics():
    """Get the global experimental metrics collector if available"""
    try:
        import builtins
        return getattr(builtins, 'experimental_metrics', None)
    except:
        return None

def detect_query_type(query_text: str, context: Dict[str, Any] = None) -> str:
    """
    Automatically detect query type based on content and context
    
    Args:
        query_text: The query text to analyze
        context: Additional context information
    
    Returns:
        Detected query type string
    """
    query_lower = query_text.lower()
    
    # Object location queries
    if any(word in query_lower for word in ['where', 'location', 'find', 'locate', 'position']):
        return "Object Location"
    
    # Person queries
    if any(word in query_lower for word in ['who', 'person', 'people', 'face', 'human']):
        return "Person Query"
    
    # Scene description queries
    if any(word in query_lower for word in ['describe', 'what', 'scene', 'happening', 'see']):
        return "Scene Description"
    
    # Object appearance queries
    if any(word in query_lower for word in ['look', 'appearance', 'color', 'shape', 'size']):
        return "Object Appearance"
    
    # Document reading queries
    if any(word in query_lower for word in ['read', 'text', 'document', 'sign', 'writing']):
        return "Document Reading"
    
    # Default
    return "General Query"

def detect_scene_type(objects_count: int, previous_counts: list = None) -> str:
    """
    Detect scene type based on object count and stability
    
    Args:
        objects_count: Current number of objects detected
        previous_counts: List of previous object counts for stability analysis
    
    Returns:
        Scene type: "Stable", "Dynamic", or "Mixed"
    """
    if previous_counts is None or len(previous_counts) < 5:
        return "Mixed"  # Not enough data
    
    # Calculate variance in object counts
    variance = sum((count - objects_count) ** 2 for count in previous_counts[-5:]) / 5
    
    if variance < 1:  # Very stable
        return "Stable"
    elif variance > 5:  # Very dynamic
        return "Dynamic"
    else:
        return "Mixed"

class QueryTracker:
    """Context manager for tracking query execution"""
    
    def __init__(self, query_text: str, query_type: str = None):
        self.query_id = str(uuid.uuid4())
        self.query_text = query_text
        self.query_type = query_type or detect_query_type(query_text)
        self.metrics = get_experimental_metrics()
        self.accuracy_score = None
    
    def __enter__(self):
        if self.metrics:
            self.metrics.start_query(self.query_id, self.query_type, self.query_text)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.metrics:
            self.metrics.end_query(self.query_id, self.accuracy_score)
    
    def mark_fast_path_start(self):
        """Mark the start of fast-path execution"""
        if self.metrics:
            self.metrics.mark_fast_path_start(self.query_id)
    
    def mark_llm_start(self, is_cached: bool = False):
        """Mark the start of LLM execution"""
        if self.metrics:
            self.metrics.mark_llm_start(self.query_id, is_cached)
    
    def record_llm_call(self):
        """Record an LLM API call"""
        if self.metrics:
            self.metrics.record_llm_call(self.query_id)
    
    def set_accuracy_score(self, score: float):
        """Set the accuracy score for this query"""
        self.accuracy_score = score

def track_query(query_type: str = None):
    """
    Decorator to automatically track query execution
    
    Usage:
        @track_query("Object Location")
        def find_objects(query):
            # Your query processing code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract query text from arguments
            query_text = ""
            if args:
                query_text = str(args[0])
            elif 'query' in kwargs:
                query_text = str(kwargs['query'])
            
            detected_type = query_type or detect_query_type(query_text)
            
            with QueryTracker(query_text, detected_type) as tracker:
                result = func(*args, **kwargs)
                return result
        
        return wrapper
    return decorator

def record_vlm_call(with_tracking: bool = False):
    """Record a VLM API call"""
    metrics = get_experimental_metrics()
    if metrics:
        metrics.record_vlm_call(with_tracking)

def record_frame_metrics(scene_type: str, objects_count: int, fps: float, 
                        vlm_calls: int = 0, vlm_with_tracking: int = 0):
    """Record frame processing metrics"""
    metrics = get_experimental_metrics()
    if metrics:
        metrics.record_frame_metrics(scene_type, objects_count, fps, vlm_calls, vlm_with_tracking)

class PerformanceTimer:
    """Simple performance timer for measuring execution times"""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

# Convenience functions for common patterns

def auto_track_api_calls(func: Callable) -> Callable:
    """
    Decorator that automatically tracks API calls in a function
    Looks for common API call patterns and records them
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get function source to detect API calls (simplified)
        try:
            source = inspect.getsource(func)
            
            # Count potential API calls (this is a simple heuristic)
            llm_calls = source.count('mistral') + source.count('llm') + source.count('chat')
            vlm_calls = source.count('pixtral') + source.count('vision') + source.count('vlm')
            
            # Record calls before execution
            metrics = get_experimental_metrics()
            if metrics:
                for _ in range(llm_calls):
                    metrics.record_llm_call("auto_detected")
                for _ in range(vlm_calls):
                    metrics.record_vlm_call()
        except:
            # If we can't get source, just continue
            pass
        
        return func(*args, **kwargs)
    
    return wrapper

def create_query_tracker(query_text: str, query_type: str = None) -> QueryTracker:
    """Factory function to create a query tracker"""
    return QueryTracker(query_text, query_type)
