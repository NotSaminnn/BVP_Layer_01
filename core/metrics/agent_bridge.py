"""
Agent Bridge for Experimental Metrics

This module bridges the agent's existing performance monitoring system
with the experimental metrics collector to capture all query data.
"""

import os
import time
from typing import Optional, Dict, Any
from core.integration import detect_query_type, detect_scene_type


class AgentMetricsBridge:
    """
    Bridge between the agent's performance monitoring and experimental metrics
    """
    
    def __init__(self):
        self.experimental_metrics = None
        self.active_queries: Dict[str, Dict] = {}
        
        # Try to get the experimental metrics collector
        try:
            import builtins
            self.experimental_metrics = getattr(builtins, 'experimental_metrics', None)
        except:
            pass
        
        if self.experimental_metrics:
            print("+ Agent metrics bridge initialized - experimental metrics connected")
        else:
            print("- Agent metrics bridge initialized - experimental metrics not available")
    
    def start_query_tracking(self, request_id: str, query_text: str, query_type: str = None) -> None:
        """Start tracking a query from the agent"""
        if not self.experimental_metrics:
            return
        
        # Auto-detect query type if not provided
        if not query_type or query_type == "Unknown":
            query_type = detect_query_type(query_text)
        
        # Create a unique query ID for experimental metrics
        exp_query_id = f"agent_{request_id}"
        
        # Store mapping
        self.active_queries[request_id] = {
            'exp_query_id': exp_query_id,
            'start_time': time.time(),
            'query_type': query_type,
            'fast_path_started': False,
            'llm_started': False
        }
        
        # Start tracking in experimental metrics
        self.experimental_metrics.start_query(exp_query_id, query_type, query_text)
    
    def mark_planning_start(self, request_id: str) -> None:
        """Mark the start of planning phase (fast-path)"""
        if not self.experimental_metrics or request_id not in self.active_queries:
            return
        
        query_info = self.active_queries[request_id]
        if not query_info['fast_path_started']:
            self.experimental_metrics.mark_fast_path_start(query_info['exp_query_id'])
            query_info['fast_path_started'] = True
    
    def mark_llm_start(self, request_id: str, is_cached: bool = False) -> None:
        """Mark the start of LLM processing"""
        if not self.experimental_metrics or request_id not in self.active_queries:
            return
        
        query_info = self.active_queries[request_id]
        if not query_info['llm_started']:
            self.experimental_metrics.mark_llm_start(query_info['exp_query_id'], is_cached)
            query_info['llm_started'] = True
    
    def record_llm_call(self, request_id: str) -> None:
        """Record an LLM API call"""
        if not self.experimental_metrics or request_id not in self.active_queries:
            return
        
        query_info = self.active_queries[request_id]
        self.experimental_metrics.record_llm_call(query_info['exp_query_id'])
    
    def end_query_tracking(self, request_id: str, success: bool = True, accuracy_score: Optional[float] = None) -> None:
        """End query tracking"""
        if not self.experimental_metrics or request_id not in self.active_queries:
            return
        
        query_info = self.active_queries[request_id]
        
        # End tracking in experimental metrics
        self.experimental_metrics.end_query(query_info['exp_query_id'], accuracy_score)
        
        # Clean up
        del self.active_queries[request_id]
    
    def record_vlm_call(self, with_tracking: bool = False) -> None:
        """Record a VLM API call"""
        if self.experimental_metrics:
            self.experimental_metrics.record_vlm_call(with_tracking)
    
    def record_frame_metrics(self, objects_count: int, fps: float, vlm_calls: int = 0, vlm_with_tracking: int = 0) -> None:
        """Record frame processing metrics"""
        if not self.experimental_metrics:
            return
        
        # Detect scene type based on object count (simple heuristic)
        scene_type = detect_scene_type(objects_count)
        
        self.experimental_metrics.record_frame_metrics(
            scene_type=scene_type,
            objects_count=objects_count,
            fps=fps,
            vlm_calls=vlm_calls,
            vlm_with_tracking=vlm_with_tracking
        )


# Global bridge instance
_agent_bridge: Optional[AgentMetricsBridge] = None

def get_agent_bridge() -> Optional[AgentMetricsBridge]:
    """Get the global agent metrics bridge"""
    global _agent_bridge
    if _agent_bridge is None:
        # Only initialize if experimental metrics is enabled
        if os.environ.get("EXPERIMENTAL_METRICS_ENABLED") == "1":
            _agent_bridge = AgentMetricsBridge()
    return _agent_bridge

def init_agent_bridge() -> Optional[AgentMetricsBridge]:
    """Initialize the agent metrics bridge"""
    global _agent_bridge
    if os.environ.get("EXPERIMENTAL_METRICS_ENABLED") == "1":
        _agent_bridge = AgentMetricsBridge()
        return _agent_bridge
    return None
