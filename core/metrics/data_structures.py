"""
Data Structures for Experimental Metrics

Defines the core data structures used throughout the experimental metrics system.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class QueryMetrics:
    """Metrics for a single query execution"""
    timestamp: float
    query_type: str  # Object Location, Person Query, Scene Description, etc.
    end_to_end_latency: float
    fast_path_time: Optional[float] = None
    llm_first_pass_time: Optional[float] = None
    llm_cached_time: Optional[float] = None
    planning_mode: str = "Unknown"  # Fast-Path Only, LLM Only, Hybrid
    cache_hit: bool = False
    llm_calls_made: int = 0
    accuracy_score: Optional[float] = None
    session_id: str = ""

@dataclass
class FrameMetrics:
    """Metrics for frame processing and VLM optimization"""
    timestamp: float
    scene_type: str  # Stable, Dynamic, Mixed
    objects_count: int
    fps: float
    vlm_calls_made: int
    vlm_calls_with_tracking: int
    vlm_calls_without_tracking: int
    cost_estimate: float
    session_id: str = ""

@dataclass
class APICallMetrics:
    """Timestamped API call tracking"""
    timestamp: float
    api_type: str  # VLM, LLM, etc.
    call_count: int
    cost_estimate: float
    session_id: str = ""

@dataclass
class SessionSummary:
    """Summary statistics for a complete session"""
    session_id: str
    start_time: float
    end_time: float
    total_queries: int
    total_api_calls: int
    total_cost: float
    avg_latency: float
    cache_hit_rate: float
    most_common_query_type: str
    performance_score: float
