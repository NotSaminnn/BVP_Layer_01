"""
Unified Logging System for LUMENAA Agent

Consolidates all performance metrics, system logs, and agent activities into a single
comprehensive log file per run, with automatic summary generation.

Features:
- Single JSON log file per session with timestamps
- Real-time performance tracking
- Comprehensive session summary
- Cross-computer comparison support
- Structured data for analysis
"""

import json
import time
import os
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
import statistics

from core.infrastructure.performance_monitor import PerformanceMetrics, SystemConfiguration


@dataclass
class LogEntry:
    """Single log entry in the unified log"""
    timestamp: float
    log_level: str  # INFO, DEBUG, WARNING, ERROR, PERFORMANCE
    category: str   # detection, query, audio, system, face_recognition, etc.
    event_id: str
    message: str
    data: Optional[Dict[str, Any]] = None
    performance_data: Optional[Dict[str, Any]] = None


@dataclass
class SessionSummary:
    """Summary statistics for a complete session"""
    session_id: str
    start_time: float
    end_time: float
    duration_minutes: float
    
    # Query statistics
    total_queries: int
    successful_queries: int
    failed_queries: int
    average_query_latency_ms: float
    
    # Detection statistics  
    total_detection_frames: int
    average_fps: float
    total_objects_detected: int
    unique_object_types: List[str]
    
    # Face recognition statistics
    total_faces_detected: int
    face_recognition_calls: int
    average_face_recognition_latency_ms: float
    
    # Audio processing statistics
    audio_queries: int
    total_stt_time_ms: float
    total_tts_time_ms: float
    average_stt_latency_ms: float
    average_tts_latency_ms: float
    
    # VLM statistics
    total_vlm_calls: int
    vlm_calls_per_query: float
    average_vlm_latency_ms: float
    
    # System statistics
    peak_cpu_usage_percent: float
    peak_memory_usage_percent: float
    peak_gpu_usage_percent: float
    average_cpu_usage_percent: float
    average_memory_usage_percent: float
    average_gpu_usage_percent: float
    
    # System configuration
    system_config: Dict[str, Any]
    
    # Error statistics
    total_errors: int
    error_categories: Dict[str, int]


class UnifiedLogger:
    """
    Unified logging system that captures all agent activities and performance
    metrics into a single structured log file per session.
    """
    
    def __init__(self, output_dir: str = "unified_logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Session identification
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start = time.time()
        
        # Log file setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            output_dir, 
            f"lumenaa_session_{timestamp}_{self.session_id}.json"
        )
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self._performance_metrics: List[Dict[str, Any]] = []
        self._log_entries: List[LogEntry] = []
        
        # Statistics tracking
        self._query_latencies: List[float] = []
        self._detection_fps_samples: List[float] = []
        self._face_recognition_latencies: List[float] = []
        self._stt_latencies: List[float] = []
        self._tts_latencies: List[float] = []
        self._vlm_latencies: List[float] = []
        self._cpu_usage_samples: List[float] = []
        self._memory_usage_samples: List[float] = []
        self._gpu_usage_samples: List[float] = []
        
        # Object detection tracking
        self._detected_objects: List[str] = []
        self._faces_detected = 0
        self._face_recognition_calls = 0
        
        # Query tracking
        self._total_queries = 0
        self._successful_queries = 0
        self._failed_queries = 0
        self._total_vlm_calls = 0
        self._audio_queries = 0
        
        # Error tracking
        self._errors: List[LogEntry] = []
        
        # System configuration
        self._system_config = self._detect_system_config()
        
        # Initialize log file
        self._initialize_log_file()
        
        print(f"Unified logging started - Session ID: {self.session_id}")
        print(f"Log file: {self.log_file}")
    
    def _detect_system_config(self) -> Dict[str, Any]:
        """Detect system configuration"""
        import platform
        import psutil
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_info = gpus[0].name if gpus else "No GPU"
        except:
            gpu_info = "GPU info unavailable"
        
        return {
            "platform": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_cores": psutil.cpu_count(),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu_model": gpu_info,
            "hostname": platform.node()
        }
    
    def _initialize_log_file(self):
        """Initialize the log file with session header"""
        with self._lock:
            log_data = {
                "session_info": {
                    "session_id": self.session_id,
                    "start_time": self.session_start,
                    "start_time_iso": datetime.fromtimestamp(self.session_start).isoformat(),
                    "system_config": self._system_config
                },
                "log_entries": [],
                "performance_metrics": [],
                "session_summary": None
            }
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
    
    def log(self, level: str, category: str, message: str, data: Optional[Dict[str, Any]] = None, 
            performance_data: Optional[Dict[str, Any]] = None):
        """Log a general message with optional data"""
        entry = LogEntry(
            timestamp=time.time(),
            log_level=level,
            category=category,
            event_id=str(uuid.uuid4())[:8],
            message=message,
            data=data,
            performance_data=performance_data
        )
        
        with self._lock:
            self._log_entries.append(entry)
            
            # Track errors
            if level in ['ERROR', 'WARNING']:
                self._errors.append(entry)
        
        self._update_log_file()
        
        # Console output for important events
        if level in ['ERROR', 'WARNING'] or category in ['query', 'system']:
            timestamp_str = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
            print(f"[{timestamp_str}] [{level}] [{category}] {message}")
    
    def log_query_start(self, query_text: str, query_type: str = "user") -> str:
        """Log the start of a query and return a query ID for tracking"""
        query_id = str(uuid.uuid4())[:8]
        
        with self._lock:
            self._total_queries += 1
            if query_type == "audio":
                self._audio_queries += 1
        
        self.log("INFO", "query", f"Query started: {query_text[:100]}...", {
            "query_id": query_id,
            "query_text": query_text,
            "query_type": query_type,
            "query_length": len(query_text)
        })
        
        return query_id
    
    def log_query_end(self, query_id: str, success: bool, response: str = "", 
                     latency_ms: float = 0, error_msg: str = ""):
        """Log the completion of a query"""
        with self._lock:
            if success:
                self._successful_queries += 1
            else:
                self._failed_queries += 1
            
            if latency_ms > 0:
                self._query_latencies.append(latency_ms)
        
        level = "INFO" if success else "ERROR"
        message = f"Query completed - Success: {success}"
        if error_msg:
            message += f" - Error: {error_msg}"
        
        self.log(level, "query", message, {
            "query_id": query_id,
            "success": success,
            "response_length": len(response) if response else 0,
            "latency_ms": latency_ms,
            "error_message": error_msg
        }, {
            "query_latency_ms": latency_ms
        })
    
    def log_detection_frame(self, fps: float, objects: List[str], latency_ms: float = 0):
        """Log a detection frame with performance data"""
        with self._lock:
            if fps > 0:
                self._detection_fps_samples.append(fps)
            self._detected_objects.extend(objects)
        
        self.log("DEBUG", "detection", f"Frame processed - FPS: {fps:.1f}, Objects: {len(objects)}", {
            "fps": fps,
            "objects_detected": objects,
            "object_count": len(objects),
            "latency_ms": latency_ms
        }, {
            "detection_fps": fps,
            "detection_latency_ms": latency_ms,
            "objects_count": len(objects)
        })
    
    def log_face_recognition(self, faces_detected: int, latency_ms: float = 0):
        """Log face recognition activity"""
        with self._lock:
            self._faces_detected += faces_detected
            self._face_recognition_calls += 1
            if latency_ms > 0:
                self._face_recognition_latencies.append(latency_ms)
        
        self.log("DEBUG", "face_recognition", f"Face recognition - Faces: {faces_detected}", {
            "faces_detected": faces_detected,
            "latency_ms": latency_ms
        }, {
            "face_recognition_latency_ms": latency_ms,
            "faces_count": faces_detected
        })
    
    def log_audio_processing(self, stt_time_ms: float = 0, tts_time_ms: float = 0, 
                           text_length: int = 0):
        """Log audio processing times"""
        with self._lock:
            if stt_time_ms > 0:
                self._stt_latencies.append(stt_time_ms)
            if tts_time_ms > 0:
                self._tts_latencies.append(tts_time_ms)
        
        self.log("DEBUG", "audio", f"Audio processing - STT: {stt_time_ms}ms, TTS: {tts_time_ms}ms", {
            "stt_time_ms": stt_time_ms,
            "tts_time_ms": tts_time_ms,
            "text_length": text_length
        }, {
            "stt_latency_ms": stt_time_ms,
            "tts_latency_ms": tts_time_ms
        })
    
    def log_vlm_call(self, latency_ms: float, model: str = "pixtral"):
        """Log VLM API call"""
        with self._lock:
            self._total_vlm_calls += 1
            if latency_ms > 0:
                self._vlm_latencies.append(latency_ms)
        
        self.log("DEBUG", "vlm", f"VLM call completed - Model: {model}, Latency: {latency_ms}ms", {
            "model": model,
            "latency_ms": latency_ms
        }, {
            "vlm_latency_ms": latency_ms
        })
    
    def log_system_metrics(self, cpu_percent: float, memory_percent: float, 
                          gpu_percent: float = 0):
        """Log system resource usage"""
        with self._lock:
            self._cpu_usage_samples.append(cpu_percent)
            self._memory_usage_samples.append(memory_percent)
            if gpu_percent > 0:
                self._gpu_usage_samples.append(gpu_percent)
        
        self.log("DEBUG", "system", f"System metrics - CPU: {cpu_percent}%, RAM: {memory_percent}%, GPU: {gpu_percent}%", 
               performance_data={
                   "cpu_usage_percent": cpu_percent,
                   "memory_usage_percent": memory_percent,
                   "gpu_usage_percent": gpu_percent
               })
    
    def _update_log_file(self):
        """Update the log file with current data"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # Update log entries (only add new ones)
            current_entry_count = len(log_data.get("log_entries", []))
            new_entries = self._log_entries[current_entry_count:]
            
            for entry in new_entries:
                log_data["log_entries"].append({
                    "timestamp": entry.timestamp,
                    "timestamp_iso": datetime.fromtimestamp(entry.timestamp).isoformat(),
                    "level": entry.log_level,
                    "category": entry.category,
                    "event_id": entry.event_id,
                    "message": entry.message,
                    "data": entry.data,
                    "performance_data": entry.performance_data
                })
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            print(f"Error updating log file: {e}")
    
    def generate_summary(self) -> SessionSummary:
        """Generate comprehensive session summary"""
        end_time = time.time()
        duration_minutes = (end_time - self.session_start) / 60
        
        # Calculate averages safely
        def safe_avg(values: List[float]) -> float:
            return statistics.mean(values) if values else 0.0
        
        def safe_max(values: List[float]) -> float:
            return max(values) if values else 0.0
        
        # Error categories
        error_categories = {}
        for error in self._errors:
            category = error.category
            error_categories[category] = error_categories.get(category, 0) + 1
        
        summary = SessionSummary(
            session_id=self.session_id,
            start_time=self.session_start,
            end_time=end_time,
            duration_minutes=round(duration_minutes, 2),
            
            # Query statistics
            total_queries=self._total_queries,
            successful_queries=self._successful_queries,
            failed_queries=self._failed_queries,
            average_query_latency_ms=round(safe_avg(self._query_latencies), 1),
            
            # Detection statistics
            total_detection_frames=len(self._detection_fps_samples),
            average_fps=round(safe_avg(self._detection_fps_samples), 1),
            total_objects_detected=len(self._detected_objects),
            unique_object_types=list(set(self._detected_objects)),
            
            # Face recognition statistics
            total_faces_detected=self._faces_detected,
            face_recognition_calls=self._face_recognition_calls,
            average_face_recognition_latency_ms=round(safe_avg(self._face_recognition_latencies), 1),
            
            # Audio statistics
            audio_queries=self._audio_queries,
            total_stt_time_ms=round(sum(self._stt_latencies), 1),
            total_tts_time_ms=round(sum(self._tts_latencies), 1),
            average_stt_latency_ms=round(safe_avg(self._stt_latencies), 1),
            average_tts_latency_ms=round(safe_avg(self._tts_latencies), 1),
            
            # VLM statistics
            total_vlm_calls=self._total_vlm_calls,
            vlm_calls_per_query=round(self._total_vlm_calls / max(1, self._total_queries), 2),
            average_vlm_latency_ms=round(safe_avg(self._vlm_latencies), 1),
            
            # System statistics
            peak_cpu_usage_percent=round(safe_max(self._cpu_usage_samples), 1),
            peak_memory_usage_percent=round(safe_max(self._memory_usage_samples), 1),
            peak_gpu_usage_percent=round(safe_max(self._gpu_usage_samples), 1),
            average_cpu_usage_percent=round(safe_avg(self._cpu_usage_samples), 1),
            average_memory_usage_percent=round(safe_avg(self._memory_usage_samples), 1),
            average_gpu_usage_percent=round(safe_avg(self._gpu_usage_samples), 1),
            
            # System configuration
            system_config=self._system_config,
            
            # Error statistics
            total_errors=len(self._errors),
            error_categories=error_categories
        )
        
        return summary
    
    def finalize_session(self):
        """Finalize the session and write summary"""
        summary = self.generate_summary()
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            log_data["session_summary"] = asdict(summary)
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"\n=== SESSION SUMMARY ===")
            print(f"Session ID: {summary.session_id}")
            print(f"Duration: {summary.duration_minutes} minutes")
            print(f"Total Queries: {summary.total_queries}")
            print(f"Success Rate: {summary.successful_queries}/{summary.total_queries}")
            print(f"Average Query Latency: {summary.average_query_latency_ms}ms")
            print(f"Detection Frames: {summary.total_detection_frames}")
            print(f"Average FPS: {summary.average_fps}")
            print(f"Total VLM Calls: {summary.total_vlm_calls}")
            print(f"VLM Calls per Query: {summary.vlm_calls_per_query}")
            print(f"Objects Detected: {summary.total_objects_detected}")
            print(f"Unique Object Types: {len(summary.unique_object_types)}")
            print(f"Peak CPU Usage: {summary.peak_cpu_usage_percent}%")
            print(f"Peak Memory Usage: {summary.peak_memory_usage_percent}%")
            print(f"Total Errors: {summary.total_errors}")
            print(f"Log File: {self.log_file}")
            print("=" * 30)
            
        except Exception as e:
            print(f"Error finalizing session: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        summary = self.generate_summary()
        return asdict(summary)


# Global unified logger instance
_global_logger: Optional[UnifiedLogger] = None

def get_unified_logger() -> UnifiedLogger:
    """Get the global unified logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = UnifiedLogger()
    return _global_logger

def init_unified_logger(output_dir: str = "unified_logs") -> UnifiedLogger:
    """Initialize global unified logger with custom output directory"""
    global _global_logger
    if _global_logger:
        _global_logger.finalize_session()
    _global_logger = UnifiedLogger(output_dir)
    return _global_logger

def finalize_logging():
    """Finalize current logging session"""
    global _global_logger
    if _global_logger:
        _global_logger.finalize_session()
        _global_logger = None