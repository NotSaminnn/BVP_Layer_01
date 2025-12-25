"""
Comprehensive Performance Monitor for LUMENAA Agent System

Tracks all metrics mentioned in CVPR_RESULTS_SECTION_PLAN.md:
- Detection FPS and latency
- Face recognition latency  
- End-to-end query processing latency
- VLM API call frequency and reduction
- Temporal memory performance
- Audio processing times (STT/TTS)
- System resource usage

Logs metrics to CSV files for analysis across different computer configurations.
Integrates with unified logger when available.
"""

import asyncio
import csv
import time
import psutil
import platform
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

# Optional GPU monitoring - gracefully handle if GPUtil is not available
try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    GPUtil = None

# Import unified logging integration if available
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from logger_integration import get_logger_integration
    UNIFIED_LOGGING_AVAILABLE = True
except ImportError:
    UNIFIED_LOGGING_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Single performance measurement record"""
    timestamp: float
    request_id: str
    query_text: str = ""
    query_type: str = ""
    
    # Detection metrics
    detection_fps: float = 0.0
    detection_latency_ms: float = 0.0
    objects_detected: int = 0
    
    # Face recognition metrics
    faces_detected: int = 0
    face_recognition_latency_ms: float = 0.0
    faces_recognized: int = 0
    
    # End-to-end query processing
    total_query_latency_ms: float = 0.0
    planning_latency_ms: float = 0.0
    execution_latency_ms: float = 0.0
    
    # Audio processing
    audio_duration_ms: float = 0.0
    stt_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    
    # VLM/Pixtral processing
    vlm_calls_made: int = 0
    vlm_latency_ms: float = 0.0
    scene_analysis_latency_ms: float = 0.0
    object_description_latency_ms: float = 0.0
    
    # Planning metrics
    plan_cache_hit: bool = False
    fast_path_used: bool = False
    
    # System metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    
    # Temporal memory
    temporal_objects_count: int = 0
    temporal_memory_mb: float = 0.0
    
    # Cost tracking
    estimated_cost_cents: float = 0.0


@dataclass
class SystemConfiguration:
    """System hardware/software configuration"""
    platform: str
    cpu_model: str
    cpu_cores: int
    total_memory_gb: float
    gpu_model: str = "None"
    gpu_memory_gb: float = 0.0
    python_version: str = ""
    pytorch_version: str = ""
    cuda_version: str = ""


class PerformanceMonitor:
    """
    Central performance monitoring system that tracks all metrics
    and logs them to CSV files for analysis.
    """
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Current metrics being collected
        self._current_metrics: Dict[str, PerformanceMetrics] = {}
        self._metrics_lock = threading.Lock()
        
        # CSV file paths
        self.metrics_csv = os.path.join(output_dir, f"performance_metrics_{int(time.time())}.csv")
        self.system_csv = os.path.join(output_dir, f"system_config_{int(time.time())}.csv")
        
        # System configuration
        self.system_config = self._detect_system_configuration()
        self._write_system_config()
        
        # Initialize CSV headers
        self._initialize_csv()
        
        # Background monitoring
        self._monitoring_active = True
        self._monitoring_task = None
        
        # Metrics counters
        self.total_vlm_calls = 0
        self.total_queries = 0
        self.detection_frame_count = 0
        self.detection_fps_samples = []
        
    def start_monitoring(self):
        """Start background system monitoring"""
        try:
            if self._monitoring_task is None or self._monitoring_task.done():
                self._monitoring_task = asyncio.create_task(self._background_monitoring())
        except RuntimeError:
            # No event loop running - monitoring will be disabled
            print(f"[INFO] Background system monitoring disabled (no event loop)")
            self._monitoring_active = False
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
    
    async def _background_monitoring(self):
        """Background task that monitors system resources every second"""
        while self._monitoring_active:
            try:
                # Get current system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_mb = psutil.virtual_memory().used / 1024 / 1024
                memory_percent = psutil.virtual_memory().percent
                gpu_percent = 0
                gpu_memory_mb = 0
                
                # GPU metrics (optional)
                if GPU_MONITORING_AVAILABLE and GPUtil:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Primary GPU
                            gpu_percent = gpu.load * 100
                            gpu_memory_mb = gpu.memoryUsed
                    except:
                        pass  # GPU monitoring optional
                
                # Update system metrics for all active requests
                with self._metrics_lock:
                    for metrics in self._current_metrics.values():
                        metrics.cpu_usage_percent = cpu_percent
                        metrics.memory_usage_mb = memory_mb
                        metrics.gpu_usage_percent = gpu_percent
                        metrics.gpu_memory_mb = gpu_memory_mb
                
                # Also feed to unified logger if available
                if UNIFIED_LOGGING_AVAILABLE:
                    try:
                        logger_integration = get_logger_integration()
                        logger_integration.log_system_metrics(cpu_percent, memory_percent, gpu_percent)
                    except:
                        pass  # Graceful fallback
                
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[WARNING] Background monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    def _detect_system_configuration(self) -> SystemConfiguration:
        """Detect current system hardware and software configuration"""
        config = SystemConfiguration(
            platform=platform.platform(),
            cpu_model=platform.processor() or "Unknown",
            cpu_cores=psutil.cpu_count(),
            total_memory_gb=psutil.virtual_memory().total / 1024 / 1024 / 1024,
            python_version=platform.python_version()
        )
        
        # GPU detection (optional)
        if GPU_MONITORING_AVAILABLE and GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    config.gpu_model = gpu.name
                    config.gpu_memory_gb = gpu.memoryTotal / 1024
            except:
                pass
        
        # PyTorch version
        try:
            import torch
            config.pytorch_version = torch.__version__
            if torch.cuda.is_available():
                config.cuda_version = torch.version.cuda or "Unknown"
        except:
            pass
        
        return config
    
    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            'timestamp', 'request_id', 'query_text', 'query_type',
            'detection_fps', 'detection_latency_ms', 'objects_detected',
            'faces_detected', 'face_recognition_latency_ms', 'faces_recognized',
            'total_query_latency_ms', 'planning_latency_ms', 'execution_latency_ms',
            'audio_duration_ms', 'stt_latency_ms', 'tts_latency_ms',
            'vlm_calls_made', 'vlm_latency_ms', 'scene_analysis_latency_ms', 'object_description_latency_ms',
            'plan_cache_hit', 'fast_path_used',
            'cpu_usage_percent', 'memory_usage_mb', 'gpu_usage_percent', 'gpu_memory_mb',
            'temporal_objects_count', 'temporal_memory_mb', 'estimated_cost_cents'
        ]
        
        with open(self.metrics_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def _write_system_config(self):
        """Write system configuration to CSV"""
        headers = ['platform', 'cpu_model', 'cpu_cores', 'total_memory_gb', 
                  'gpu_model', 'gpu_memory_gb', 'python_version', 'pytorch_version', 'cuda_version']
        
        with open(self.system_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow([
                self.system_config.platform, self.system_config.cpu_model, self.system_config.cpu_cores,
                self.system_config.total_memory_gb, self.system_config.gpu_model, self.system_config.gpu_memory_gb,
                self.system_config.python_version, self.system_config.pytorch_version, self.system_config.cuda_version
            ])
    
    def start_query_measurement(self, request_id: str, query_text: str = "", query_type: str = "") -> str:
        """
        Start measuring performance for a new query.
        Returns the request_id for tracking.
        """
        with self._metrics_lock:
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                request_id=request_id,
                query_text=query_text,
                query_type=query_type
            )
            self._current_metrics[request_id] = metrics
        
        self.total_queries += 1
        return request_id
    
    def record_detection_metrics(self, request_id: str, fps: float, latency_ms: float, objects_count: int):
        """Record object detection performance metrics"""
        
        # For continuous detection, write immediately to CSV instead of storing in _current_metrics
        if request_id.startswith("detection_"):
            self.detection_frame_count += 1
            self.detection_fps_samples.append(fps)
            if len(self.detection_fps_samples) > 100:  # Keep last 100 samples
                self.detection_fps_samples.pop(0)
            
            # Write detection frame to CSV immediately
            self._write_detection_frame_to_csv(fps, latency_ms, objects_count)
        else:
            # For query-associated detection, store in current metrics
            with self._metrics_lock:
                if request_id in self._current_metrics:
                    metrics = self._current_metrics[request_id]
                    metrics.detection_fps = fps
                    metrics.detection_latency_ms = latency_ms
                    metrics.objects_detected = objects_count
        
        # Also feed to unified logger if available
        if UNIFIED_LOGGING_AVAILABLE:
            try:
                logger_integration = get_logger_integration()
                # For detection frames, we'll use empty objects list since we only have count
                placeholder_objects = [f"object_{i}" for i in range(objects_count)]
                logger_integration.log_detection_frame(fps, placeholder_objects, latency_ms)
            except:
                pass  # Graceful fallback
    
    def record_face_recognition_metrics(self, request_id: str, faces_detected: int, 
                                       latency_ms: float, faces_recognized: int):
        """Record face recognition performance metrics"""
        with self._metrics_lock:
            if request_id in self._current_metrics:
                metrics = self._current_metrics[request_id]
                metrics.faces_detected = faces_detected
                metrics.face_recognition_latency_ms = latency_ms
                metrics.faces_recognized = faces_recognized
        
        # Also feed to unified logger if available
        if UNIFIED_LOGGING_AVAILABLE:
            try:
                logger_integration = get_logger_integration()
                logger_integration.log_face_recognition(faces_detected, latency_ms)
            except:
                pass  # Graceful fallback
    
    def record_planning_metrics(self, request_id: str, latency_ms: float, cache_hit: bool, fast_path: bool):
        """Record query planning performance metrics"""
        with self._metrics_lock:
            if request_id in self._current_metrics:
                metrics = self._current_metrics[request_id]
                metrics.planning_latency_ms = latency_ms
                metrics.plan_cache_hit = cache_hit
                metrics.fast_path_used = fast_path
    
    def record_execution_metrics(self, request_id: str, latency_ms: float):
        """Record plan execution performance metrics"""
        with self._metrics_lock:
            if request_id in self._current_metrics:
                metrics = self._current_metrics[request_id]
                metrics.execution_latency_ms = latency_ms
    
    def record_audio_metrics(self, request_id: str, audio_duration_ms: float, 
                           stt_latency_ms: float, tts_latency_ms: float = 0.0):
        """Record audio processing performance metrics"""
        with self._metrics_lock:
            if request_id in self._current_metrics:
                metrics = self._current_metrics[request_id]
                metrics.audio_duration_ms = audio_duration_ms
                metrics.stt_latency_ms = stt_latency_ms
                metrics.tts_latency_ms = tts_latency_ms
        
        # Also feed to unified logger if available
        if UNIFIED_LOGGING_AVAILABLE:
            try:
                logger_integration = get_logger_integration()
                logger_integration.log_audio_processing(
                    stt_time_ms=stt_latency_ms,
                    tts_time_ms=tts_latency_ms,
                    text_length=int(audio_duration_ms / 100)  # Approximate text length
                )
            except:
                pass  # Graceful fallback
    
    def record_vlm_metrics(self, request_id: str, calls_made: int, latency_ms: float, 
                          scene_analysis_ms: float = 0.0, object_description_ms: float = 0.0):
        """Record VLM/Pixtral processing performance metrics"""
        with self._metrics_lock:
            if request_id in self._current_metrics:
                metrics = self._current_metrics[request_id]
                metrics.vlm_calls_made = calls_made
                metrics.vlm_latency_ms = latency_ms
                metrics.scene_analysis_latency_ms = scene_analysis_ms
                metrics.object_description_latency_ms = object_description_ms
        
        self.total_vlm_calls += calls_made
        
        # Also feed to unified logger if available
        if UNIFIED_LOGGING_AVAILABLE and calls_made > 0:
            try:
                logger_integration = get_logger_integration()
                model_type = "pixtral-scene" if scene_analysis_ms > 0 else "pixtral-objects"
                logger_integration.log_vlm_call(latency_ms, model_type)
            except:
                pass  # Graceful fallback
    
    def record_temporal_memory_metrics(self, request_id: str, objects_count: int, memory_mb: float):
        """Record temporal memory performance metrics"""
        with self._metrics_lock:
            if request_id in self._current_metrics:
                metrics = self._current_metrics[request_id]
                metrics.temporal_objects_count = objects_count
                metrics.temporal_memory_mb = memory_mb
    
    def finish_query_measurement(self, request_id: str, total_latency_ms: float) -> Optional[PerformanceMetrics]:
        """
        Finish measuring performance for a query and write to CSV.
        Returns the final metrics object.
        """
        with self._metrics_lock:
            if request_id not in self._current_metrics:
                return None
            
            metrics = self._current_metrics[request_id]
            metrics.total_query_latency_ms = total_latency_ms
            
            # If system metrics weren't captured by background monitoring, capture them now
            if metrics.cpu_usage_percent == 0.0:
                try:
                    metrics.cpu_usage_percent = psutil.cpu_percent(interval=None)
                    metrics.memory_usage_mb = psutil.virtual_memory().used / 1024 / 1024
                    
                    if GPU_MONITORING_AVAILABLE and GPUtil:
                        try:
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                gpu = gpus[0]
                                metrics.gpu_usage_percent = gpu.load * 100
                                metrics.gpu_memory_mb = gpu.memoryUsed
                        except:
                            pass
                except:
                    pass
            
            # Calculate estimated cost (rough estimation based on VLM calls)
            # Assuming ~$0.002 per VLM call (Mistral API pricing)
            metrics.estimated_cost_cents = metrics.vlm_calls_made * 0.2  # 0.2 cents per call
            
            # Write to CSV
            self._write_metrics_to_csv(metrics)
            
            # Remove from active tracking
            del self._current_metrics[request_id]
            
            return metrics
    
    def _write_metrics_to_csv(self, metrics: PerformanceMetrics):
        """Write metrics to CSV file"""
        try:
            with open(self.metrics_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics.timestamp, metrics.request_id, metrics.query_text, metrics.query_type,
                    metrics.detection_fps, metrics.detection_latency_ms, metrics.objects_detected,
                    metrics.faces_detected, metrics.face_recognition_latency_ms, metrics.faces_recognized,
                    metrics.total_query_latency_ms, metrics.planning_latency_ms, metrics.execution_latency_ms,
                    metrics.audio_duration_ms, metrics.stt_latency_ms, metrics.tts_latency_ms,
                    metrics.vlm_calls_made, metrics.vlm_latency_ms, metrics.scene_analysis_latency_ms, 
                    metrics.object_description_latency_ms,
                    metrics.plan_cache_hit, metrics.fast_path_used,
                    metrics.cpu_usage_percent, metrics.memory_usage_mb, metrics.gpu_usage_percent, metrics.gpu_memory_mb,
                    metrics.temporal_objects_count, metrics.temporal_memory_mb, metrics.estimated_cost_cents
                ])
        except Exception as e:
            print(f"[WARNING] Failed to write metrics to CSV: {e}")
    
    def _write_detection_frame_to_csv(self, fps: float, latency_ms: float, objects_count: int):
        """Write continuous detection frame metrics directly to CSV"""
        try:
            # Create a detection-only metrics entry
            current_time = time.time()
            
            # Get current system metrics
            cpu_percent = 0.0
            memory_mb = 0.0
            gpu_percent = 0.0
            gpu_memory_mb = 0.0
            
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_mb = psutil.virtual_memory().used / 1024 / 1024
                
                if GPU_MONITORING_AVAILABLE and GPUtil:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            gpu_percent = gpu.load * 100
                            gpu_memory_mb = gpu.memoryUsed
                    except:
                        pass
            except:
                pass
            
            with open(self.metrics_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    current_time, f"detection_{int(current_time * 1000)}", "[Detection Frame]", "Detection",
                    fps, latency_ms, objects_count,
                    0, 0.0, 0,  # faces_detected, face_recognition_latency_ms, faces_recognized
                    0.0, 0.0, 0.0,  # total_query_latency_ms, planning_latency_ms, execution_latency_ms
                    0.0, 0.0, 0.0,  # audio_duration_ms, stt_latency_ms, tts_latency_ms
                    0, 0.0, 0.0, 0.0,  # vlm_calls_made, vlm_latency_ms, scene_analysis_latency_ms, object_description_latency_ms
                    False, False,  # plan_cache_hit, fast_path_used
                    cpu_percent, memory_mb, gpu_percent, gpu_memory_mb,
                    0, 0.0,  # temporal_objects_count, temporal_memory_mb
                    0.0  # estimated_cost_cents
                ])
        except Exception as e:
            print(f"[WARNING] Failed to write detection frame to CSV: {e}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get current summary statistics"""
        avg_fps = sum(self.detection_fps_samples) / len(self.detection_fps_samples) if self.detection_fps_samples else 0
        
        return {
            "total_queries": self.total_queries,
            "total_vlm_calls": self.total_vlm_calls,
            "detection_frames": self.detection_frame_count,
            "average_detection_fps": round(avg_fps, 2),
            "vlm_calls_per_100_queries": round((self.total_vlm_calls / max(1, self.total_queries)) * 100, 1),
            "system_config": {
                "platform": self.system_config.platform,
                "cpu_cores": self.system_config.cpu_cores,
                "memory_gb": round(self.system_config.total_memory_gb, 1),
                "gpu_model": self.system_config.gpu_model
            }
        }
    
    def print_summary(self):
        """Print current performance summary"""
        stats = self.get_summary_statistics()
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Total VLM Calls: {stats['total_vlm_calls']}")
        print(f"VLM Calls per 100 Queries: {stats['vlm_calls_per_100_queries']}")
        print(f"Detection Frames: {stats['detection_frames']}")
        print(f"Average Detection FPS: {stats['average_detection_fps']}")
        print(f"System: {stats['system_config']['platform']}")
        print(f"CPU Cores: {stats['system_config']['cpu_cores']}")
        print(f"Memory: {stats['system_config']['memory_gb']} GB")
        print(f"GPU: {stats['system_config']['gpu_model']}")
        print(f"CSV Output: {self.metrics_csv}")
        print("=" * 30)


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor

def init_performance_monitor(output_dir: str = "test_results") -> PerformanceMonitor:
    """Initialize global performance monitor with custom output directory"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()
    _global_monitor = PerformanceMonitor(output_dir)
    _global_monitor.start_monitoring()
    return _global_monitor
