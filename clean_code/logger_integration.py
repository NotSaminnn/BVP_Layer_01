"""
Integration script to add unified logging to existing LUMENAA agent

This script modifies the existing agent_runner.py to include unified logging
without changing the core functionality.
"""

import sys
import os
import time
import threading
import atexit

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from core.infrastructure.unified_logger import get_unified_logger, finalize_logging
    LOGGING_AVAILABLE = True
except ImportError:
    print("Unified logging not available - running without detailed logging")
    LOGGING_AVAILABLE = False


class AgentLoggerIntegration:
    """Integration layer for adding unified logging to existing agent"""
    
    def __init__(self):
        if LOGGING_AVAILABLE:
            self.logger = get_unified_logger()
            print(f"Unified logging initialized - Session ID: {self.logger.session_id}")
            atexit.register(self._cleanup)
        else:
            self.logger = None
    
    def _cleanup(self):
        """Cleanup function called on exit"""
        if self.logger:
            print("Finalizing logging session...")
            finalize_logging()
    
    def log_detection_frame(self, fps=0, objects=None, latency_ms=0):
        """Log a detection frame"""
        if self.logger:
            objects = objects or []
            self.logger.log_detection_frame(fps, objects, latency_ms)
    
    def log_query_start(self, query_text, query_type="user"):
        """Log query start and return query ID"""
        if self.logger:
            return self.logger.log_query_start(query_text, query_type)
        return "no_logging"
    
    def log_query_end(self, query_id, success, response="", latency_ms=0, error_msg=""):
        """Log query completion"""
        if self.logger:
            self.logger.log_query_end(query_id, success, response, latency_ms, error_msg)
    
    def log_face_recognition(self, faces_detected, latency_ms=0):
        """Log face recognition"""
        if self.logger:
            self.logger.log_face_recognition(faces_detected, latency_ms)
    
    def log_audio_processing(self, stt_time_ms=0, tts_time_ms=0, text_length=0):
        """Log audio processing"""
        if self.logger:
            self.logger.log_audio_processing(stt_time_ms, tts_time_ms, text_length)
    
    def log_vlm_call(self, latency_ms, model="pixtral"):
        """Log VLM call"""
        if self.logger:
            self.logger.log_vlm_call(latency_ms, model)
    
    def log_system_metrics(self, cpu_percent, memory_percent, gpu_percent=0):
        """Log system metrics"""
        if self.logger:
            self.logger.log_system_metrics(cpu_percent, memory_percent, gpu_percent)
    
    def log_info(self, category, message, data=None):
        """Log general info message"""
        if self.logger:
            self.logger.log("INFO", category, message, data)
    
    def log_error(self, category, message, data=None):
        """Log error message"""
        if self.logger:
            self.logger.log("ERROR", category, message, data)
    
    def get_stats(self):
        """Get current session stats"""
        if self.logger:
            return self.logger.get_stats()
        return {}
    
    def print_stats(self):
        """Print current session stats"""
        if not self.logger:
            print("No logging available")
            return
        
        stats = self.logger.get_stats()
        print(f"\n=== SESSION STATS ===")
        print(f"Duration: {stats.get('duration_minutes', 0):.1f} minutes")
        print(f"Queries: {stats.get('total_queries', 0)}")
        print(f"Detection Frames: {stats.get('total_detection_frames', 0)}")
        print(f"Average FPS: {stats.get('average_fps', 0):.1f}")
        print(f"VLM Calls: {stats.get('total_vlm_calls', 0)}")
        print(f"Objects Detected: {stats.get('total_objects_detected', 0)}")
        print(f"Unique Object Types: {len(stats.get('unique_object_types', []))}")
        print(f"Log File: {self.logger.log_file if self.logger else 'N/A'}")
        print("=" * 20)


# Global logger integration instance
_logger_integration = None

def get_logger_integration():
    """Get the global logger integration instance"""
    global _logger_integration
    if _logger_integration is None:
        _logger_integration = AgentLoggerIntegration()
    return _logger_integration


# Performance monitoring helper
class PerformanceTimer:
    """Simple performance timer context manager"""
    
    def __init__(self, name, logger_integration=None):
        self.name = name
        self.logger = logger_integration or get_logger_integration()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed_ms = (time.time() - self.start_time) * 1000
            if exc_type is None:
                print(f"[PERF] {self.name}: {elapsed_ms:.1f}ms")
            else:
                self.logger.log_error("performance", f"{self.name} failed after {elapsed_ms:.1f}ms")


def timer(name):
    """Decorator for timing functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceTimer(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# System monitoring in background
class SystemMonitor:
    """Background system monitoring"""
    
    def __init__(self):
        self.monitoring = False
        self.thread = None
        self.logger = get_logger_integration()
    
    def start(self):
        """Start background monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            print("System monitoring started")
    
    def stop(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        try:
            import psutil
            
            # Try to detect GPU once at startup
            gpu_method = self._detect_gpu_method()
            
            while self.monitoring:
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    gpu_percent = self._get_gpu_usage(gpu_method)
                    
                    self.logger.log_system_metrics(cpu_percent, memory_percent, gpu_percent)
                    
                    time.sleep(5)  # Monitor every 5 seconds
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(10)
                    
        except ImportError:
            print("psutil not available - system monitoring disabled")
    
    def _detect_gpu_method(self):
        """Detect which GPU monitoring method to use"""
        # Try PyTorch CUDA first
        try:
            import torch
            if torch.cuda.is_available():
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                return 'torch'
        except:
            pass
        
        # Try GPUtil for NVIDIA
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                print(f"GPU detected: {gpus[0].name}")
                return 'gputil'
        except:
            pass
        
        # Try pynvml for NVIDIA
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            print(f"GPU detected: {name}")
            return 'pynvml'
        except:
            pass
        
        print("No GPU detected or GPU monitoring not available")
        return None
    
    def _get_gpu_usage(self, method):
        """Get GPU usage based on detected method"""
        if method == 'torch':
            try:
                import torch
                if torch.cuda.is_available():
                    # Get memory usage as a proxy for utilization
                    memory_allocated = torch.cuda.memory_allocated(0)
                    memory_reserved = torch.cuda.memory_reserved(0)
                    max_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_percent = (memory_reserved / max_memory) * 100
                    return gpu_percent
            except Exception as e:
                pass
        
        elif method == 'gputil':
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].load * 100
            except:
                pass
        
        elif method == 'pynvml':
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return util.gpu
            except:
                pass
        
        return 0


# Global system monitor
_system_monitor = None

def start_system_monitoring():
    """Start background system monitoring"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    _system_monitor.start()

def stop_system_monitoring():
    """Stop background system monitoring"""
    global _system_monitor
    if _system_monitor:
        _system_monitor.stop()


if __name__ == "__main__":
    # Test the integration
    logger = get_logger_integration()
    
    print("Testing unified logging integration...")
    
    # Test logging
    query_id = logger.log_query_start("test query")
    time.sleep(0.1)
    logger.log_query_end(query_id, True, "test response", 100)
    
    logger.log_detection_frame(30.0, ["person", "chair"], 33)
    logger.log_face_recognition(2, 50)
    logger.log_vlm_call(500)
    
    # Start system monitoring
    start_system_monitoring()
    
    # Wait a bit
    time.sleep(2)
    
    # Print stats
    logger.print_stats()
    
    # Cleanup
    stop_system_monitoring()
    finalize_logging()
    
    print("Integration test completed!")