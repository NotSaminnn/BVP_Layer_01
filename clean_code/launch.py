"""
Launch LUMENAA Agent with Unified Logging and Experimental Metrics

This script launches the LUMENAA agent with comprehensive unified logging
that saves all activities and performance metrics to a single JSON log file.
Additionally, it captures experimental metrics for research and optimization:

- End-to-End Query Latency by type
- Fast-path vs LLM execution times
- Hybrid Planning Performance
- Tracking-Guided VLM Optimization
- API Call Frequency and Cost Analysis
- Automated reporting and visualization
"""

import os
import sys
import asyncio
import signal
import atexit

# Add the current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the logger integration first
try:
    from logger_integration import get_logger_integration, finalize_logging, start_system_monitoring, stop_system_monitoring
    LOGGING_AVAILABLE = True
except ImportError:
    print("WARNING: Unified logging not available")
    LOGGING_AVAILABLE = False

# Import the main agent runner
import core.agent_runner as agent_runner

# Import the experimental metrics module
try:
    from core.metrics.system import ExperimentalMetricsSystem
    EXPERIMENTAL_METRICS_AVAILABLE = True
except ImportError:
    print("WARNING: Experimental metrics module not available")
    EXPERIMENTAL_METRICS_AVAILABLE = False
    ExperimentalMetricsSystem = None

# Global variables for cleanup
logger_integration = None
system_monitoring_started = False
experimental_metrics_system = None

def cleanup_handler():
    """Cleanup function called on exit"""
    global logger_integration, system_monitoring_started, experimental_metrics_system
    
    print("\nCleaning up...")
    
    # Stop experimental metrics system
    if experimental_metrics_system:
        experimental_metrics_system.stop()
        experimental_metrics_system.print_stats()
    
    if system_monitoring_started:
        stop_system_monitoring()
    
    if logger_integration:
        logger_integration.print_stats()
    
    if LOGGING_AVAILABLE:
        finalize_logging()

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    cleanup_handler()
    sys.exit(0)

async def main():
    """Main entry point"""
    global logger_integration, system_monitoring_started, experimental_metrics_system
    
    print("=" * 60)
    print("LUMENAA Agent with Unified Logging & Experimental Metrics")
    print("=" * 60)
    
    # Initialize experimental metrics system
    if EXPERIMENTAL_METRICS_AVAILABLE:
        try:
            experimental_metrics_system = ExperimentalMetricsSystem()
            print("✓ Experimental metrics system initialized")
        except Exception as e:
            print(f"⚠ Failed to initialize experimental metrics: {e}")
            experimental_metrics_system = None
    else:
        print("⚠ Experimental metrics not available")
    
    # Initialize unified logging
    if LOGGING_AVAILABLE:
        logger_integration = get_logger_integration()
        start_system_monitoring()
        system_monitoring_started = True
        
        print(f"✓ Unified logging initialized")
        print(f"✓ Session ID: {logger_integration.logger.session_id if logger_integration.logger else 'N/A'}")
        print(f"✓ Log file: {logger_integration.logger.log_file if logger_integration.logger else 'N/A'}")
        print(f"✓ System monitoring started")
    else:
        print("⚠ Unified logging not available")
    
    # Register cleanup handlers
    atexit.register(cleanup_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("✓ Agent initialization complete")
    print()
    print("Commands:")
    print("  'r' - Start/stop recording")
    print("  'q' - Quit agent")
    print("  'Ctrl+C' - Emergency stop")
    print()
    print("The agent will save comprehensive logs for this session.")
    print("All performance metrics, queries, detections, and system stats")
    print("will be logged to a single JSON file with a summary.")
    print()
    if experimental_metrics_system:
        print("Experimental metrics are being collected:")
        print("- Query latency and performance analysis")
        print("- API call frequency and cost tracking")
        print("- VLM optimization measurements")
        print("- Automated reporting every 5 minutes")
        print(f"- Data saved to: {experimental_metrics_system.get_data_directory()}")
        print(f"- Session ID: {experimental_metrics_system.session_id}")
        print()
    print("Starting agent...")
    print("=" * 60)
    
    try:
        # Set environment variables to enable logging and metrics in agent
        os.environ["UNIFIED_LOGGING_ENABLED"] = "1"
        if experimental_metrics_system:
            os.environ["EXPERIMENTAL_METRICS_ENABLED"] = "1"
            
            # Make experimental metrics available to the agent
            import builtins
            builtins.experimental_metrics = experimental_metrics_system.collector
        
        # Run the main agent
        result = await agent_runner.run_agent()
        
        return result
        
    except KeyboardInterrupt:
        print("\nAgent stopped by user")
        return 0
    except Exception as e:
        print(f"\nAgent crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cleanup_handler()

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\nForced shutdown")
        sys.exit(130)  # Standard exit code for Ctrl+C