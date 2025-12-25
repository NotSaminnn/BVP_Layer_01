from __future__ import annotations
import asyncio
import os
import sys
import cv2
import time

# Import unified logging integration
try:
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from logger_integration import get_logger_integration, start_system_monitoring, PerformanceTimer
	UNIFIED_LOGGING_AVAILABLE = True
except ImportError:
	UNIFIED_LOGGING_AVAILABLE = False

# Robust imports: allow running as a script (python agent_runner.py) and as a module (-m)
try:
	from core.event_bus import EventBus
	from core.infrastructure.tool_registry import ToolRegistry
	from core.adapters.stt import STTAdapter
	from core.adapters.tts import TTSAdapter
	from core.adapters.object_detector import ObjectDetectorAdapter
	from core.adapters.scene_analysis import SceneAnalysisAdapter
	from core.adapters.chatbot import ChatbotAdapter
	from core.adapters.document_scan import DocumentScanAdapter
	from core.adapters.pixtral_analysis import PixtralAnalysisAdapter
	from core.adapters.face_recognition import FaceRecognitionAdapter
	from core.controller import AgentController
	from core.infrastructure.recorder_fsm import RecordingFSM
	from core.infrastructure.frame_provider import FrameProvider
	from core.infrastructure.observability import log
except ImportError:
	# Fallback for direct execution from this folder
	PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	if PROJECT_ROOT not in sys.path:
		sys.path.append(PROJECT_ROOT)
	from BVP_LAYER01.agent.event_bus import EventBus  # type: ignore
	from BVP_LAYER01.agent.tool_registry import ToolRegistry  # type: ignore
	from BVP_LAYER01.agent.stt_adapter import STTAdapter  # type: ignore
	from BVP_LAYER01.agent.tts_adapter import TTSAdapter  # type: ignore
	from BVP_LAYER01.agent.object_detector_adapter import ObjectDetectorAdapter  # type: ignore
	from BVP_LAYER01.agent.scene_analysis_adapter import SceneAnalysisAdapter  # type: ignore
	from BVP_LAYER01.agent.chatbot_adapter import ChatbotAdapter  # type: ignore
	from BVP_LAYER01.agent.document_scan_adapter import DocumentScanAdapter  # type: ignore
	from BVP_LAYER01.agent.pixtral_analysis_adapter import PixtralAnalysisAdapter  # type: ignore
	from BVP_LAYER01.agent.face_recognition_adapter import FaceRecognitionAdapter  # type: ignore
	from BVP_LAYER01.agent.controller import AgentController  # type: ignore
	from BVP_LAYER01.agent.recorder_fsm import RecordingFSM  # type: ignore
	from BVP_LAYER01.agent.frame_provider import FrameProvider  # type: ignore
	from BVP_LAYER01.agent.observability import log  # type: ignore


# Global performance tracking
performance_monitor = None


async def continuous_detection_loop(cap: cv2.VideoCapture, detector: ObjectDetectorAdapter, frames: FrameProvider, context_window=None, face_recognition=None):
	"""Continuous detection loop that runs in background, shows live monitor window, and updates temporal memory."""
	global performance_monitor
	
	try:
		from core.modules.object_detection.utils import draw_simple_detection  # type: ignore
	except ImportError:
		draw_simple_detection = None
	
	window_name = "Agent Detection Monitor"
	show_monitor = os.environ.get("AGENT_VISUAL_MONITOR", "1") in ("1", "true", "True", "yes", "Yes")
	
	# Performance tracking variables
	last_fps_time = time.time()
	frame_count = 0
	
	try:
		while True:
			frame_start_time = time.time()
			
			ret, frame = cap.read()
			if not ret:
				await asyncio.sleep(0.1)
				continue
			frames.set_latest(frame)
			
			# Run detection (non-blocking via thread pool)
			detection_start_time = time.time()
			detections = await asyncio.to_thread(detector.detect, frame, target_classes=None)
			detection_latency_ms = (time.time() - detection_start_time) * 1000
			
			# Calculate FPS
			frame_count += 1
			current_time = time.time()
			fps = 1.0 / (current_time - last_fps_time) if (current_time - last_fps_time) > 0 else 0
			last_fps_time = current_time
			
			# Log detection frame for unified logging
			if UNIFIED_LOGGING_AVAILABLE:
				try:
					logger_int = get_logger_integration()
					object_classes = [obj.get("class", "") for obj in detections] if detections else []
					logger_int.log_detection_frame(fps, object_classes, detection_latency_ms)
				except:
					pass
			
			# Record detection metrics if performance monitor is active
			if performance_monitor:
				# Use a general detection request ID for continuous monitoring
				detection_request_id = f"detection_{int(current_time * 1000)}"
				performance_monitor.record_detection_metrics(
					detection_request_id, fps, detection_latency_ms, len(detections) if detections else 0
				)
			
			# Record frame metrics in experimental metrics
			try:
				from core.metrics.agent_bridge import get_agent_bridge
				bridge = get_agent_bridge()
				if bridge:
					objects_count = len(detections) if detections else 0
					bridge.record_frame_metrics(objects_count, fps, vlm_calls=0, vlm_with_tracking=0)
			except:
				pass  # Experimental metrics is optional
			
			# Run face recognition on person objects if face recognition is available
			if face_recognition and detections:
				person_objects = [obj for obj in detections if obj.get("class", "").lower() == "person"]
				if person_objects:
					try:
						# Run face recognition in thread pool (non-blocking)
						face_recognition_start_time = time.time()
						recognized_persons = await asyncio.to_thread(
							face_recognition.recognize_person_objects, frame, person_objects
						)
						face_recognition_latency_ms = (time.time() - face_recognition_start_time) * 1000
						
						# Count successfully recognized faces
						faces_recognized = sum(1 for rp in recognized_persons 
						                     if rp.get("recognized_name") and rp.get("recognized_name") != "Unknown") if recognized_persons else 0
						
						# Log face recognition for unified logging
						if UNIFIED_LOGGING_AVAILABLE:
							try:
								logger_int = get_logger_integration()
								logger_int.log_face_recognition(len(person_objects), face_recognition_latency_ms)
							except:
								pass
						
						# Record face recognition metrics if performance monitor is active
						if performance_monitor:
							performance_monitor.record_face_recognition_metrics(
								detection_request_id, len(person_objects), face_recognition_latency_ms, faces_recognized
							)
						
						# Debug: Log recognition results
						if recognized_persons:
							for rec_person in recognized_persons:
								rec_name = rec_person.get("recognized_name")
								rec_conf = rec_person.get("face_confidence", 0)
								if rec_name and rec_name != "Unknown":
									print(f"[DEBUG] Face recognition: Recognized '{rec_name}' with confidence {rec_conf:.2f}")
						
						# Update detections with recognized names
						# recognize_person_objects processes persons in order and returns in same order,
						# so we can directly match by index
						
						# Get indices of person objects in detections list
						person_indices = []
						for i, obj in enumerate(detections):
							if obj.get("class", "").lower() == "person":
								person_indices.append(i)
						
						# Match recognized persons to detections by index (order is preserved)
						# Each recognized_person corresponds to the person at the same index
						for rec_idx, rec_person in enumerate(recognized_persons):
							if rec_idx >= len(person_indices):
								break  # More recognized persons than detections (shouldn't happen)
							
							# Get the corresponding person detection index
							person_det_idx = person_indices[rec_idx]
							rec_name = rec_person.get("recognized_name")
							
							# Update detection with recognized name if recognition was successful
							if rec_name and rec_name != "Unknown":
								detections[person_det_idx]["class"] = rec_name
								detections[person_det_idx]["recognized_name"] = rec_name
								detections[person_det_idx]["face_confidence"] = rec_person.get("face_confidence", 0)
								print(f"[DEBUG] Matched person {rec_idx} (detection index {person_det_idx}) to '{rec_name}'")
					except Exception as e:
						# Face recognition failed - log the error but continue
						print(f"[WARNING] Face recognition error: {e}")
						import traceback
						traceback.print_exc()
			
			# Update temporal memory if context window available
			if context_window:
				context_window.update_environment_facts(detections, scene_context=None)
				# Also update cache so chatbot step can access recent objects
				if detections:
					context_window.cache_set("last_objects", detections)
			if show_monitor:
				# Only copy frame if we have detections to draw, otherwise use frame directly
				if draw_simple_detection and detections:
					img = frame.copy()
					for d in detections:
						bbox = d.get("bbox") or []
						if not bbox or len(bbox) != 4:
							continue
						cls = d.get("class", "object")
						conf = float(d.get("confidence", 0.0))
						dist = float(d.get("approxDistance", 0.0))
						ax = float(d.get("angle_x", 0.0))
						ay = float(d.get("angle_y", 0.0))
						img = draw_simple_detection(img, bbox, str(cls), conf, dist, ax, ay)
					cv2.imshow(window_name, img)
				else:
					# No detections - show frame directly without copy
					cv2.imshow(window_name, frame)
				cv2.waitKey(1)
			await asyncio.sleep(0.067)  # ~15 FPS detection loop (reduced from 30 FPS for better performance)
	except asyncio.CancelledError:
		pass
	finally:
		try:
			cv2.destroyWindow(window_name)
		except Exception:
			pass


async def run_agent(video_source: int | str = 0) -> int:
	global performance_monitor
	
	# Initialize performance monitoring
	try:
		from core.infrastructure.performance_monitor import init_performance_monitor
		
		# Check if custom output directory is specified via environment variable
		custom_output_dir = os.environ.get("PERFORMANCE_MONITOR_OUTPUT_DIR")
		if custom_output_dir:
			performance_monitor = init_performance_monitor(custom_output_dir)
			print(f"[INFO] Performance monitoring initialized with custom directory: {custom_output_dir}")
		else:
			performance_monitor = init_performance_monitor()
			print(f"[INFO] Performance monitoring initialized: {performance_monitor.metrics_csv}")
			
	except ImportError:
		print(f"[WARNING] Performance monitoring not available")
		performance_monitor = None
	
	# Initialize unified logging
	if UNIFIED_LOGGING_AVAILABLE:
		logger_integration = get_logger_integration()
		start_system_monitoring()
		print(f"[INFO] Unified logging initialized - Session ID: {logger_integration.logger.session_id if logger_integration.logger else 'N/A'}")
	else:
		logger_integration = None
		print(f"[WARNING] Unified logging not available")
	
	# Initialize experimental metrics bridge
	try:
		from core.metrics.agent_bridge import init_agent_bridge
		bridge = init_agent_bridge()
		if bridge:
			print(f"[INFO] Experimental metrics bridge initialized")
		else:
			print(f"[INFO] Experimental metrics bridge not enabled")
	except ImportError:
		print(f"[WARNING] Experimental metrics bridge not available")
	
	# Ensure API key is available for adapters that need it
	if not os.environ.get("MISTRAL_API_KEY"):
		os.environ["MISTRAL_API_KEY"] = "n3ttDOXkxoavn3ClnGBcqaDdCr0HtKob"

	event_bus = EventBus()
	registry = ToolRegistry()
	config_path = os.path.join(os.path.dirname(__file__), "..", "config", "tools.yaml")
	registry.load_from_file(os.path.abspath(config_path))

	# Initialize camera early (stays open)
	cap = cv2.VideoCapture(video_source)
	if not cap.isOpened():
		log("Could not open video source", {"video_source": video_source})
		return 1
	cap.set(3, 1280)
	cap.set(4, 720)

	# Initialize detector early (loaded once at startup)
	log("Initializing YOLO detector...")
	
	# Check if GPU should be forced to CPU (via environment variable)
	force_cpu = os.environ.get("FORCE_CPU", "false").lower() in ("true", "1", "yes")
	obj = ObjectDetectorAdapter(force_cpu=force_cpu)
	frames = FrameProvider()

	# Initialize other adapters
	# Enable accent optimization for better Indian English and other accent recognition
	accent_optimization = os.environ.get("STT_ACCENT_OPTIMIZATION", "true").lower() in ("true", "1", "yes")
	stt = STTAdapter(event_bus, accent_optimization=accent_optimization)
	tts = TTSAdapter(event_bus)
	scene = SceneAnalysisAdapter(api_key=os.environ.get("MISTRAL_API_KEY"))
	chat = ChatbotAdapter(api_key=os.environ.get("MISTRAL_API_KEY"))
	doc = DocumentScanAdapter(api_key=os.environ.get("MISTRAL_API_KEY"))
	pixtral = PixtralAnalysisAdapter(api_key=os.environ.get("MISTRAL_API_KEY"), verbose=False)
	
	# Initialize face recognition adapter (optional - will work even if gallery is empty)
	try:
		face_recognition = FaceRecognitionAdapter(
			gallery_path=None,  # Auto-detect gallery.pkl in Facenet directory
			threshold=0.70,  # Lower threshold for better recognition of tilted faces
			force_cpu=force_cpu,
			verbose=False
		)
		gallery_names = face_recognition.get_gallery_names()
		if gallery_names:
			log(f"Face recognition initialized with {len(gallery_names)} enrolled people: {', '.join(gallery_names)}")
		else:
			log("Face recognition initialized but gallery is empty. Enroll people first using facenet_multi.py")
	except Exception as e:
		print(f"[WARNING] Face recognition initialization failed: {e}")
		face_recognition = None

	controller = AgentController(
		event_bus,
		registry,
		obj,
		scene,
		chat,
		document_scan_fn=doc,  # Pass adapter instance, not just scan method
		pixtral_analysis=pixtral,
		face_recognition=face_recognition,
		frame_provider=frames.get_latest,
	)
	
	# Start continuous detection loop in background (with temporal memory updates and face recognition)
	detection_task = asyncio.create_task(continuous_detection_loop(cap, obj, frames, context_window=controller.context, face_recognition=face_recognition))
	
	fsm = RecordingFSM(event_bus, stt)

	log("Agent ready. Controls: r=start/stop recording, q=quit (press in terminal)")
	log("Detection monitor window shows live detections continuously")

	# Cross-platform terminal key loop - MUST always work, no fallback to disabled state
	import platform
	key_handler = None
	key_handler_type = None  # Track handler type: "crossplatform", "msvcrt", or "unix"
	key_handler_initialized = False
	
	# Initialize keyboard handler with multiple fallbacks
	try:
		from BVP_LAYER01.utils.cross_platform_keypress import CrossPlatformKeypress
		key_handler = CrossPlatformKeypress()
		key_handler_type = "crossplatform"
		key_handler_initialized = True
		log("Keyboard input: Using CrossPlatformKeypress handler")
	except ImportError:
		# Fallback: try to import from relative path
		utils_path = os.path.join(os.path.dirname(__file__), "..", "utils")
		if utils_path not in sys.path:
			sys.path.insert(0, utils_path)
		try:
			from cross_platform_keypress import CrossPlatformKeypress  # type: ignore
			key_handler = CrossPlatformKeypress()
			key_handler_type = "crossplatform"
			key_handler_initialized = True
			log("Keyboard input: Using CrossPlatformKeypress handler (relative import)")
		except ImportError:
			# Initialize platform-specific handlers
			if platform.system() == "Windows":
				try:
					import msvcrt
					key_handler_type = "msvcrt"
					key_handler_initialized = True
					log("Keyboard input: Using Windows msvcrt handler")
				except ImportError:
					pass
			else:
				# Unix/Linux/macOS
				try:
					import select
					import termios
					import tty
					key_handler_type = "unix"
					key_handler_initialized = True
					log("Keyboard input: Using Unix select/termios handler")
				except ImportError:
					pass
	
	# If keyboard handler couldn't be initialized, exit with error
	if not key_handler_initialized:
		log("ERROR: Could not initialize keyboard input handler. Exiting.", level="error")
		print("\n[ERROR] Keyboard input is required but could not be initialized.")
		print("[ERROR] Please ensure you're running in a terminal that supports keyboard input.")
		return 1
	
	# Main keyboard input loop - with per-iteration error handling
	_last_keyboard_error = None
	try:
		while True:
			await asyncio.sleep(0.05)
			
			try:
				# Check for key press based on handler type
				ch = None
				
				if key_handler_type == "crossplatform" and key_handler is not None:
					# Use cross-platform handler
					if key_handler._check_key_pressed():
						ch = key_handler._get_pressed_key()
				elif key_handler_type == "msvcrt":
					# Windows fallback using msvcrt
					import msvcrt
					if msvcrt.kbhit():
						ch = msvcrt.getwch()
				elif key_handler_type == "unix":
					# Unix fallback using select
					import select
					import termios
					import tty
					if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
						fd = sys.stdin.fileno()
						old_settings = termios.tcgetattr(fd)
						try:
							tty.setraw(fd)
							ch = sys.stdin.read(1)
						finally:
							termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
				
				# Process key press
				if ch:
					if ch.lower() == 'q':
						break
					if ch.lower() == 'r':
						try:
							await fsm.handle_key_r()
						except Exception as e:
							# Log error but continue keyboard monitoring
							print(f"[WARNING] Error handling 'r' key: {e}")
							import traceback
							traceback.print_exc()
							continue
				
				# Reset error tracking on successful iteration
				_last_keyboard_error = None
			except KeyboardInterrupt:
				# User pressed Ctrl+C - exit gracefully
				print("\n[INFO] Interrupted by user (Ctrl+C)")
				break
			except Exception as e:
				# Log error but continue keyboard monitoring - don't disable it
				# Only log if it's a new error (avoid spam)
				error_str = str(e)
				if _last_keyboard_error != error_str:
					print(f"[WARNING] Keyboard input error (continuing): {e}")
					_last_keyboard_error = error_str
					import traceback
					traceback.print_exc()
				await asyncio.sleep(0.1)  # Brief pause before retrying
				continue
				
	except KeyboardInterrupt:
		print("\n[INFO] Interrupted by user")
	finally:
		detection_task.cancel()
		try:
			await detection_task
		except asyncio.CancelledError:
			pass
		cap.release()
		cv2.destroyAllWindows()
		
		# Cleanup performance monitoring and print summary
		if performance_monitor:
			performance_monitor.stop_monitoring()
			performance_monitor.print_summary()
		
		log("Agent runner stopped")
	return 0


def main() -> int:
	return asyncio.run(run_agent())


if __name__ == "__main__":
	exit(main())
