from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
import os
import time
import sys
import numpy as np

# Import unified logging integration
try:
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from logger_integration import get_logger_integration
	UNIFIED_LOGGING_AVAILABLE = True
except ImportError:
	UNIFIED_LOGGING_AVAILABLE = False

try:
	from core.modules.vlm.pixtral_analyzer import PixtralAnalyzer  # type: ignore
except Exception:
	PixtralAnalyzer = None  # type: ignore


class PixtralAnalysisAdapter:
	"""
	Adapter for PixtralAnalyzer that integrates AI-powered visual descriptions into the agentic system.
	Wraps PixtralAnalyzer.analyze_detections_with_tracking() to match main.py's behavior.
	"""
	
	def __init__(self, api_key: Optional[str] = None, verbose: bool = False) -> None:
		"""
		Initialize Pixtral Analysis Adapter.
		
		Args:
			api_key: Mistral API key (if None, uses MISTRAL_API_KEY env var)
			verbose: Whether to print status messages
		"""
		api_key = api_key or os.environ.get("MISTRAL_API_KEY") or ""
		self._pixtral: Optional[PixtralAnalyzer] = None
		if PixtralAnalyzer and api_key:
			try:
				self._pixtral = PixtralAnalyzer(api_key=api_key, verbose=verbose)
			except Exception as e:
				if verbose:
					print(f"[WARNING] PixtralAnalyzer initialization failed: {e}")
				self._pixtral = None
	
	def is_available(self) -> bool:
		"""Check if Pixtral analyzer is available."""
		return self._pixtral is not None
	
	def _convert_to_pixtral_format(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""
		Convert agent detection format to Pixtral-compatible format.
		
		Agent format: {
			"class": "pen",
			"bbox": [x1, y1, x2, y2],
			"confidence": 0.85,
			"approxDistance": 1.5,
			"angle_x": 15.2,
			"angle_y": -8.1,
			"id": "track_123"
		}
		
		Pixtral format: {
			"class_name": "pen",
			"bbox": [x1, y1, x2, y2],
			"confidence": 0.85,
			"distance": 1.5,
			"angle_x": 15.2,
			"angle_y": -8.1,
			"track_id": 123
		}
		"""
		compatible = []
		for d in detections:
			# Extract track_id from "track_123" format
			track_id = None
			obj_id = d.get("id", "")
			if obj_id and isinstance(obj_id, str) and obj_id.startswith("track_"):
				try:
					track_id = int(obj_id.replace("track_", ""))
				except ValueError:
					pass
			
			compatible.append({
				"class_name": d.get("class", "unknown"),
				"bbox": d.get("bbox", []),
				"confidence": float(d.get("confidence", 0.0)),
				"distance": float(d.get("approxDistance", 0.0)),
				"angle_x": float(d.get("angle_x", 0.0)),
				"angle_y": float(d.get("angle_y", 0.0)),
				"track_id": track_id,
			})
		return compatible
	
	def _convert_from_pixtral_format(self, enhanced_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""
		Convert Pixtral format back to agent format.
		
		Pixtral format (enhanced): {
			"class_name": "pen",
			"bbox": [x1, y1, x2, y2],
			"confidence": 0.85,
			"distance": 1.5,
			"angle_x": 15.2,
			"angle_y": -8.1,
			"track_id": 123,
			"pixtral_description": "A blue ballpoint pen..."
		}
		
		Agent format: {
			"class": "pen",
			"bbox": [x1, y1, x2, y2],
			"confidence": 0.85,
			"approxDistance": 1.5,
			"angle_x": 15.2,
			"angle_y": -8.1,
			"id": "track_123",
			"pixtral_description": "A blue ballpoint pen..."  # NEW FIELD
		}
		"""
		converted = []
		for d in enhanced_detections:
			track_id = d.get("track_id")
			obj_id = f"track_{track_id}" if track_id is not None else ""
			
			converted.append({
				"class": d.get("class_name", "unknown"),
				"bbox": d.get("bbox", []),
				"confidence": float(d.get("confidence", 0.0)),
				"approxDistance": float(d.get("distance", 0.0)),
				"angle_x": float(d.get("angle_x", 0.0)),
				"angle_y": float(d.get("angle_y", 0.0)),
				"id": obj_id,
				"pixtral_description": d.get("pixtral_description", ""),  # NEW FIELD
				"timestamp": d.get("analysis_timestamp"),
			})
		return converted
	
	def analyze_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
	                      new_track_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
		"""
		Analyze detections using Pixtral (like main.py does).
		Uses clipped ROI images for individual object descriptions.
		
		Args:
			frame: Input frame (numpy array)
			detections: List of detections in agent format
			new_track_ids: Set of new track IDs to analyze (for tracking-aware analysis)
		
		Returns:
			List of enhanced detections with pixtral_description field
		"""
		if not self._pixtral or not detections:
			# Return detections as-is if Pixtral not available
			return detections
		
		if frame is None or frame.size == 0:
			return detections
		
		try:
			# Convert to Pixtral format
			compatible_detections = self._convert_to_pixtral_format(detections)
			
			# Log VLM call start for unified logging
			vlm_start_time = time.time()
			
			# Run Pixtral analysis (like main.py does)
			# This analyzes individual objects using clipped ROI images
			enhanced_detections = self._pixtral.analyze_detections_with_tracking(
				frame, compatible_detections, new_track_ids=new_track_ids or set()
			)
			
			# Log VLM call completion for unified logging
			vlm_latency = (time.time() - vlm_start_time) * 1000
			if UNIFIED_LOGGING_AVAILABLE:
				try:
					logger_integration = get_logger_integration()
					logger_integration.log_vlm_call(vlm_latency, "pixtral-detections")
				except:
					pass
			
			# Record VLM call in experimental metrics
			try:
				from core.metrics.agent_bridge import get_agent_bridge
				bridge = get_agent_bridge()
				if bridge:
					# Determine if tracking was used (if new_track_ids were provided)
					with_tracking = bool(new_track_ids)
					bridge.record_vlm_call(with_tracking=with_tracking)
			except:
				pass  # Experimental metrics is optional
			
			# Convert back to agent format
			enhanced_objects = self._convert_from_pixtral_format(enhanced_detections)
			
			return enhanced_objects
			
		except Exception as e:
			# If analysis fails, return original detections
			print(f"[WARNING] Pixtral analysis failed: {e}")
			return detections
	
	def analyze_scene(self, frame: np.ndarray, prompt_type: str = "default") -> str:
		"""
		Analyze the entire scene using Pixtral 12B model.
		Feeds the entire frame (not individual objects) into Pixtral for scene description.
		
		Args:
			frame: Input frame (numpy array) - entire frame
			prompt_type: Type of prompt to use ("default", "detailed", "simple", etc.)
		
		Returns:
			Scene description string from Pixtral
		"""
		if not self._pixtral:
			return "Scene analysis not available"
		
		if frame is None or frame.size == 0:
			return "Invalid frame"
		
		try:
			# Log VLM call start for unified logging
			vlm_start_time = time.time()
			
			# Use PixtralAnalyzer's analyze_scene method which feeds entire frame
			description = self._pixtral.analyze_scene(frame, prompt_type=prompt_type)
			
			# Log VLM call completion for unified logging
			vlm_latency = (time.time() - vlm_start_time) * 1000
			if UNIFIED_LOGGING_AVAILABLE:
				try:
					logger_integration = get_logger_integration()
					logger_integration.log_vlm_call(vlm_latency, "pixtral-scene")
				except:
					pass
			
			return description
		except Exception as e:
			print(f"[WARNING] Scene analysis failed: {e}")
			return f"Scene analysis failed: {str(e)[:100]}"

