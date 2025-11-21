from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import numpy as np

from core.event_bus import EventBus
from core.infrastructure.device_manager import get_device_manager, print_device_info  # type: ignore

# Import detector without modifying original code
from core.modules.object_detection.main import (
	ObjectDetectionWithDistanceAngle,
)  # type: ignore


class ObjectDetectorAdapter:
	def __init__(self, model_path: Optional[str] = None, confidence: Optional[float] = None, 
	             force_cpu: bool = False) -> None:
		"""
		Initialize object detector adapter.
		
		Args:
			model_path: Path to YOLO model file (auto-detected if None)
			confidence: Confidence threshold (default: 0.25)
			force_cpu: If True, force CPU usage even if GPU is available
		"""
		# Initialize device manager and print info
		device_mgr = get_device_manager(force_cpu=force_cpu)
		print_device_info()
		
		resolved = model_path or self._resolve_model_path()
		self._detector = ObjectDetectionWithDistanceAngle(
			model_path=resolved,
			confidence_threshold=confidence or 0.25,
		)

	def _resolve_model_path(self) -> str:
		"""Resolve model path robustly across invocation locations."""
		# Known filenames in the detection package
		candidates = [
			"yolo11n_object365.onnx",
			"yolo11n_object365.pt",
			"object_detection_with_distance_and_angle_mapping/yolo11n_object365.onnx",
			"object_detection_with_distance_and_angle_mapping/yolo11n_object365.pt",
		]
		# Start from detection module directory
		pkg_dir = os.path.dirname(os.path.abspath(__file__))
		root = os.path.abspath(os.path.join(pkg_dir, ".."))  # BVP_LAYER01
		det_dir = os.path.join(root, "object_detection_with_distance_and_angle_mapping")
		search_dirs = [det_dir, root, os.getcwd()]
		for d in search_dirs:
			for name in candidates:
				p = os.path.join(d, os.path.basename(name) if os.path.isdir(d) and os.path.basename(d) == os.path.basename(det_dir) else name)
				if os.path.exists(p):
					return os.path.abspath(p)
		# Fallback to expected location under detection package
		fallback = os.path.join(det_dir, "yolo11n_object365.onnx")
		if os.path.exists(fallback):
			return os.path.abspath(fallback)
		raise FileNotFoundError("Could not find model file yolo11n_object365.(onnx|pt). Place it under object_detection_with_distance_and_angle_mapping/")

	def _normalize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
		bbox = raw.get("bbox")
		if hasattr(bbox, "tolist"):
			bbox = bbox.tolist()
		return {
			"id": str(raw.get("track_id") or ""),
			"class": raw.get("class_name") or raw.get("class") or "unknown",
			"bbox": bbox,
			"confidence": float(raw.get("confidence", 0.0)),
			"approxDistance": float(raw.get("distance", 0.0)),
			"angle_x": float(raw.get("angle_x", 0.0)),
			"angle_y": float(raw.get("angle_y", 0.0)),
			"timestamp": None,
		}

	def detect(self, frame: np.ndarray, target_classes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
		result = self._detector.process_single_frame(frame, enable_tracking=True)
		detections: List[Dict[str, Any]] = result.get("detections", [])
		if target_classes:
			target_set = {c.lower() for c in target_classes}
			detections = [d for d in detections if (d.get("class_name", "").lower() in target_set)]
		return [self._normalize(d) for d in detections]
	
	def detect_with_tracking(self, frame: np.ndarray, target_classes: Optional[List[str]] = None) -> Dict[str, Any]:
		"""
		Detect objects with tracking info (for Pixtral analysis).
		Returns dict with objects, new_track_ids, and current_track_ids.
		"""
		result = self._detector.process_single_frame(frame, enable_tracking=True)
		detections: List[Dict[str, Any]] = result.get("detections", [])
		new_track_ids = result.get("new_track_ids", set())
		current_track_ids = result.get("current_track_ids", set())
		
		if target_classes:
			target_set = {c.lower() for c in target_classes}
			detections = [d for d in detections if (d.get("class_name", "").lower() in target_set)]
		
		return {
			"objects": [self._normalize(d) for d in detections],
			"new_track_ids": new_track_ids,
			"current_track_ids": current_track_ids,
			"tracking_enabled": True
		}
