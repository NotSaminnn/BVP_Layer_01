from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.modules.vlm.vision_chat import VisionChat  # type: ignore


class SceneAnalysisAdapter:
	def __init__(self, api_key: Optional[str] = None, verbose: bool = False) -> None:
		# VisionChat needs an API key; if not provided, we can still compute geometric relations locally
		self._vision: Optional[VisionChat] = None
		if api_key:
			self._vision = VisionChat(api_key=api_key, verbose=verbose)

	def _center(self, bbox: List[float]) -> Tuple[float, float]:
		x1, y1, x2, y2 = bbox
		return (x1 + x2) / 2.0, (y1 + y2) / 2.0

	def _relation(self, a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		ax, ay = self._center(a["bbox"]) if a.get("bbox") else (None, None)
		bx, by = self._center(b["bbox"]) if b.get("bbox") else (None, None)
		if ax is None or bx is None:
			return None
		pred: Optional[str] = None
		if ax < bx:
			pred = "leftOf"
		elif ax > bx:
			pred = "rightOf"
		# Near if centers within 15% of frame width/height (approx, absent frame we use bbox width)
		near = False
		try:
			aw = abs(a["bbox"][2] - a["bbox"][0])
			bw = abs(b["bbox"][2] - b["bbox"][0])
			threshold = 0.15 * max(aw, bw)
			near = abs(ax - bx) < threshold and abs(ay - by) < threshold
		except Exception:
			pass
		return {
			"subjectId": a.get("id") or a.get("class"),
			"predicate": "near" if near else (pred or "unknown"),
			"objectId": b.get("id") or b.get("class"),
			"confidence": 0.6 if near else 0.5,
		}

	def analyze(self, objects: List[Dict[str, Any]], query: Optional[str] = None, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
		relations: List[Dict[str, Any]] = []
		for i in range(len(objects)):
			for j in range(i + 1, len(objects)):
				rel = self._relation(objects[i], objects[j])
				if rel:
					relations.append(rel)
		salient = sorted(objects, key=lambda o: float(o.get("confidence", 0.0)), reverse=True)[:5]
		result: Dict[str, Any] = {"relations": relations, "salientRegions": [{"bbox": o.get("bbox"), "label": o.get("class"), "score": o.get("confidence")} for o in salient]}
		# Optional enrichment using VisionChat
		if self._vision and frame is not None and objects:
			try:
				# VisionChat can add descriptions to detections
				_ = self._vision.analyze_detections_with_descriptions(frame, [{
					"bbox": o.get("bbox"),
					"class_name": o.get("class"),
					"confidence": o.get("confidence", 0.0),
					"distance": o.get("approxDistance", 0.0),
					"horizontal_angle": o.get("angle_x", 0.0),
					"vertical_angle": o.get("angle_y", 0.0),
				} for o in objects])
			except Exception:
				pass
		return result
