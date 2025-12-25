from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

_WINDOW_NAME = "Agent Detection Monitor"


def _distance_color(distance: float) -> Tuple[int, int, int]:
	if distance < 0.5:
		return (0, 0, 255)
	elif distance < 1.0:
		return (0, 165, 255)
	elif distance < 2.0:
		return (0, 255, 255)
	elif distance < 5.0:
		return (0, 255, 0)
	return (255, 0, 0)


def draw_and_show(frame: np.ndarray, detections: List[Dict], window_name: str = _WINDOW_NAME) -> None:
	if frame is None:
		return
	img = frame.copy()
	for det in detections:
		bbox = det.get("bbox") or []
		if not bbox or len(bbox) != 4:
			continue
		x1, y1, x2, y2 = list(map(int, bbox))
		cls = str(det.get("class", "object"))
		conf = float(det.get("confidence", 0.0))
		dist = float(det.get("approxDistance", 0.0))
		ax = float(det.get("angle_x", 0.0))
		ay = float(det.get("angle_y", 0.0))
		oid = str(det.get("id") or "-")
		color = _distance_color(dist)
		cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
		# Header line 1: class/id/conf
		line1 = f"{cls}#{oid} ({conf:.2f})"
		# Header line 2: distance and angles
		line2 = f"{dist:.2f}m  H:{ax:+.1f}° V:{ay:+.1f}°"
		# Draw background rectangles for readability
		cv2.rectangle(img, (x1, max(0, y1-40)), (x1 + 260, y1), (0,0,0), -1)
		cv2.putText(img, line1, (x1+5, y1-22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
		cv2.putText(img, line2, (x1+5, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
	cv2.imshow(window_name, img)
	cv2.waitKey(1)


def close(window_name: str = _WINDOW_NAME) -> None:
	try:
		cv2.destroyWindow(window_name)
	except Exception:
		pass
