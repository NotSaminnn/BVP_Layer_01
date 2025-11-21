from __future__ import annotations
from typing import Optional
import threading
import numpy as np


class FrameProvider:
	def __init__(self) -> None:
		self._lock = threading.Lock()
		self._frame: Optional[np.ndarray] = None

	def set_latest(self, frame: np.ndarray) -> None:
		with self._lock:
			self._frame = frame

	def get_latest(self) -> Optional[np.ndarray]:
		with self._lock:
			return None if self._frame is None else self._frame
