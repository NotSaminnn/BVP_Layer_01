from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional


def log(event: str, data: Optional[Dict[str, Any]] = None) -> None:
	try:
		stamp = time.strftime("%H:%M:%S")
		payload = f" {data}" if data else ""
		print(f"[agent {stamp}] {event}{payload}")
	except Exception:
		pass


@contextmanager
def timed(event: str, extra: Optional[Dict[str, Any]] = None):
	start = time.time()
	try:
		yield
	finally:
		dur_ms = int((time.time() - start) * 1000)
		log(event, {"duration_ms": dur_ms, **(extra or {})})
