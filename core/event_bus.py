import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


class CancellationToken:
	def __init__(self) -> None:
		self._event = asyncio.Event()

	def cancel(self) -> None:
		self._event.set()

	@property
	def is_cancelled(self) -> bool:
		return self._event.is_set()

	async def wait(self) -> None:
		await self._event.wait()


Subscriber = Callable[[str, Any], Awaitable[None]]


class EventBus:
	"""
	Simple in-process async pub/sub with per-topic subscribers.
	Each publish carries a request_id for tracing; if not provided, one is generated.
	Supports cancellation tokens for cooperative cancellation.
	"""

	def __init__(self) -> None:
		self._subscribers: Dict[str, List[Subscriber]] = {}
		self._lock = asyncio.Lock()

	async def subscribe(self, topic: str, handler: Subscriber) -> None:
		async with self._lock:
			self._subscribers.setdefault(topic, []).append(handler)

	async def unsubscribe(self, topic: str, handler: Subscriber) -> None:
		async with self._lock:
			if topic in self._subscribers:
				self._subscribers[topic] = [h for h in self._subscribers[topic] if h != handler]
				if not self._subscribers[topic]:
					self._subscribers.pop(topic, None)

	async def publish(self, topic: str, payload: Any, request_id: Optional[str] = None) -> str:
		request_id = request_id or str(uuid.uuid4())
		async with self._lock:
			handlers = list(self._subscribers.get(topic, []))
		# Dispatch without holding the lock
		await asyncio.gather(*(h(topic, {"request_id": request_id, "payload": payload, "ts": time.time()}) for h in handlers))
		return request_id

	def has_subscribers(self, topic: str) -> bool:
		return topic in self._subscribers and len(self._subscribers[topic]) > 0
