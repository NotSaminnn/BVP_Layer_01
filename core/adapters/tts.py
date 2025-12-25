from __future__ import annotations
import asyncio
from typing import Optional

from core.event_bus import EventBus
from core. import schemas

# Use existing TTS implementation
from core.modules.audio_output.tts_engine import TextToSpeech, TTSConfig  # type: ignore


class TTSAdapter:
	def __init__(self, event_bus: EventBus, subscribe_to_events: bool = True) -> None:
		self._event_bus = event_bus
		self._engine = TextToSpeech(TTSConfig())
		self._current_task: Optional[asyncio.Task] = None
		if subscribe_to_events:
			asyncio.create_task(self._subscribe())

	async def _subscribe(self) -> None:
		async def on_response_ready(_topic: str, envelope) -> None:
			payload: schemas.ResponseReady = envelope["payload"]
			request_id = envelope["request_id"]
			# Spawn background task so we don't block the event loop
			asyncio.create_task(self.speak(payload.answer, request_id=request_id))
		
		async def on_clarification_needed(_topic: str, envelope) -> None:
			payload: schemas.ClarificationNeeded = envelope["payload"]
			request_id = envelope["request_id"]
			# Speak the clarification question (will emit ResponseSpoken when done)
			asyncio.create_task(self.speak(payload.question, request_id=request_id))
		
		await self._event_bus.subscribe("ResponseReady", on_response_ready)
		await self._event_bus.subscribe("ClarificationNeeded", on_clarification_needed)

	async def speak(self, text: str, request_id: Optional[str] = None) -> None:
		# Cancel any ongoing speech
		if self._current_task and not self._current_task.done():
			self._current_task.cancel()
			try:
				await self._current_task
			except Exception:
				pass

		async def _do_speak_blocking() -> None:
			# Run blocking TTS in a thread to avoid freezing the event loop
			from core.modules.audio_output.tts_engine import TextToSpeech as _TTS  # type: ignore
			await asyncio.to_thread(self._engine.speak, text)
			if request_id:
				await self._event_bus.publish("ResponseSpoken", schemas.BaseEvent(request_id=request_id), request_id=request_id)

		self._current_task = asyncio.create_task(_do_speak_blocking())

	async def stop(self) -> None:
		if self._current_task and not self._current_task.done():
			self._current_task.cancel()
			try:
				await self._current_task
			except Exception:
				pass
