from __future__ import annotations
import asyncio
from enum import Enum
from typing import Optional

from core.event_bus import EventBus
from core.adapters.stt import STTAdapter


class State(str, Enum):
	IDLE = "Idle"
	RECORDING = "Recording"
	PROCESSING = "Processing"
	WAITING_FOR_CONFIRMATION = "WaitingForConfirmation"
	SPEAKING = "Speaking"


class RecordingFSM:
	def __init__(self, event_bus: EventBus, stt: STTAdapter) -> None:
		self._bus = event_bus
		self._stt = stt
		self._state: State = State.IDLE
		self._current_request_id: Optional[str] = None
		self._waiting_for_clarification: bool = False  # Track if last spoken message was a clarification
		asyncio.create_task(self._subscribe())

	async def _subscribe(self) -> None:
		async def on_response_ready(_topic: str, envelope) -> None:
			if self._state == State.PROCESSING:
				self._state = State.SPEAKING
		async def on_response_spoken(_topic: str, envelope) -> None:
			# After TTS finishes
			if self._state == State.SPEAKING:
				if self._waiting_for_clarification:
					# Clarification question was spoken - wait for user response
					self._state = State.WAITING_FOR_CONFIRMATION
					# Don't reset flag yet - wait for user response
				else:
					# Regular response - return to idle
					self._state = State.IDLE
		async def on_clarification_needed(_topic: str, envelope) -> None:
			# When clarification is needed, mark that we're waiting for clarification
			self._waiting_for_clarification = True
			if self._state == State.PROCESSING:
				self._state = State.SPEAKING
		await self._bus.subscribe("ResponseReady", on_response_ready)
		await self._bus.subscribe("ResponseSpoken", on_response_spoken)
		await self._bus.subscribe("ClarificationNeeded", on_clarification_needed)
		await self._bus.subscribe("EmptyTranscriptIgnored", self.on_empty_transcript_ignored)

	def state(self) -> State:
		return self._state

	async def handle_key_r(self) -> None:
		if self._state == State.IDLE:
			self._state = State.RECORDING
			self._current_request_id = await self._stt.start_recording()
			print("[agent] Recording started... (press 'r' again to stop)")
			return
		if self._state == State.RECORDING:
			self._state = State.PROCESSING
			print("[agent] Recording stopped. Processing audio...")
			await self._stt.stop_recording()
			return
		if self._state == State.WAITING_FOR_CONFIRMATION:
			# User responding to clarification - start recording
			self._waiting_for_clarification = False  # Reset flag
			self._state = State.RECORDING
			self._current_request_id = await self._stt.start_recording()
			print("[agent] Recording started... (press 'r' again to stop)")
			return
		# Ignore presses during PROCESSING/SPEAKING to prevent re-entry

	async def on_tts_finished(self) -> None:
		self._state = State.IDLE
	
	def reset_to_idle(self) -> None:
		"""Force reset FSM state to IDLE. Used when processing is cancelled or empty transcript is ignored."""
		if self._state == State.PROCESSING:
			# If we're in PROCESSING, reset to IDLE directly
			self._state = State.IDLE
			self._waiting_for_clarification = False
			print(f"[agent] FSM reset to IDLE from PROCESSING state")
	
	async def on_empty_transcript_ignored(self, _topic: str, envelope) -> None:
		"""Handle empty transcript ignored event - reset to IDLE."""
		if self._state == State.PROCESSING:
			self._state = State.IDLE
			self._waiting_for_clarification = False
			print(f"[agent] FSM reset to IDLE after empty transcript ignored")
