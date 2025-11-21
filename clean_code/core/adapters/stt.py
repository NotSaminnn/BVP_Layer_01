from __future__ import annotations
import asyncio
import os
import uuid
import time
from typing import Optional, Tuple

from core.event_bus import EventBus
from core. import schemas

# Import existing modules without modification
from core.modules.audio_input import recorder as whisper_recorder  # type: ignore
from core.modules.audio_input import transcriber as whisper_transcriber  # type: ignore


class STTAdapter:
	"""
	Wraps the existing whisper recorder/transcriber to expose a uniform interface
	and emit Agent events: AudioCaptured -> TranscriptReady.
	"""

	def __init__(self, event_bus: EventBus, recordings_dir: Optional[str] = None, 
	             accent_optimization: bool = True) -> None:
		"""
		Initialize STT Adapter.
		
		Args:
			event_bus: Event bus for publishing events
			recordings_dir: Directory to save recordings (optional)
			accent_optimization: If True, optimize for accent recognition (Indian English, etc.)
		"""
		self._event_bus = event_bus
		self._recordings_dir = recordings_dir or os.path.join(os.getcwd(), "BVP_LAYER01", "audio_transcription_whisper", "recordings")
		os.makedirs(self._recordings_dir, exist_ok=True)
		self._recorder = whisper_recorder.AudioRecorder(whisper_recorder.RecordingConfig())
		
		# Configure Whisper for optimized speed and accuracy
		if accent_optimization:
			# Use small model for good balance of speed and accuracy (2-3x faster than medium)
			# Optimized parameters for fast inference while maintaining accent recognition
			whisper_config = whisper_transcriber.WhisperConfig(
				model_size="small",  # Small model: 2-3x faster than medium with minimal accuracy loss
				language="en",  # English (handles all English accents)
				temperature=0.0,  # Single temperature for faster inference (deterministic)
				beam_size=3,  # Reduced beam size for faster inference (still accurate)
				best_of=3,  # Reduced best_of for faster inference
				condition_on_previous_text=True,  # Context awareness (helps with accents)
				initial_prompt="This is a conversation in English. The speaker may have an Indian English accent or other non-native accent variations. Please transcribe accurately regardless of accent.",
				compression_ratio_threshold=2.4,  # Skip bad transcriptions early (faster)
				logprob_threshold=-1.0  # Skip low-probability transcriptions early (faster)
			)
			self._transcriber = whisper_transcriber.WhisperTranscriber(config=whisper_config)
			print("[STT] Whisper configured for optimized speed and accuracy (Indian English, etc.)")
		else:
			# Default configuration (faster but less accurate for accents)
			self._transcriber = whisper_transcriber.WhisperTranscriber()
		
		self._current_request_id: Optional[str] = None

	async def start_recording(self) -> str:
		if self._current_request_id is not None:
			raise RuntimeError("Recording already in progress")
		request_id = str(uuid.uuid4())
		self._current_request_id = request_id
		# Start recording (non-async)
		self._recorder.start()
		return request_id

	async def stop_recording(self) -> None:
		if self._current_request_id is None:
			return
		request_id = self._current_request_id
		self._current_request_id = None
		# Stop and get audio (numpy array)
		audio = self._recorder.stop()
		
		# Check if audio is empty
		if audio.size == 0:
			print("[agent] No audio captured, skipping transcription")
			ready = schemas.TranscriptReady(request_id=request_id, transcript="", confidence=None)
			await self._event_bus.publish("TranscriptReady", ready, request_id=request_id)
			return
		
		# Calculate duration for logging
		sample_rate = self._recorder.config.sample_rate
		duration_sec = len(audio) / sample_rate if audio.ndim == 1 else len(audio) / sample_rate
		
		# Save to wav (for backup/debugging, but use in-memory transcription for speed)
		timestamp = int(time.time())
		file_path = os.path.join(self._recordings_dir, f"rec_{timestamp}.wav")
		self._recorder.save_wav(file_path, audio)
		print(f"[agent] Recording received ({duration_sec:.2f}s) and saved: {file_path}")
		
		# Publish AudioCaptured
		event = schemas.AudioCaptured(request_id=request_id, audio_path=file_path, sample_rate=sample_rate, duration_sec=duration_sec)
		await self._event_bus.publish("AudioCaptured", event, request_id=request_id)
		
		# Transcribe in thread to avoid blocking loop (using in-memory transcription for faster processing)
		print("[agent] Transcribing audio (optimized mode)...")
		def _do_transcribe(audio_data, sr) -> Tuple[str, Optional[float]]:
			# Use in-memory transcription (faster than file I/O)
			if hasattr(self._transcriber, 'transcribe_audio_array'):
				text = self._transcriber.transcribe_audio_array(audio_data, sr)
			else:
				# Fallback to file-based transcription
				text = self._transcriber.transcribe(file_path)
			return text, None
		
		# Ensure audio is in correct format for transcription
		if audio.ndim == 2:
			audio_mono = audio.mean(axis=1)
		else:
			audio_mono = audio
		
		transcript, conf = await asyncio.to_thread(_do_transcribe, audio_mono, sample_rate)
		print(f"[agent] Transcript ready: \"{transcript}\"")
		ready = schemas.TranscriptReady(request_id=request_id, transcript=transcript, confidence=conf)
		await self._event_bus.publish("TranscriptReady", ready, request_id=request_id)

	# Optional convenience methods
	async def transcribe(self, audio_bytes: bytes) -> Tuple[str, Optional[float]]:
		# Not used in current flow; kept for interface completeness
		return "", None

	async def transcribe_file(self, path: str) -> Tuple[str, Optional[float]]:
		text = await asyncio.to_thread(self._transcriber.transcribe, path)
		return text, None
