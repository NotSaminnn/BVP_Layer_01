import threading
import time
import queue
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass
class RecordingConfig:
	sample_rate: int = 16000
	channels: int = 1
	dtype: str = "float32"  # sounddevice dtype
	blocksize: int = 0  # 0 lets sounddevice choose optimal


class AudioRecorder:
	"""Simple audio recorder using sounddevice with a background buffer thread."""

	def __init__(self, config: Optional[RecordingConfig] = None) -> None:
		self.config = config or RecordingConfig()
		self._stream: Optional[sd.InputStream] = None
		self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
		self._frames: List[np.ndarray] = []
		self._collector_thread: Optional[threading.Thread] = None
		self._collecting: bool = False
		self._lock = threading.Lock()

	def _audio_callback(self, indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
		if status:  # status contains warnings/errors from PortAudio
			# Avoid printing repeatedly in callback; could add logging here
			pass
		self._queue.put(indata.copy())

	def _collector_loop(self) -> None:
		while self._collecting:
			try:
				chunk = self._queue.get(timeout=0.1)
				with self._lock:
					self._frames.append(chunk)
			except queue.Empty:
				continue

	def start(self) -> None:
		if self._stream is not None:
			raise RuntimeError("Recorder already started")
		self._frames = []
		self._collecting = True
		self._stream = sd.InputStream(
			samplerate=self.config.sample_rate,
			channels=self.config.channels,
			dtype=self.config.dtype,
			blocksize=self.config.blocksize,
			callback=self._audio_callback,
		)
		self._stream.start()
		self._collector_thread = threading.Thread(target=self._collector_loop, daemon=True)
		self._collector_thread.start()

	def stop(self) -> np.ndarray:
		if self._stream is None:
			raise RuntimeError("Recorder not started")
		self._collecting = False
		self._stream.stop()
		self._stream.close()
		self._stream = None
		if self._collector_thread is not None:
			self._collector_thread.join(timeout=2.0)
			self._collector_thread = None
		with self._lock:
			if not self._frames:
				return np.zeros((0, self.config.channels), dtype=np.float32)
			audio = np.concatenate(self._frames, axis=0)
		return audio

	def save_wav(self, filepath: str, audio_data: np.ndarray) -> str:
		"""Save audio data to a WAV file using soundfile."""
		if audio_data.ndim == 1:
			audio_to_save = audio_data
		else:
			audio_to_save = audio_data
		sf.write(filepath, audio_to_save, self.config.sample_rate)
		return filepath


