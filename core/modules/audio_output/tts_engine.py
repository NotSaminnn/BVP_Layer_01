import asyncio
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import tempfile


@dataclass
class TTSConfig:
	voice: str = "en-US-AriaNeural"  # soothing, natural voice
	rate: str = "+0%"  # speaking rate
	pitch: str = "+0Hz"  # pitch adjustment
	volume: str = "+0%"
	save_format: str = "audio-24khz-48kbitrate-mono-mp3"  # compact mp3 when saving


class TextToSpeech:
	def __init__(self, config: Optional[TTSConfig] = None) -> None:
		self.config = config or TTSConfig()

	async def synthesize_to_file_async(self, text: str, output_path: str) -> str:
		"""Synthesize text to an audio file (mp3)."""
		from edge_tts import Communicate
		Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		communicate = Communicate(
			text,
			voice=self.config.voice,
			rate=self.config.rate,
			pitch=self.config.pitch,
			volume=self.config.volume,
		)
		await communicate.save(output_path)
		return output_path

	def synthesize_to_file(self, text: str, output_path: str) -> str:
		"""Sync wrapper to synthesize text to file."""
		return asyncio.run(self.synthesize_to_file_async(text, output_path))

	async def speak_async(self, text: str) -> None:
		"""Speak text directly using a fresh system TTS engine per call (pyttsx3/SAPI5)."""
		import pyttsx3
		engine = pyttsx3.init()
		try:
			rate = engine.getProperty('rate')
			engine.setProperty('rate', max(120, int(rate * 0.9)))
			engine.setProperty('volume', 0.9)
			try:
				voices = engine.getProperty('voices')
				preferred = None
				for v in voices:
					name = (v.name or '').lower()
					if 'zira' in name or 'aria' in name or 'hazel' in name or 'female' in name:
						preferred = v.id
						break
				if preferred:
					engine.setProperty('voice', preferred)
			except Exception:
				pass
			engine.say(text)
			engine.runAndWait()
		finally:
			try:
				engine.stop()
			except Exception:
				pass

	def speak(self, text: str) -> None:
		"""Sync wrapper to speak text without creating a persistent file."""
		asyncio.run(self.speak_async(text))


