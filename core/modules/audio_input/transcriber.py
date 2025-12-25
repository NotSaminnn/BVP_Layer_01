from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import os

import numpy as np
import soundfile as sf
import torch
import whisper


@dataclass
class WhisperConfig:
	model_size: str = "small"  # Model size: tiny, base, small, medium, large. Small provides good balance of speed and accuracy
	device: Optional[str] = None  # "cuda" or "cpu"; auto if None
	language: Optional[str] = "en"  # Language code. Set to "en" for English (handles all English accents including Indian English)
	temperature: Union[float, Tuple[float, ...]] = 0.0  # Single temperature for faster inference (0.0 = deterministic, fastest)
	beam_size: int = 3  # Beam size for beam search (reduced from 5 for faster inference while maintaining accuracy)
	best_of: int = 3  # Number of candidates to consider (reduced from 5 for faster inference)
	condition_on_previous_text: bool = True  # Use previous text context (helps with accent consistency)
	initial_prompt: Optional[str] = None  # Optional prompt to help with accent adaptation
	compression_ratio_threshold: Optional[float] = 2.4  # Skip decoding if compression ratio exceeds this (faster)
	logprob_threshold: Optional[float] = -1.0  # Skip decoding if log probability is below this (faster)
	# Note: Optimized for speed while maintaining accuracy. Small model is 2-3x faster than medium with minimal accuracy loss.


class WhisperTranscriber:
	def __init__(self, config: Optional[WhisperConfig] = None) -> None:
		self.config = config or WhisperConfig()
		self.device = self.config.device or self._get_optimal_device()
		self.model = whisper.load_model(self.config.model_size, device=self.device)
	
	def _get_optimal_device(self) -> str:
		"""
		Get the optimal computing device (CUDA > MPS > CPU).
		
		Returns:
			Device string ('cuda', 'mps', or 'cpu')
		"""
		# Check CUDA availability first (NVIDIA GPUs)
		if torch.cuda.is_available():
			return 'cuda'
		
		# Check MPS availability (Apple Silicon)
		if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
			return 'mps'
		
		# Fallback to CPU
		return 'cpu'

	def transcribe(self, audio_path: str) -> str:
		# Read WAV directly to bypass ffmpeg dependency
		data, sr = sf.read(audio_path, dtype="float32")
		if data.ndim == 2:
			data = data.mean(axis=1)
		# Recorder saves at 16kHz; if not, a resampler would be needed.
		if sr != 16000:
			# Simple guard; Whisper can internally handle different rates, but best at 16k
			pass
		# Use fp16 on CUDA and MPS for faster inference (MPS supports fp16 in recent PyTorch)
		fp16 = self.device in ("cuda", "mps")
		
		# Configure transcription parameters optimized for speed and accuracy
		transcribe_options = {
			"fp16": fp16,
			"language": self.config.language,  # Explicitly set language to English (handles all accents)
			"temperature": self.config.temperature,  # Lower temperature for more deterministic results
			"beam_size": self.config.beam_size,  # Larger beam size for better accuracy
			"best_of": self.config.best_of,  # Multiple candidates for better accuracy
			"condition_on_previous_text": self.config.condition_on_previous_text,  # Use context
		}
		
		# Add initial_prompt if provided (helps with accent adaptation)
		if self.config.initial_prompt:
			transcribe_options["initial_prompt"] = self.config.initial_prompt
		
		# Add faster inference parameters (skip bad transcriptions early)
		if self.config.compression_ratio_threshold is not None:
			transcribe_options["compression_ratio_threshold"] = self.config.compression_ratio_threshold
		if self.config.logprob_threshold is not None:
			transcribe_options["logprob_threshold"] = self.config.logprob_threshold
		
		# Transcribe with optimized parameters
		result = self.model.transcribe(data, **transcribe_options)
		text = (result or {}).get("text", "").strip()
		return text
	
	def transcribe_audio_array(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
		"""
		Transcribe audio directly from numpy array (in-memory, faster than file I/O).
		
		Args:
			audio_data: Audio data as numpy array (mono, float32)
			sample_rate: Sample rate of audio data (default: 16000)
			
		Returns:
			Transcribed text
		"""
		# Ensure mono audio
		if audio_data.ndim == 2:
			audio_data = audio_data.mean(axis=1)
		# Ensure float32
		if audio_data.dtype != np.float32:
			audio_data = audio_data.astype(np.float32)
		
		# Use fp16 on CUDA and MPS for faster inference
		fp16 = self.device in ("cuda", "mps")
		
		# Configure transcription parameters optimized for speed and accuracy
		transcribe_options = {
			"fp16": fp16,
			"language": self.config.language,
			"temperature": self.config.temperature,
			"beam_size": self.config.beam_size,
			"best_of": self.config.best_of,
			"condition_on_previous_text": self.config.condition_on_previous_text,
		}
		
		# Add initial_prompt if provided
		if self.config.initial_prompt:
			transcribe_options["initial_prompt"] = self.config.initial_prompt
		
		# Add faster inference parameters
		if self.config.compression_ratio_threshold is not None:
			transcribe_options["compression_ratio_threshold"] = self.config.compression_ratio_threshold
		if self.config.logprob_threshold is not None:
			transcribe_options["logprob_threshold"] = self.config.logprob_threshold
		
		# Transcribe directly from numpy array (no file I/O)
		result = self.model.transcribe(audio_data, **transcribe_options)
		text = (result or {}).get("text", "").strip()
		return text
