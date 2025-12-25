"""
Device Detection Utility - GPU/CPU Detection for All Models

This module provides a centralized device detection system that works across
the entire agentic system, supporting:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon GPUs)
- CPU (fallback)
- ONNX Runtime GPU providers
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Literal
import os

DeviceType = Literal["cuda", "mps", "cpu"]


class DeviceManager:
	"""
	Centralized device manager for GPU/CPU detection and management.
	"""
	
	def __init__(self, prefer_gpu: bool = True):
		"""
		Initialize device manager.
		
		Args:
			prefer_gpu: If True, prefer GPU when available; if False, force CPU
		"""
		self.prefer_gpu = prefer_gpu
		self._device: Optional[DeviceType] = None
		self._device_info: Optional[Dict[str, Any]] = None
		self._detect_device()
	
	def _detect_device(self) -> None:
		"""Detect and set the optimal device."""
		if not self.prefer_gpu:
			self._device = "cpu"
			self._device_info = {
				"type": "cpu",
				"name": "CPU (forced)",
				"available": True
			}
			return
		
		# Check CUDA availability first (NVIDIA GPUs)
		try:
			import torch
			if torch.cuda.is_available():
				self._device = "cuda"
				self._device_info = {
					"type": "cuda",
					"name": torch.cuda.get_device_name(0),
					"memory": torch.cuda.get_device_properties(0).total_memory,
					"available": True,
					"device_id": 0
				}
				return
		except ImportError:
			pass
		
		# Check MPS availability (Apple Silicon)
		try:
			import torch
			if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
				self._device = "mps"
				self._device_info = {
					"type": "mps",
					"name": "Apple Silicon GPU",
					"available": True
				}
				return
		except (ImportError, AttributeError):
			pass
		
		# Fallback to CPU
		self._device = "cpu"
		self._device_info = {
			"type": "cpu",
			"name": "CPU",
			"available": True
		}
	
	@property
	def device(self) -> DeviceType:
		"""Get the current device type."""
		return self._device or "cpu"
	
	@property
	def device_info(self) -> Dict[str, Any]:
		"""Get detailed device information."""
		return self._device_info or {"type": "cpu", "name": "CPU", "available": True}
	
	def is_gpu_available(self) -> bool:
		"""Check if GPU is available."""
		return self.device in ("cuda", "mps")
	
	def get_torch_device(self) -> str:
		"""Get PyTorch device string."""
		return self.device
	
	def get_onnx_runtime_providers(self) -> list[str]:
		"""
		Get ONNX Runtime execution providers in priority order.
		
		Returns:
			List of execution provider names
		"""
		providers = []
		
		if self.device == "cuda":
			# Try CUDA Execution Provider first
			try:
				import onnxruntime as ort
				if 'CUDAExecutionProvider' in ort.get_available_providers():
					providers.append('CUDAExecutionProvider')
			except ImportError:
				pass
		
		# Always add CPU provider as fallback
		providers.append('CPUExecutionProvider')
		
		return providers
	
	def print_device_info(self) -> None:
		"""Print device information."""
		info = self.device_info
		device_type = info.get("type", "unknown").upper()
		device_name = info.get("name", "Unknown")
		
		print(f"[DeviceManager] Device: {device_type}")
		print(f"[DeviceManager] Name: {device_name}")
		
		if device_type == "CUDA":
			memory = info.get("memory", 0)
			if memory:
				memory_gb = memory / (1024**3)
				print(f"[DeviceManager] GPU Memory: {memory_gb:.2f} GB")
		
		if self.is_gpu_available():
			print(f"[DeviceManager] ✅ GPU acceleration enabled")
		else:
			print(f"[DeviceManager] ℹ️  Using CPU (GPU not available)")


# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager(force_cpu: bool = False) -> DeviceManager:
	"""
	Get or create the global device manager instance.
	
	Args:
		force_cpu: If True, force CPU usage even if GPU is available
		
	Returns:
		DeviceManager instance
	"""
	global _global_device_manager
	
	# Check environment variable for forced CPU
	env_force_cpu = os.environ.get("FORCE_CPU", "false").lower() in ("true", "1", "yes")
	force_cpu = force_cpu or env_force_cpu
	
	if _global_device_manager is None:
		_global_device_manager = DeviceManager(prefer_gpu=not force_cpu)
	elif force_cpu:
		# Recreate if forcing CPU
		_global_device_manager = DeviceManager(prefer_gpu=False)
	
	return _global_device_manager


def get_device() -> DeviceType:
	"""Get the current device type."""
	return get_device_manager().device


def get_device_info() -> Dict[str, Any]:
	"""Get device information."""
	return get_device_manager().device_info


def is_gpu_available() -> bool:
	"""Check if GPU is available."""
	return get_device_manager().is_gpu_available()


def get_torch_device() -> str:
	"""Get PyTorch device string."""
	return get_device_manager().get_torch_device()


def get_onnx_runtime_providers() -> list[str]:
	"""Get ONNX Runtime execution providers."""
	return get_device_manager().get_onnx_runtime_providers()


def print_device_info() -> None:
	"""Print device information."""
	get_device_manager().print_device_info()

