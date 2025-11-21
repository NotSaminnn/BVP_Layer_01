from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time


@dataclass
class BaseEvent:
	request_id: str
	ts: float = field(default_factory=lambda: time.time())


@dataclass
class AudioCaptured(BaseEvent):
	audio_path: Optional[str] = None
	audio_bytes: Optional[bytes] = None
	sample_rate: Optional[int] = None
	duration_sec: Optional[float] = None


@dataclass
class TranscriptReady(BaseEvent):
	transcript: str = ""
	confidence: Optional[float] = None


@dataclass
class PlanReady(BaseEvent):
	plan: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult(BaseEvent):
	tool_name: str = ""
	status: str = "success"  # success | error | partial
	data: Dict[str, Any] = field(default_factory=dict)
	confidence: Optional[float] = None
	evidence: List[Dict[str, Any]] = field(default_factory=list)
	errors: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ResponseReady(BaseEvent):
	answer: str = ""


@dataclass
class ClarificationNeeded(BaseEvent):
	question: str = ""
	original_query: str = ""
	ambiguity_reason: str = ""


@dataclass
class ClarificationReceived(BaseEvent):
	user_response: str = ""
	original_query: str = ""
	confirmed_query: Optional[str] = None


@dataclass
class ErrorEvent(BaseEvent):
	code: str = ""
	message: str = ""
	context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmptyTranscriptIgnored(BaseEvent):
	"""Event published when an empty transcript is ignored to reset FSM state."""
	pass
