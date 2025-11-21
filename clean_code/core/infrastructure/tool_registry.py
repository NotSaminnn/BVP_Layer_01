from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
	import yaml  # type: ignore
except Exception:  # pragma: no cover
	yaml = None


@dataclass
class ToolSpec:
	name: str
	description: str
	inputs: Dict[str, Any]
	outputs: Dict[str, Any]
	dependencies: List[str]
	timeout_ms: Optional[int] = None
	min_confidence: Optional[float] = None


class ToolRegistry:
	def __init__(self) -> None:
		self._tools: Dict[str, ToolSpec] = {}

	def load_from_file(self, path: str) -> None:
		if path.endswith((".yaml", ".yml")):
			if yaml is None:
				raise RuntimeError("PyYAML is required to load YAML configs")
			with open(path, "r", encoding="utf-8") as f:
				data = yaml.safe_load(f)
		elif path.endswith(".json"):
			with open(path, "r", encoding="utf-8") as f:
				data = json.load(f)
		else:
			raise ValueError("Unsupported config format; use .yaml/.yml or .json")
		self._from_dict(data)

	def _from_dict(self, data: Dict[str, Any]) -> None:
		tools = data.get("tools")
		if not isinstance(tools, list):
			raise ValueError("Invalid config: 'tools' must be a list")
		seen: set[str] = set()
		loaded: Dict[str, ToolSpec] = {}
		for t in tools:
			name = t.get("name")
			desc = t.get("description", "")
			inputs = t.get("inputs", {})
			outputs = t.get("outputs", {})
			deps = t.get("dependencies", [])
			timeout_ms = t.get("timeout_ms")
			min_conf = t.get("min_confidence")
			if not name or not isinstance(name, str):
				raise ValueError("Each tool must have a string 'name'")
			if name in seen:
				raise ValueError(f"Duplicate tool name: {name}")
			if not isinstance(inputs, dict) or not isinstance(outputs, dict):
				raise ValueError(f"Tool {name} must have dict 'inputs' and 'outputs'")
			if not isinstance(deps, list):
				raise ValueError(f"Tool {name} 'dependencies' must be a list")
			loaded[name] = ToolSpec(
				name=name,
				description=desc,
				inputs=inputs,
				outputs=outputs,
				dependencies=list(deps),
				timeout_ms=timeout_ms,
				min_confidence=min_conf,
			)
			seen.add(name)
		# Validate dependencies
		for spec in loaded.values():
			for dep in spec.dependencies:
				if dep not in loaded:
					raise ValueError(f"Tool {spec.name} depends on unknown tool '{dep}'")
		self._tools = loaded

	def get(self, name: str) -> Optional[ToolSpec]:
		return self._tools.get(name)

	def list_tools(self) -> List[ToolSpec]:
		return list(self._tools.values())

	def require(self, name: str) -> ToolSpec:
		spec = self.get(name)
		if not spec:
			raise KeyError(f"Tool not found: {name}")
		return spec
