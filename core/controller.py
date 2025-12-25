from __future__ import annotations
import asyncio
import json
import re
import time
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple

from core.event_bus import EventBus
from core. import schemas
from core.infrastructure.tool_registry import ToolRegistry
from core.adapters.object_detector import ObjectDetectorAdapter
from core.adapters.scene_analysis import SceneAnalysisAdapter
from core.adapters.chatbot import ChatbotAdapter
from core.infrastructure.location_formatter import format_location_response, angles_to_direction, _fuzzy_match_class

# Import unified logging integration
try:
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from logger_integration import get_logger_integration
	UNIFIED_LOGGING_AVAILABLE = True
except ImportError:
	UNIFIED_LOGGING_AVAILABLE = False

# Reuse existing visualization util from the detection package
def _try_draw_and_show(frame, detections):
	try:
		import cv2
		from core.modules.object_detection.utils import draw_simple_detection  # type: ignore
		img = frame.copy()
		for d in detections:
			bbox = d.get("bbox") or []
			if not bbox or len(bbox) != 4:
				continue
			cls = d.get("class", "object")
			conf = float(d.get("confidence", 0.0))
			dist = float(d.get("approxDistance", 0.0))
			ax = float(d.get("angle_x", 0.0))
			ay = float(d.get("angle_y", 0.0))
			# Draw using existing util
			img = draw_simple_detection(img, bbox, str(cls), conf, dist, ax, ay)
		cv2.imshow("Agent Detection Monitor", img)
		cv2.waitKey(1)
	except Exception:
		pass

try:
	from core.modules.vlm.mistral_chat import MistralChat  # type: ignore
except Exception:
	MistralChat = None  # type: ignore


@dataclass
class PlanStep:
	id: str
	tool: str
	params: Dict[str, Any] = field(default_factory=dict)
	deps: List[str] = field(default_factory=list)


@dataclass
class Plan:
	id: str
	steps: List[PlanStep] = field(default_factory=list)


class ClarificationChecker:
	"""Checks queries for ambiguity and generates clarification questions. Uses fast heuristics before LLM call."""
	
	def __init__(self, api_key: Optional[str] = None, enable_llm: bool = True) -> None:
		self._api_key = api_key or os.environ.get("MISTRAL_API_KEY") or ""
		self._llm: Optional[MistralChat] = None
		self._enable_llm = enable_llm
		# Common ambiguous patterns (fast heuristic check)
		self._ambiguous_patterns = [
			r"\bit\b",  # "where is it" without context
			r"\bthe thing\b", r"\bthat\b", r"\bthis\b",  # Vague references
		]
		# Common ambiguous object pairs (similar-sounding words)
		self._ambiguous_pairs = {
			"pins": ["pens", "pins"],
			"pens": ["pins", "pens"],
			"pin": ["pen", "pin"],
			"pen": ["pin", "pen"],
		}
		if self._enable_llm and self._api_key and MistralChat:
			try:
				self._llm = MistralChat(api_key=self._api_key, verbose=False, max_tokens=50)
			except Exception:
				pass
	
	def _is_object_finding_query(self, query: str) -> bool:
		"""Check if query is about finding/searching for objects. Uses simple keyword matching (not strict regex)."""
		query_lower = query.lower().strip()
		# Simple keyword-based check - let LLM handle the nuances
		object_finding_keywords = ["find", "where", "locate", "search", "see", "show", "look", "detect", "spot"]
		return any(keyword in query_lower for keyword in object_finding_keywords)
	
	def _extract_object_name(self, query: str) -> Optional[str]:
		"""Extract object name from query. Uses simple heuristics, but LLM should handle most cases."""
		query_lower = query.lower().strip()
		# Articles and common words to skip
		skip_words = {'the', 'a', 'an', 'some', 'any', 'these', 'this', 'that', 'those', 'my', 'your', 'me', 'on', 'in', 'at', 'with', 'for', 'to'}
		
		# Very simple extraction - just find first meaningful word after common verbs
		# This is a fallback; LLM should handle most cases
		verbs = ["find", "where", "show", "look", "see", "search", "locate", "detect"]
		for verb in verbs:
			if verb in query_lower:
				# Try to find word after verb
				parts = query_lower.split(verb, 1)
				if len(parts) > 1:
					remaining = parts[1].strip()
					words = remaining.split()
					for word in words:
						word_clean = word.strip('.,?!').lower()
						if word_clean and len(word_clean) > 2 and word_clean not in skip_words:
							return word_clean
		return None
	
	def _fast_heuristic_check(self, query: str) -> Optional[Dict[str, Any]]:
		"""Fast heuristic check for obvious ambiguities - returns immediately without LLM call."""
		query_lower = query.lower().strip()
		
		# Only check object-finding queries
		if not self._is_object_finding_query(query):
			return None
		
		# Extract object name
		object_name = self._extract_object_name(query)
		if not object_name:
			# Check for vague references
			if re.search(r"\b(where|what|find|show).*(it|that|this|the thing)\b", query_lower):
				return {"question": "What object are you looking for?", "reason": "Vague reference without object name"}
			return None
		
		# Check for ambiguous object pairs (pins/pens)
		if object_name in self._ambiguous_pairs:
			alternatives = self._ambiguous_pairs[object_name]
			# Ask for clarification
			if object_name == "pins" or object_name == "pin":
				return {"question": "Do you mean pens?", "reason": "similar-sounding word"}
			elif object_name == "pens" or object_name == "pen":
				return {"question": "Do you mean pins?", "reason": "similar-sounding word"}
		
		# Check for vague references
		if re.search(r"\b(where|what|find|show).*(it|that|this|the thing)\b", query_lower):
			if not re.search(r"\b(pen|bottle|chair|sofa|table|book|phone|laptop)\b", query_lower):
				return {"question": "What object are you looking for?", "reason": "Vague reference without object name"}
		
		return None  # Pass through to LLM or skip
	
	def _is_specific_detail_query(self, query: str) -> bool:
		"""Check if query is asking for specific details about an object (not location)."""
		query_lower = query.lower().strip()
		# Queries asking for object details/descriptions should NOT trigger clarification
		detail_keywords = [
			"what kind", "what type", "how does", "how do", "describe", 
			"what does it look", "what does that look", "what is", "tell me about",
			"explain", "details", "specifications", "features"
		]
		return any(keyword in query_lower for keyword in detail_keywords)
	
	def _has_clear_object_name(self, query: str) -> bool:
		"""Check if query has a clear, specific object name (not vague)."""
		query_lower = query.lower().strip()
		
		# Common object names that should be recognized
		common_objects = [
			"cup", "table", "chair", "bottle", "pen", "phone", "laptop",
			"book", "desk", "monitor", "screen", "keyboard", "mouse",
			"bag", "backpack", "wallet", "keys", "remote", "tv", "television",
			"sofa", "couch", "bed", "door", "window", "light", "lamp",
			"person", "people", "car", "vehicle", "bike", "bicycle",
			"plate", "bowl", "spoon", "fork", "knife", "glass", "mug",
			"computer", "printer", "scanner", "camera", "headphones",
			"watch", "clock", "calendar", "notebook", "paper", "document"
		]
		
		# First check if there's a clear object name
		has_object = any(obj in query_lower for obj in common_objects)
		
		# If we have a clear object name, it's clear (even if there are vague words like "that")
		if has_object:
			return True
		
		# If no clear object name, check for truly vague references
		# These patterns indicate vague references without object names
		vague_only_patterns = [
			r"where\s+is\s+it\b",  # "where is it" (standalone)
			r"where\s+is\s+that\s+thing\b",  # "where is that thing"
			r"where\s+is\s+this\s+thing\b",  # "where is this thing"
			r"find\s+it\b",  # "find it" (standalone)
			r"show\s+it\b",  # "show it" (standalone)
			r"where\s+is\s+something\b",  # "where is something"
			r"where\s+is\s+anything\b",  # "where is anything"
		]
		
		for pattern in vague_only_patterns:
			if re.search(pattern, query_lower):
				return False  # Truly vague, no clear object
		
		# If we get here, assume it might be clear (let the system try to process it)
		# This is more permissive - better to try than to ask for clarification
		return True
	
	def check_and_clarify(self, query: str) -> Optional[Dict[str, Any]]:
		"""
		Check if query needs clarification. 
		Only asks for clarification for truly ambiguous queries (vague references).
		Does NOT ask for clarification for:
		- Simple "where is [object]" queries with clear object names
		- Queries asking for object details ("what kind", "how does it look", etc.)
		"""
		# Only check object-finding queries
		if not self._is_object_finding_query(query):
			return None
		
		# If query is asking for specific details, don't ask for clarification
		# Those queries should proceed to get detailed descriptions
		if self._is_specific_detail_query(query):
			return None
		
		# If query has a clear object name, don't ask for clarification
		# Simple "where is cup" or "where is table" should proceed directly
		if self._has_clear_object_name(query):
			return None
		
		# Only ask for clarification for truly vague queries
		# Fast heuristic check for vague references
		query_lower = query.lower().strip()
		
		# Check for vague references without object names
		vague_patterns = [
			r"\bwhere.*\bit\b",  # "where is it" without context
			r"\bwhere.*\bthat\b.*\bthing\b",  # "where is that thing"
			r"\bfind.*\bit\b",  # "find it"
			r"\bshow.*\bit\b",  # "show it"
		]
		
		for pattern in vague_patterns:
			if re.search(pattern, query_lower):
				# Only ask if there's no clear object name
				if not self._has_clear_object_name(query):
					return {
						"question": "What object are you looking for?",
						"reason": "Vague reference without object name"
					}
		
		# Check for ambiguous object pairs (pins/pens) - only for these specific cases
		object_name = self._extract_object_name(query)
		if object_name in self._ambiguous_pairs:
			# Only ask if it's the pins/pens case (very similar sounding)
			if object_name in ["pins", "pin", "pens", "pen"]:
				# For pins/pens, we might want to ask, but let's be conservative
				# Only ask if context suggests confusion
				if "pin" in query_lower and "pen" in query_lower:
					# Both mentioned - probably need clarification
					return {"question": "Do you mean pens or pins?", "reason": "similar-sounding words"}
		
		# Don't use LLM for clarification - it's too aggressive
		# Only use fast heuristics for truly vague queries
		return None


class Planner:
	"""Hybrid LLM-based planner: 30% fast-paths for common queries, 70% LLM with caching."""
	
	def __init__(self, registry: ToolRegistry, api_key: Optional[str] = None) -> None:
		self._registry = registry
		self._api_key = api_key or os.environ.get("MISTRAL_API_KEY") or ""
		self._llm: Optional[MistralChat] = None
		if self._api_key and MistralChat:
			try:
				# Increased max_tokens to 200 to handle complex plans with multiple steps
				# Lower temperature for more deterministic JSON output
				self._llm = MistralChat(api_key=self._api_key, verbose=False, max_tokens=200, temperature=0.1)
			except Exception:
				pass
		
		# Plan cache: stores LLM-generated plans for similar queries
		# Key: normalized query string, Value: (Plan, timestamp)
		self._plan_cache: Dict[str, Tuple[Plan, float]] = {}
		self._cache_ttl_sec = 300.0  # Cache plans for 5 minutes
		self._max_cache_size = 100  # Maximum number of cached plans
		
		# Performance tracking variables (for integration with performance monitor)
		self._last_cache_hit = False
		self._last_fast_path = False

	def _build_tools_description(self) -> str:
		"""Build a description of available tools for the LLM."""
		tools = self._registry.list_tools()
		desc = "Available tools:\n"
		for t in tools:
			desc += f"- {t.name}: {t.description}\n"
			desc += f"  Inputs: {t.inputs}\n"
			desc += f"  Outputs: {t.outputs}\n"
			if t.dependencies:
				desc += f"  Requires: {', '.join(t.dependencies)}\n"
		return desc

	def _normalize_query_for_cache(self, query: str) -> str:
		"""Normalize query for cache key generation (remove punctuation, lowercase, strip)."""
		normalized = query.lower().strip()
		# Remove punctuation except spaces
		normalized = re.sub(r'[^\w\s]', '', normalized)
		# Normalize whitespace
		normalized = ' '.join(normalized.split())
		return normalized
	
	def _check_plan_cache(self, query: str) -> Optional[Plan]:
		"""Check if a plan exists in cache for this query."""
		cache_key = self._normalize_query_for_cache(query)
		current_time = time.time()
		
		if cache_key in self._plan_cache:
			cached_plan, cache_time = self._plan_cache[cache_key]
			age = current_time - cache_time
			
			if age < self._cache_ttl_sec:
				print(f"[DEBUG] Plan cache HIT for query (age: {age:.1f}s)")
				# Return a new plan with fresh ID but same structure
				return Plan(
					id=str(int(time.time()*1000)),
					steps=[PlanStep(
						id=step.id,
						tool=step.tool,
						params=step.params.copy(),
						deps=step.deps.copy()
					) for step in cached_plan.steps]
				)
			else:
				# Cache expired, remove it
				del self._plan_cache[cache_key]
				print(f"[DEBUG] Plan cache EXPIRED for query (age: {age:.1f}s > {self._cache_ttl_sec}s)")
		
		return None
	
	def _store_plan_in_cache(self, query: str, plan: Plan) -> None:
		"""Store plan in cache."""
		cache_key = self._normalize_query_for_cache(query)
		current_time = time.time()
		
		# Clean expired entries if cache is getting large
		if len(self._plan_cache) >= self._max_cache_size:
			# Remove oldest entries
			sorted_entries = sorted(self._plan_cache.items(), key=lambda x: x[1][1])
			entries_to_remove = len(self._plan_cache) - self._max_cache_size + 1
			for i in range(entries_to_remove):
				del self._plan_cache[sorted_entries[i][0]]
			print(f"[DEBUG] Plan cache: Removed {entries_to_remove} old entries")
		
		self._plan_cache[cache_key] = (plan, current_time)
		print(f"[DEBUG] Plan cache: Stored plan for query (cache size: {len(self._plan_cache)})")
	
	def plan(self, query: str, context: Dict[str, Any]) -> Plan:
		"""
		Hybrid planning: 30% fast-paths for common object location queries, 70% LLM with caching.
		
		Flow:
		1. Check plan cache (LLM-generated plans)
		2. Check fast-path (only simple object location queries)
		3. Use LLM planner (with context-aware planning)
		4. Cache LLM-generated plan
		"""
		# Reset tracking variables
		self._last_cache_hit = False
		self._last_fast_path = False
		
		query_lower = query.lower().strip()
		print(f"[DEBUG] Planner.plan: Query='{query}' (lowercase: '{query_lower}')")
		
		# Step 1: Check plan cache first (fastest path for repeated queries)
		cached_plan = self._check_plan_cache(query)
		if cached_plan:
			self._last_cache_hit = True
			return cached_plan
		
		# Step 2: Fast-path ONLY for simple object location queries (30% of queries)
		# These are the most common and simplest queries - keep them fast
		
		# Enhanced skip words - includes pronouns and common words that should never be object names
		skip_words = {"the", "a", "an", "any", "is", "where", "there", "you", "see", "can", "do", 
		              "look", "for", "find", "locate", "search", "what", "that", "this", "it",
		              "them", "all", "some", "many", "few", "one", "ones", "which", "who", "how",
		              "are", "were", "was", "be", "been", "being", "have", "has", "had", "will",
		              "would", "could", "should", "may", "might", "must", "shall"}
		
		# First, try to extract object name from "all the X" or "all X" patterns (most common for plural queries)
		all_patterns = [
			(r"all\s+the\s+([a-z]+(?:s|es|ies)?)", 1),  # "all the bottles", "all the boxes"
			(r"all\s+([a-z]+(?:s|es|ies)?)", 1),  # "all bottles", "all boxes"
		]
		
		for pattern, object_group_idx in all_patterns:
			match = re.search(pattern, query_lower)
			if match:
				groups = match.groups()
				if len(groups) >= object_group_idx:
					object_name = groups[object_group_idx - 1]
					if object_name:
						object_name = object_name.strip().rstrip('.,!?;:')
					if object_name and len(object_name) > 2 and object_name not in skip_words:
						# Convert plural to singular for better matching (bottles -> bottle)
						singular_name = object_name.rstrip('s') if object_name.endswith('s') and len(object_name) > 3 else object_name
						print(f"[DEBUG] Fast-path planning: simple object location query detected for '{object_name}' (singular: '{singular_name}')")
						self._last_fast_path = True
						plan = Plan(
							id=str(int(time.time()*1000)),
							steps=[
								PlanStep(id="s1", tool="object_detector", params={"targetClasses": [singular_name]}, deps=[]),
								PlanStep(id="s2", tool="chatbot", params={"query": query}, deps=["s1"])
							]
						)
						# Cache fast-path plans too (for consistency)
						self._store_plan_in_cache(query, plan)
						return plan
		
		# Then try standard patterns (but skip pronouns)
		simple_location_patterns = [
			(r"(?:where\s+are|where\s+is|where's)\s+(?:all\s+the\s+|all\s+)?(?:the\s+|a\s+|an\s+)?([a-z]+(?:s|es|ies)?)", 1),  # "where are all the bottles"
			(r"find\s+(?:all\s+the\s+|all\s+)?(?:the\s+|a\s+|an\s+)?([a-z]+(?:s|es|ies)?)", 1),  # "find all the bottles"
			(r"look\s+for\s+(?:all\s+the\s+|all\s+)?(?:the\s+|a\s+|an\s+)?([a-z]+(?:s|es|ies)?)", 1),  # "look for all the bottles"
			(r"is\s+there\s+(?:a|an|any|the)?\s+([a-z]+(?:s|es|ies)?)", 1),
			(r"do\s+you\s+see\s+(?:a|an|any|the)?\s+([a-z]+(?:s|es|ies)?)", 1),
			(r"can\s+you\s+see\s+(?:a|an|any|the)?\s+([a-z]+(?:s|es|ies)?)", 1),
			(r"locate\s+(?:all\s+the\s+|all\s+)?(?:the\s+|a\s+|an\s+)?([a-z]+(?:s|es|ies)?)", 1),
			(r"search\s+for\s+(?:all\s+the\s+|all\s+)?(?:the\s+|a\s+|an\s+)?([a-z]+(?:s|es|ies)?)", 1),
		]
		
		for pattern, object_group_idx in simple_location_patterns:
			match = re.search(pattern, query_lower)
			if match:
				groups = match.groups()
				if len(groups) >= object_group_idx:
					object_name = groups[object_group_idx - 1]
					if object_name:
						object_name = object_name.strip().rstrip('.,!?;:')
					if object_name and len(object_name) > 2 and object_name not in skip_words:
						# Convert plural to singular for better matching (bottles -> bottle)
						singular_name = object_name.rstrip('s') if object_name.endswith('s') and len(object_name) > 3 else object_name
						print(f"[DEBUG] Fast-path planning: simple object location query detected for '{object_name}' (singular: '{singular_name}')")
						self._last_fast_path = True
						plan = Plan(
							id=str(int(time.time()*1000)),
							steps=[
								PlanStep(id="s1", tool="object_detector", params={"targetClasses": [singular_name]}, deps=[]),
								PlanStep(id="s2", tool="chatbot", params={"query": query}, deps=["s1"])
							]
						)
						# Cache fast-path plans too (for consistency)
						self._store_plan_in_cache(query, plan)
						return plan
		
		# Step 3: Use LLM planner for all other queries (70% of queries)
		print(f"[DEBUG] Using LLM planner for intelligent planning...")
		
		if not self._llm:
			# Fallback: simple chatbot-only plan if LLM unavailable
			fallback_plan = Plan(id=str(int(time.time()*1000)), steps=[PlanStep(id="answer", tool="chatbot", params={"query": query})])
			self._store_plan_in_cache(query, fallback_plan)
			return fallback_plan
		
		# Build context-aware system prompt with conversation history and detected objects
		tools_desc = self._build_tools_description()
		
		# Extract context information
		conversation_history = context.get("turns", [])[-3:] if context.get("turns") else []  # Last 3 turns
		recent_objects = context.get("recent_snapshots_count", 0)
		conversation_str = "\n".join([f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in conversation_history]) if conversation_history else "No previous conversation"
		
		system_prompt = f"""You are an intelligent planning agent for a blind person assistant.

Available tools:
{tools_desc}

Context:
- Previous conversation: {conversation_str}
- Recently detected objects: {recent_objects} objects in memory

Your task:
1. Understand the user's intent from the query
2. Select appropriate tools based on intent
3. Create execution plan with correct dependencies
4. Consider context from previous conversation turns

CRITICAL: You MUST return ONLY valid JSON. No explanations, no markdown, just pure JSON.

Required JSON format:
{{"steps": [{{"id": "s1", "tool": "tool_name", "params": {{}}, "deps": []}}]}}

Example for scene description:
{{"steps": [{{"id": "s1", "tool": "object_detector", "params": {{"targetClasses": null}}, "deps": []}}, {{"id": "s2", "tool": "pixtral_analysis", "params": {{"scene_mode": true}}, "deps": ["s1"]}}, {{"id": "s3", "tool": "chatbot", "params": {{"query": "..."}}, "deps": ["s2"]}}]}}

RULES:
1. Scene description queries ("describe the scene", "what can you see in the scene", "what do you see"):
   - object_detector(targetClasses: null) → pixtral_analysis(scene_mode: true) → chatbot

2. Object location queries (where/find/locate/is there):
   - object_detector(targetClasses: ["object_name"]) → chatbot

3. Person queries by name (e.g., "is reshad around", "where is maliha"):
   - object_detector(targetClasses: ["person"]) → face_recognition → chatbot

4. Object description queries ("how does [object] look", "describe [object]"):
   - object_detector(targetClasses: ["object"]) → pixtral_analysis(scene_mode: false) → chatbot

5. Document queries (read/scan/summarize):
   - document_scan(intent: "read"/"summarize"/"analyze"/"extract") → chatbot

6. ALWAYS end with chatbot step that depends on previous steps

Return ONLY valid JSON - no text before or after."""
		
		# Enhanced prompt with context
		prompt = f"Query: {query}\n\nCreate the execution plan JSON:"
		
		try:
			response = self._llm.chat(prompt, system_prompt=system_prompt)
			
			# Check if response is an error message (user-friendly error messages from rate limiting, etc.)
			error_indicators = [
				"I'm experiencing high demand",
				"I encountered an error",
				"Please wait a moment",
				"API authentication",
				"service tier",
				"capacity exceeded"
			]
			if any(indicator in response for indicator in error_indicators):
				# LLM call returned an error message, fall back to simple plan
				print(f"[DEBUG] LLM planning returned error message: {response[:100]}... Falling back to simple plan")
				raise ValueError("LLM planning failed due to API error")
			
			# Parse JSON from response (may have markdown code blocks or extra text)
			# Extract JSON if wrapped in ```json or ```
			json_start = response.find('{')
			json_end = response.rfind('}') + 1
			if json_start >= 0 and json_end > json_start:
				json_str = response[json_start:json_end]
			else:
				json_str = response
			
			# Try to parse JSON
			try:
				plan_data = json.loads(json_str)
			except json.JSONDecodeError as json_err:
				# Log the actual response for debugging
				print(f"[DEBUG] LLM response (first 500 chars): {response[:500]}")
				print(f"[DEBUG] JSON parsing error: {json_err}")
				print(f"[DEBUG] Extracted JSON string (first 300 chars): {json_str[:300]}")
				
				# If JSON parsing fails and response looks like an error, treat as error
				if any(indicator in response for indicator in error_indicators):
					print(f"[DEBUG] LLM planning response is not valid JSON and contains error indicators. Falling back to simple plan")
					raise ValueError("LLM planning failed - invalid JSON response")
				# Otherwise, re-raise the JSON error
				raise
			steps = []
			for s in plan_data.get("steps", []):
				step = PlanStep(
					id=str(s.get("id", "")),
					tool=str(s.get("tool", "")),
					params=s.get("params", {}),
					deps=[str(d) for d in s.get("deps", [])]
				)
				# Validate tool exists
				if not self._registry.get(step.tool):
					raise ValueError(f"LLM selected unknown tool: {step.tool}")
				steps.append(step)
			
			# Fallback: If LLM created a plan with only chatbot step, check if it should have object_detector or document_scan
			# Use simple keyword check (not strict regex) - let LLM handle nuances
			if len(steps) == 1 and steps[0].tool == "chatbot":
				# Check if query is a document scanning query first
				document_keywords = ["scan", "read", "extract", "text", "document", "paper", "page"]
				if any(keyword in query_lower for keyword in document_keywords):
					# Detect intent from query
					intent = "extract"  # default
					if any(word in query_lower for word in ["read", "what does", "say"]):
						intent = "read"
					elif "summarize" in query_lower:
						intent = "summarize"
					elif "analyze" in query_lower:
						intent = "analyze"
					
					# Detect document type
					doc_type = None
					if "receipt" in query_lower:
						doc_type = "receipt"
					elif "letter" in query_lower:
						doc_type = "letter"
					elif "form" in query_lower:
						doc_type = "form"
					elif "label" in query_lower:
						doc_type = "label"
					
					# Document scanning query - add document_scan step
					print(f"[DEBUG] LLM created chatbot-only plan for document query. Adding document_scan step with intent='{intent}'")
					steps.insert(0, PlanStep(id="s1", tool="document_scan", params={"intent": intent, "document_type": doc_type}, deps=[]))
					steps[1].id = "s2"
					steps[1].deps = ["s1"]
				# Check if query is a scene description query (needs pixtral_analysis with scene_mode=True)
				# IMPORTANT: Check for scene queries with comprehensive patterns including "what can you see" and "describe me"
				is_scene_query = (
					any(phrase in query_lower for phrase in ["describe the scene", "describe scene", "what do you see", 
					    "what can you see", "what's in front", "what is in front", "tell me about the scene", 
					    "what's in the scene", "what is in the scene", "can you describe the scene"]) or
					("describe" in query_lower and "scene" in query_lower) or  # "describe me" + "scene"
					("what can you see" in query_lower and "scene" in query_lower)  # "what can you see in the scene"
				)
				
				if is_scene_query:
					# Scene description query - needs object_detector → pixtral_analysis(scene_mode=True) → chatbot
					print(f"[DEBUG] LLM created chatbot-only plan for scene description query. Adding object_detector and pixtral_analysis steps")
					steps.insert(0, PlanStep(id="s1", tool="object_detector", params={"targetClasses": None}, deps=[]))
					steps.insert(1, PlanStep(id="s2", tool="pixtral_analysis", params={"scene_mode": True, "query": query}, deps=["s1"]))
					steps[2].id = "s3"
					steps[2].deps = ["s2"]
				# Check if query seems to be about object detection (simple keyword-based check)
				# BUT exclude scene queries (already handled above) - "see" + "scene" = scene query, not object query
				elif any(keyword in query_lower for keyword in ["find", "where", "locate", "search", "show", "look", "detect", "spot", "visible", "around"]) and not ("scene" in query_lower and ("describe" in query_lower or "see" in query_lower)):
					# LLM should have included object_detector but didn't - add it with null targetClasses
					# (let object_detector detect all objects; LLM will understand from context)
					print(f"[DEBUG] LLM created chatbot-only plan for object-related query. Adding object_detector step (detect all objects)")
					steps.insert(0, PlanStep(id="s1", tool="object_detector", params={"targetClasses": None}, deps=[]))
					steps[1].id = "s2"
					steps[1].deps = ["s1"]
			
			if not steps:
				# Fallback if LLM returned empty plan
				steps = [PlanStep(id="answer", tool="chatbot", params={"query": query})]
			
			plan = Plan(id=str(int(time.time()*1000)), steps=steps)
			
			# Step 4: Cache the LLM-generated plan for future use
			self._store_plan_in_cache(query, plan)
			
			return plan
		except Exception as e:
			print(f"[WARNING] LLM planning failed: {e}")
			# Intelligent fallback: detect query type and create appropriate plan
			query_lower = query.lower().strip()
			
			# Check for scene description queries (highest priority - needs Pixtral)
			# IMPORTANT: Check scene queries FIRST before object queries, and use more comprehensive patterns
			# Check if it's a scene description query
			is_scene_query = (
				any(phrase in query_lower for phrase in ["describe the scene", "describe scene", "what do you see", 
				    "what can you see", "what's in front", "what is in front", "tell me about the scene", 
				    "what's in the scene", "what is in the scene", "can you describe the scene"]) or
				("describe" in query_lower and "scene" in query_lower) or  # "describe me" + "scene"
				("what can you see" in query_lower and "scene" in query_lower)  # "what can you see in the scene"
			)
			
			if is_scene_query:
				print(f"[DEBUG] LLM planning failed, but detected scene description query. Creating proper plan with pixtral_analysis")
				fallback_plan = Plan(
					id=str(int(time.time()*1000)),
					steps=[
						PlanStep(id="s1", tool="object_detector", params={"targetClasses": None}, deps=[]),
						PlanStep(id="s2", tool="pixtral_analysis", params={"scene_mode": True, "query": query}, deps=["s1"]),
						PlanStep(id="s3", tool="chatbot", params={"query": query}, deps=["s2"])
					]
				)
				# Cache this fallback plan too (it's correct)
				self._store_plan_in_cache(query, fallback_plan)
				return fallback_plan
			
			# Check for document queries
			document_keywords = ["scan", "read", "extract", "text", "document", "paper", "page"]
			if any(keyword in query_lower for keyword in document_keywords):
				intent = "extract"
				if any(word in query_lower for word in ["read", "what does", "say"]):
					intent = "read"
				elif "summarize" in query_lower:
					intent = "summarize"
				elif "analyze" in query_lower:
					intent = "analyze"
				
				doc_type = None
				if "receipt" in query_lower:
					doc_type = "receipt"
				elif "letter" in query_lower:
					doc_type = "letter"
				elif "form" in query_lower:
					doc_type = "form"
				elif "label" in query_lower:
					doc_type = "label"
				
				print(f"[DEBUG] LLM planning failed, but detected document query. Creating proper plan with document_scan")
				fallback_plan = Plan(
					id=str(int(time.time()*1000)),
					steps=[
						PlanStep(id="s1", tool="document_scan", params={"intent": intent, "document_type": doc_type}, deps=[]),
						PlanStep(id="s2", tool="chatbot", params={"query": query}, deps=["s1"])
					]
				)
				self._store_plan_in_cache(query, fallback_plan)
				return fallback_plan
			
			# Check for object-related queries (but NOT scene queries - already handled above)
			# Exclude queries that have both "see" and "scene" as those are scene queries
			is_object_query = (
				any(keyword in query_lower for keyword in ["find", "where", "locate", "search", "show", "look", "detect", "spot", "visible", "around", "is there"]) and
				not ("scene" in query_lower and ("describe" in query_lower or "see" in query_lower))  # Exclude scene queries
			)
			
			if is_object_query:
				print(f"[DEBUG] LLM planning failed, but detected object-related query. Creating plan with object_detector")
				fallback_plan = Plan(
					id=str(int(time.time()*1000)),
					steps=[
						PlanStep(id="s1", tool="object_detector", params={"targetClasses": None}, deps=[]),
						PlanStep(id="s2", tool="chatbot", params={"query": query}, deps=["s1"])
					]
				)
				self._store_plan_in_cache(query, fallback_plan)
				return fallback_plan
			
			# Last resort: simple chatbot-only plan
			print(f"[DEBUG] LLM planning failed, no specific query type detected. Using simple chatbot-only plan")
			fallback_plan = Plan(id=str(int(time.time()*1000)), steps=[PlanStep(id="answer", tool="chatbot", params={"query": query})])
			# Don't cache simple fallback plans (they're not useful)
			return fallback_plan


class ContextWindow:
	"""Context window with 2-minute temporal memory for object detections stored as timestamped snapshots."""
	
	def __init__(self, max_turns: int = 6, memory_window_sec: float = 120.0) -> None:
		self._turns: List[Dict[str, Any]] = []
		self._facts: List[Dict[str, Any]] = []
		self._cache: Dict[str, Any] = {}
		self._max_turns = max_turns
		# Temporal memory: list of snapshots with timestamp and objects detected at that time
		# Format: [{"timestamp": 1234567890.0, "objects": [{"class": "keyboard", "distance": 1.5, ...}, ...]}, ...]
		self._detection_snapshots: List[Dict[str, Any]] = []
		self._memory_window_sec = memory_window_sec
		self._last_scene_context: Optional[str] = None

	def add_turn(self, role: str, content: str) -> None:
		self._turns.append({"role": role, "content": content, "ts": time.time()})
		if len(self._turns) > self._max_turns:
			self._turns.pop(0)

	def add_fact(self, fact: Dict[str, Any]) -> None:
		self._facts.append({**fact, "ts": time.time()})
		if len(self._facts) > 200:
			self._facts.pop(0)

	def update_environment_facts(self, objects: List[Dict[str, Any]], scene_context: Optional[str] = None) -> None:
		"""Store object detections as a timestamped snapshot."""
		current_time = time.time()
		
		# Clean old snapshots (older than memory window)
		old_count = len(self._detection_snapshots)
		self._detection_snapshots = [
			snap for snap in self._detection_snapshots
			if (current_time - snap.get("timestamp", 0)) <= self._memory_window_sec
		]
		if old_count != len(self._detection_snapshots):
			print(f"[DEBUG] Cleaned {old_count - len(self._detection_snapshots)} old snapshots")
		
		# Create snapshot with current objects
		object_classes = [obj.get("class") or obj.get("class_name") or "object" for obj in objects]
		snapshot = {
			"timestamp": current_time,
			"objects": objects.copy(),  # Store full object data
			"classes": object_classes,  # Quick lookup list
			"scene_context": scene_context or self._last_scene_context,
		}
		self._detection_snapshots.append(snapshot)
		
		# Format: [00:00:10] [keyboard, monitor, cup]
		time_str = self._format_timestamp(current_time)
		classes_str = ", ".join(object_classes) if object_classes else "no objects"
		print(f"[DEBUG] [{time_str}] [{classes_str}]")
		
		# Keep only last 200 snapshots to prevent memory issues
		if len(self._detection_snapshots) > 200:
			self._detection_snapshots.pop(0)
		
		if scene_context:
			self._last_scene_context = scene_context
	
	def _format_timestamp(self, timestamp: float) -> str:
		"""Format timestamp as [HH:MM:SS] relative to start or absolute."""
		# For simplicity, use elapsed time from first snapshot or current time
		if self._detection_snapshots:
			first_time = self._detection_snapshots[0].get("timestamp", timestamp)
			elapsed = timestamp - first_time
		else:
			elapsed = 0
		
		hours = int(elapsed // 3600)
		minutes = int((elapsed % 3600) // 60)
		seconds = int(elapsed % 60)
		return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

	def get_recent_objects(self, class_name: Optional[str] = None, max_age_sec: Optional[float] = None) -> List[Dict[str, Any]]:
		"""Find the most recent snapshot containing the specified object class.
		Returns list with the most recent matching snapshot (or empty if not found)."""
		current_time = time.time()
		window = max_age_sec if max_age_sec is not None else self._memory_window_sec
		
		print(f"[DEBUG] get_recent_objects: Searching for class='{class_name}', max_age={max_age_sec or self._memory_window_sec}s")
		print(f"[DEBUG] Total snapshots: {len(self._detection_snapshots)}")
		
		if not class_name:
			# Return all recent snapshots
			recent = [
				snap for snap in self._detection_snapshots
				if (current_time - snap.get("timestamp", 0)) <= window
			]
			recent.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
			return recent
		
		# Search for specific class - go through snapshots in reverse chronological order
		class_lower = str(class_name).lower().strip()
		print(f"[DEBUG] Searching snapshots for '{class_lower}'...")
		
		# Search backwards through snapshots (most recent first)
		for snap in reversed(self._detection_snapshots):
			snap_time = snap.get("timestamp", 0)
			age_sec = current_time - snap_time
			
			# Skip if too old
			if age_sec > window:
				continue
			
			# Check if this snapshot contains the object
			classes = snap.get("classes", [])
			objects = snap.get("objects", [])
			
			# Try to find matching object using fuzzy matching (same as current detection)
			for obj in objects:
				obj_class = str(obj.get("class", "")).lower()
				recognized_name = obj.get("recognized_name")  # For person name matching
				# Use same fuzzy matching logic as current detection for consistency
				if _fuzzy_match_class(class_lower, obj_class, recognized_name=recognized_name):
					# Found it! Return the object with snapshot info
					time_str = self._format_timestamp(snap_time)
					print(f"[DEBUG] ✓ Found '{class_lower}' in snapshot [{time_str}], age={age_sec:.2f}s (matched '{obj_class}' with recognized_name='{recognized_name}')")
					# Return object with snapshot timestamp
					result = obj.copy()
					result["snapshot_timestamp"] = snap_time
					result["snapshot_age_seconds"] = age_sec
					return [result]  # Return as list for compatibility
		
		print(f"[DEBUG] ✗ '{class_lower}' not found in any snapshot within {window}s window")
		return []

	def get_temporal_summary(self) -> str:
		"""Get a human-readable summary of objects seen in recent snapshots."""
		if not self._detection_snapshots:
			return "No objects detected in recent memory."
		
		current_time = time.time()
		recent = [
			snap for snap in self._detection_snapshots
			if (current_time - snap.get("timestamp", 0)) <= self._memory_window_sec
		]
		
		if not recent:
			return "No objects detected in recent memory."
		
		# Show last few snapshots
		summary_parts = []
		for snap in recent[-5:]:  # Last 5 snapshots
			time_str = self._format_timestamp(snap.get("timestamp", 0))
			classes = snap.get("classes", [])
			classes_str = ", ".join(classes) if classes else "no objects"
			summary_parts.append(f"[{time_str}] [{classes_str}]")
		
		return "\n".join(summary_parts)

	def get_snapshot(self) -> Dict[str, Any]:
		"""Get full context snapshot including temporal memory summary."""
		temporal_summary = self.get_temporal_summary()
		return {
			"turns": list(self._turns), 
			"facts": list(self._facts), 
			"cache_keys": list(self._cache.keys()),
			"temporal_memory": temporal_summary,
			"recent_snapshots_count": len([s for s in self._detection_snapshots if (time.time() - s.get("timestamp", 0)) <= self._memory_window_sec])
		}

	def cache_set(self, key: str, value: Any) -> None:
		self._cache[key] = value

	def cache_get(self, key: str) -> Any:
		return self._cache.get(key)

	def cache_recent_frame(self, frame: Any) -> None:
		self._cache["recent_frame"] = frame

	def cache_recent_detections(self, detections: List[Dict[str, Any]]) -> None:
		self._cache["recent_detections"] = detections


class ExecutionEngine:
	def __init__(self, object_detector: ObjectDetectorAdapter, scene_analysis: SceneAnalysisAdapter, chatbot: ChatbotAdapter, document_scan: Optional[Callable[..., Dict[str, Any]]] = None, pixtral_analysis: Optional[Any] = None, face_recognition: Optional[Any] = None) -> None:
		self._object_detector = object_detector
		self._scene_analysis = scene_analysis
		self._chatbot = chatbot
		self._document_scan_fn = document_scan
		self._pixtral_analysis = pixtral_analysis
		self._face_recognition = face_recognition
		# Cache for detection results (avoid redundant processing)
		self._detection_cache: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}  # key: target_classes, value: (objects, timestamp)
		self._cache_ttl_sec = 2.0  # Cache valid for 2 seconds (increased from 0.5s for better performance)

	async def run(self, plan: Plan, context: ContextWindow, frame_provider: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
		results: Dict[str, Any] = {}
		scene_context: Optional[str] = None
		
		print(f"[DEBUG] ExecutionEngine.run: Processing {len(plan.steps)} steps")
		
		for step in plan.steps:
			print(f"[DEBUG] Executing step {step.id}: tool={step.tool}, params={step.params}")
			# Wait for deps (sequential for now)
			if step.tool == "object_detector":
				frame = frame_provider() if frame_provider else None
				if frame is None:
					print(f"[DEBUG] WARNING: frame_provider returned None for step {step.id}")
				else:
					print(f"[DEBUG] Frame obtained: shape={frame.shape if hasattr(frame, 'shape') else 'unknown'}")
				target_classes = step.params.get("targetClasses")
				print(f"[DEBUG] targetClasses={target_classes}")
				if frame is None:
					results[step.id] = {"objects": [], "target_classes": target_classes}
					print(f"[DEBUG] Step {step.id}: No frame, returning empty objects")
					continue
				
				# Check cache for recent detection results (prioritize continuous loop cache)
				cache_key = str(sorted(target_classes)) if target_classes else "all"
				current_time = time.time()
				print(f"[DEBUG] Cache key: {cache_key}")
				
				# First, try to use cached objects from continuous detection loop (most recent)
				cached_objects_from_loop = context.cache_get("last_objects")
				objects = []
				detection_result = None
				use_cached_detection = False
				
				# Check if we can use continuous loop cache
				if cached_objects_from_loop:
					# Filter cached objects by target_classes if specified
					if target_classes:
						target_list = target_classes if isinstance(target_classes, list) else [target_classes]
						target_lower_list = [str(t).lower() for t in target_list]
						# Filter cached objects to match target classes
						filtered_objects = []
						for obj in cached_objects_from_loop:
							obj_class = str(obj.get("class", "")).lower()
							recognized_name = obj.get("recognized_name", "").lower() if obj.get("recognized_name") else ""
							# Check if object matches any target class
							for target in target_lower_list:
								if _fuzzy_match_class(target, obj_class, recognized_name=recognized_name):
									filtered_objects.append(obj)
									break
						objects = filtered_objects
						if objects:
							print(f"[DEBUG] Using continuous loop cache: {len(objects)} objects filtered from {len(cached_objects_from_loop)} cached objects")
							use_cached_detection = True
					else:
						# No target classes - use all cached objects
						objects = cached_objects_from_loop
						print(f"[DEBUG] Using continuous loop cache: {len(objects)} objects (all objects)")
						use_cached_detection = True
				
				# If continuous loop cache not available or insufficient, check execution cache
				if not use_cached_detection:
					if cache_key in self._detection_cache:
						cached_objects, cache_time = self._detection_cache[cache_key]
						if (current_time - cache_time) < self._cache_ttl_sec:
							objects = cached_objects
							print(f"[DEBUG] Using execution cache: {len(objects)} objects (age: {current_time - cache_time:.2f}s)")
							use_cached_detection = True
				
				# If no cache available, run detection
				if not use_cached_detection:
					print(f"[DEBUG] No cache available, running detection...")
					detection_result = self._object_detector.detect_with_tracking(frame, target_classes=target_classes)
					objects = detection_result.get("objects", [])
					print(f"[DEBUG] Detection completed: {len(objects)} objects found")
					# Update both caches
					self._detection_cache[cache_key] = (objects, current_time)
					context.cache_set("last_objects", objects)  # Update continuous loop cache
				
				# Initialize results[step.id] before accessing it
				results[step.id] = {}
				
				# Auto-run face recognition on person objects if face recognition is available
				if self._face_recognition and objects:
					person_objects = [obj for obj in objects if obj.get("class", "").lower() == "person"]
					if person_objects:
						print(f"[DEBUG] Running face recognition on {len(person_objects)} person objects")
						try:
							recognized_persons = self._face_recognition.recognize_person_objects(frame, person_objects)
							# Update objects list with recognized names
							for i, obj in enumerate(objects):
								if obj.get("class", "").lower() == "person":
									# Find matching recognized person
									for rec_person in recognized_persons:
										if rec_person.get("recognized_name"):
											# Match by bbox or position
											if (obj.get("bbox") == rec_person.get("bbox") or
											    abs(obj.get("bbox", [0])[0] - rec_person.get("bbox", [0])[0]) < 10):
												objects[i] = rec_person
												break
							print(f"[DEBUG] Face recognition completed: {len([o for o in objects if o.get('recognized_name')])} persons recognized")
						except Exception as e:
							print(f"[WARNING] Face recognition failed: {e}")
				
				# If we're looking for a specific class, check if it's in current results
				# Then decide: report current OR check historical memory
				if target_classes:
					target = target_classes[0] if isinstance(target_classes, list) else target_classes
					target_lower = str(target).lower()
					
					# Check if target class is in current detection results
					target_found_in_current = False
					matched_current_object = None
					
					if objects:
						print(f"[DEBUG] Checking {len(objects)} current objects for '{target}' (target_lower='{target_lower}')")
						for obj in objects:
							obj_class = str(obj.get("class", "")).lower()
							recognized_name = obj.get("recognized_name")
							print(f"[DEBUG]   Comparing: '{obj_class}' (recognized_name='{recognized_name}') with '{target_lower}'")
							if _fuzzy_match_class(target_lower, obj_class, recognized_name=recognized_name):
								target_found_in_current = True
								matched_current_object = obj
								print(f"[DEBUG]   ✓ MATCH FOUND: '{obj_class}' (recognized_name='{recognized_name}') matches '{target_lower}'")
								break
						if not target_found_in_current:
							print(f"[DEBUG]   ✗ No match found in current objects")
					else:
						print(f"[DEBUG] No objects in current detection")
					
					# If not found in current detection results, check cached objects from continuous loop
					if not target_found_in_current:
						cached_objects = context.cache_get("last_objects")
						if cached_objects:
							print(f"[DEBUG] Checking {len(cached_objects)} cached objects from continuous detection loop...")
							for obj in cached_objects:
								obj_class = str(obj.get("class", "")).lower()
								recognized_name = obj.get("recognized_name")
								if _fuzzy_match_class(target_lower, obj_class, recognized_name=recognized_name):
									target_found_in_current = True
									matched_current_object = obj
									print(f"[DEBUG]   ✓ MATCH FOUND in cache: '{obj_class}' (recognized_name='{recognized_name}') matches '{target_lower}'")
									# Update objects list to include this matched object
									if matched_current_object not in objects:
										objects.append(matched_current_object)
									break
					
					# Simple logic:
					# - If found in current: report as "found now" (don't check history)
					# - If NOT found in current: check history for "saw it earlier" message (with threshold)
					if target_found_in_current:
						# Object is currently visible - report it as found
						print(f"[DEBUG] {target} found in current detection - reporting as 'found now'")
						# Ensure the matched object is marked as currently found
						if matched_current_object:
							# Store matched object info for context building
							results[step.id]["matched_object"] = matched_current_object
					else:
						# Object NOT found in current view - check historical memory WITH THRESHOLD
						print(f"[DEBUG] {target} NOT found in current detection. Checking historical memory...")
						# Search historical memory with full 2-minute window (120 seconds)
						# Objects are saved for 2 minutes, but we only report if within 30 seconds threshold
						historical = context.get_recent_objects(class_name=str(target), max_age_sec=120.0)
						print(f"[DEBUG] Historical search for '{target}' returned {len(historical)} results")
						
						# Only report historical if:
						# 1. Historical detection exists
						# 2. Historical detection is within memory window (2 minutes = 120 seconds)
						# Note: Objects older than 2 minutes are automatically deleted from memory
						HISTORICAL_THRESHOLD_SEC = 120.0  # Report historical if found in memory (up to 2 minutes)
						
						if historical:
							# Get most recent historical detection
							most_recent = historical[0]  # Already sorted by most recent
							age_sec = most_recent.get("snapshot_age_seconds", time.time() - most_recent.get("snapshot_timestamp", 0))
							print(f"[DEBUG] Most recent historical: age={age_sec:.2f}s, class={most_recent.get('class')}, distance={most_recent.get('approxDistance') or most_recent.get('distance', 0)}")
							
							# Only report historical if within threshold
							if age_sec <= HISTORICAL_THRESHOLD_SEC:
								# Extract angle information from historical detection
								hist_angle_x = most_recent.get("angle_x", 0)
								hist_angle_y = most_recent.get("angle_y", 0)
								hist_distance = most_recent.get("approxDistance") or most_recent.get("distance", 0)
								
								# Store historical detection info
								results[step.id]["historical_found"] = {
									"class": target,
									"age_seconds": age_sec,
									"distance": hist_distance,
									"angle_x": hist_angle_x,
									"angle_y": hist_angle_y,
									"scene_context": most_recent.get("scene_context"),
									"last_seen": most_recent
								}
								print(f"[DEBUG] ✓ Historical context set for {target}: age={age_sec:.1f}s ago (within {HISTORICAL_THRESHOLD_SEC}s threshold)")
							else:
								print(f"[DEBUG] ✗ Historical detection too old ({age_sec:.1f}s > {HISTORICAL_THRESHOLD_SEC}s threshold) - not reporting")
						else:
							print(f"[DEBUG] ✗ No historical detection found for '{target}'")
				else:
					# No specific target classes - detecting all objects (scene description query)
					print(f"[DEBUG] Detecting all objects (targetClasses=None). Found {len(objects)} objects.")
				
				# Store results with target_classes info for context building
				results[step.id].update({
					"objects": objects,
					"target_classes": target_classes  # Track what was searched for
				})
				print(f"[DEBUG] Stored {len(objects)} objects in results[{step.id}]")
				context.cache_set("last_objects", objects)
				context.cache_recent_frame(frame)
				context.cache_recent_detections(objects)
				
				# Store tracking info for Pixtral analysis step (if needed)
				if detection_result and isinstance(detection_result, dict):
					new_track_ids = detection_result.get("new_track_ids", set())
					results[step.id]["new_track_ids"] = new_track_ids
					print(f"[DEBUG] Tracking info: {len(new_track_ids)} new track IDs")
				else:
					results[step.id]["new_track_ids"] = set()
				
				# Update temporal memory with current detections and scene context
				context.update_environment_facts(objects, scene_context=scene_context)
				
				# Optional visualization using existing detector utils
				if os.environ.get("AGENT_VISUAL_MONITOR", "1") == "1":
					_try_draw_and_show(frame, objects)
			elif step.tool == "pixtral_analysis":
				# Get detections from previous step (object_detector) if available
				prev_step_id = step.deps[0] if step.deps else None
				prev_step_result = results.get(prev_step_id, {}) if prev_step_id else {}
				detections = prev_step_result.get("objects", [])
				frame = frame_provider() if frame_provider else None
				
				# Check if this is a scene description query (entire frame) vs object descriptions (clipped ROIs)
				scene_mode = step.params.get("scene_mode", False)
				query_lower = step.params.get("query", "").lower() if isinstance(step.params.get("query"), str) else ""
				
				# Determine if scene analysis is needed
				if not scene_mode:
					scene_keywords = ["describe the scene", "what do you see", "what's in the scene", "tell me about the scene"]
					scene_mode = any(kw in query_lower for kw in scene_keywords)
				
				if scene_mode and frame is not None:
					# SCENE ANALYSIS: Feed entire frame into Pixtral 12B
					print(f"[DEBUG] Pixtral scene analysis: Analyzing entire frame")
					try:
						scene_description = self._pixtral_analysis.analyze_scene(frame, prompt_type="detailed")
						results[step.id] = {
							"scene_description": scene_description,
							"scene_mode": True
						}
						print(f"[DEBUG] Scene analysis completed: {len(scene_description)} chars")
					except Exception as e:
						print(f"[WARNING] Scene analysis failed: {e}")
						results[step.id] = {"scene_description": "Scene analysis failed", "scene_mode": True}
				elif detections and frame is not None:
					# OBJECT ANALYSIS: Analyze individual objects using clipped ROI images
					# Get tracking info if available (from object_detector result)
					new_track_ids = prev_step_result.get("new_track_ids", set())
					
					print(f"[DEBUG] Pixtral object analysis: {len(detections)} detections, {len(new_track_ids)} new track IDs")
					
					try:
						# Run Pixtral analysis on individual objects (clipped ROI images)
						# This uses analyze_detections_with_tracking which crops each object and analyzes it
						enhanced_detections = self._pixtral_analysis.analyze_detections(
							frame, detections, new_track_ids=new_track_ids
						)
						print(f"[DEBUG] Pixtral object analysis completed: {len(enhanced_detections)} enhanced detections")
						
						# Update results with enhanced detections
						results[step.id] = {
							"enhanced_detections": enhanced_detections,
							"objects": enhanced_detections,  # For compatibility with chatbot step
							"scene_mode": False
						}
						
						# Also update previous step's objects with descriptions (for chatbot context)
						if prev_step_id and prev_step_id in results:
							results[prev_step_id]["objects"] = enhanced_detections
							print(f"[DEBUG] Updated previous step {prev_step_id} objects with Pixtral descriptions")
					except Exception as e:
						print(f"[WARNING] Pixtral object analysis failed: {e}")
						# Fallback: return original detections
						results[step.id] = {"objects": detections, "scene_mode": False}
				else:
					# Pixtral not available or no detections - return as-is
					if not self._pixtral_analysis:
						print(f"[DEBUG] Pixtral analysis not available (adapter not initialized)")
					results[step.id] = {"objects": detections if detections else [], "scene_mode": scene_mode}
			elif step.tool == "scene_analysis":
				prev = results.get(step.deps[0]) if step.deps else {}
				objects = prev.get("objects") or context.cache_get("last_objects") or []
				frame = frame_provider() if frame_provider else None
				analysis = self._scene_analysis.analyze(objects, query=step.params.get("query"), frame=frame)
				results[step.id] = analysis
				# Extract scene context if available (e.g., "bedroom", "kitchen", etc.)
				if isinstance(analysis, dict):
					# Try to infer scene from relations or salient regions
					relations = analysis.get("relations", [])
					if relations:
						scene_context = f"scene with {len(objects)} objects"
			elif step.tool == "document_scan":
				if self._document_scan_fn is None:
					results[step.id] = {"blocks": [], "summary": "Document scan not available"}
				else:
					frame = frame_provider() if frame_provider else None
					
					# Extract intent and document_type from step params
					intent = step.params.get("intent", "read")  # Default to "read" for backward compatibility
					document_type = step.params.get("document_type")
					
					print(f"[DEBUG] Document scan step - intent='{intent}', document_type='{document_type}'")
					
					# Check if document_scan_fn is the adapter instance
					if hasattr(self._document_scan_fn, 'scan_with_intent'):
						# Use intelligent scanning with intent
						scan_result = self._document_scan_fn.scan_with_intent(frame, intent=intent, document_type=document_type)
					elif hasattr(self._document_scan_fn, 'scan_with_detection'):
						# Use scan_with_detection (like main.py)
						scan_result = self._document_scan_fn.scan_with_detection(frame, detect_first=True)
						# Add intent metadata
						if scan_result:
							scan_result["intent"] = intent
							scan_result["document_type"] = document_type
					elif callable(self._document_scan_fn):
						# Fallback: use basic scan method
						scan_result = self._document_scan_fn(frame)
						if scan_result:
							scan_result["intent"] = intent
							scan_result["document_type"] = document_type
					else:
						scan_result = {"blocks": [], "summary": "Document scan adapter not properly initialized"}
					
					results[step.id] = scan_result if scan_result else {"blocks": [], "summary": "Document scan failed"}
			elif step.tool == "face_recognition":
				if self._face_recognition is None:
					results[step.id] = {"recognized_persons": []}
				else:
					frame = frame_provider() if frame_provider else None
					prev_step_result = results.get(step.deps[0]) if step.deps else {}
					person_objects = prev_step_result.get("objects", [])
					# Filter to only person objects
					person_objects = [obj for obj in person_objects if obj.get("class", "").lower() == "person"]
					
					if frame is not None and person_objects:
						try:
							recognized_persons = self._face_recognition.recognize_person_objects(frame, person_objects)
							results[step.id] = {"recognized_persons": recognized_persons}
							print(f"[DEBUG] Face recognition step completed: {len(recognized_persons)} persons processed")
						except Exception as e:
							print(f"[WARNING] Face recognition step failed: {e}")
							results[step.id] = {"recognized_persons": person_objects}
					else:
						results[step.id] = {"recognized_persons": person_objects}
			elif step.tool == "chatbot":
				# Aggregate previous data
				tool_data = {k: v for k, v in results.items()}
				print(f"[DEBUG] Chatbot step - processing {len(tool_data)} result items")
				print(f"[DEBUG] Available result keys: {list(tool_data.keys())}")
				
				# Get the query from step params (needed for person name extraction)
				current_query = step.params.get("query", "") if step.params else ""
				
				# Check if this is a document scanning query (skip object detection context)
				is_document_query = False
				# Initialize defaults to avoid unbound variables in fallback branches
				enhanced_context_parts = []
				objects_found_count = 0
				objects_found_in_current = False
				historical_info = None
				scene_description = None
				# Cached objects from continuous detection loop (may be used as fallback)
				cached_objects = context.cache_get("last_objects")
				for step_id, step_result in tool_data.items():
					if isinstance(step_result, dict):
						if "full_text" in step_result or ("blocks" in step_result and step_result.get("blocks")):
							is_document_query = True
							print(f"[DEBUG] Document scan detected - skipping object detection context")
							break
				
				if is_document_query:
					# For document queries, skip object detection context building
					# The chatbot adapter will handle document text directly
					enhanced_context = None  # Let chatbot adapter handle document text from tool_data
				else:
					# Build object detection context (existing logic)
					for step_id, step_result in tool_data.items():
						print(f"[DEBUG]   Step {step_id}: {list(step_result.keys()) if isinstance(step_result, dict) else type(step_result)}")
						if isinstance(step_result, dict) and "objects" in step_result:
							print(f"[DEBUG]     Objects in {step_id}: {len(step_result.get('objects', []))}")
					
					# Also check if we can get objects from context cache (fallback)
					cached_objects = cached_objects or context.cache_get("last_objects")
					if cached_objects:
						print(f"[DEBUG] Found {len(cached_objects)} cached objects in context")
					
					# Build user-friendly context for object detection queries
					objects_searched_for = []
					# objects_found_count, enhanced_context_parts, historical_info, objects_found_in_current,
					# and scene_description were pre-initialized above to avoid UnboundLocalError
					
					# Extract object detection results and format them naturally
					# First, collect objects and recognized persons separately
					recognized_persons = []
					objects_from_detection = []
					historical_detection_info = None  # Store historical detection info if found
					
					for step_id, step_result in results.items():
						if isinstance(step_result, dict):
							# Check for face recognition results
							if "recognized_persons" in step_result:
								recognized_persons = step_result.get("recognized_persons", [])
								print(f"[DEBUG] Found {len(recognized_persons)} recognized persons from face recognition")
							
							# Check for scene description (from Pixtral scene analysis)
							if "scene_description" in step_result and step_result.get("scene_mode"):
								scene_description = step_result.get("scene_description")
								print(f"[DEBUG] Found scene description from Pixtral: {len(scene_description)} chars")
								continue  # Skip object processing for scene mode
							
							# Check for enhanced_detections first (from Pixtral analysis) - these have visual descriptions
							if "enhanced_detections" in step_result:
								enhanced_detections = step_result.get("enhanced_detections", [])
								objects_from_detection = enhanced_detections  # Use enhanced detections (have Pixtral descriptions)
								objects_found_count = len(objects_from_detection)
								print(f"[DEBUG] Found {objects_found_count} enhanced detections (with Pixtral descriptions) from step {step_id}")
								if objects_found_count > 0:
									print(f"[DEBUG] Enhanced object classes: {[obj.get('class', 'unknown') for obj in objects_from_detection[:5]]}")
									# Check if Pixtral descriptions are present
									descriptions_count = sum(1 for obj in objects_from_detection if obj.get("pixtral_description"))
									print(f"[DEBUG] Objects with Pixtral descriptions: {descriptions_count}/{objects_found_count}")
							elif "objects" in step_result:
								objects_from_detection = step_result.get("objects", [])
								objects_found_count = len(objects_from_detection)
								print(f"[DEBUG] Found {objects_found_count} objects in step_result from step {step_id}")
								if objects_found_count > 0:
									print(f"[DEBUG] Object classes detected: {[obj.get('class', 'unknown') for obj in objects_from_detection[:5]]}")
							
							# Check if we were searching for specific classes
							if step_result.get("target_classes"):
								target = step_result.get("target_classes")
								if isinstance(target, list) and target:
									objects_searched_for = target
								elif isinstance(target, str):
									objects_searched_for = [target]
							
							# Check for historical detection info (store it for later use)
							if "historical_found" in step_result:
								historical_detection_info = step_result.get("historical_found")
								print(f"[DEBUG] Found historical detection info in step_result from step {step_id}: {historical_detection_info}")
					
					# Merge recognized persons into objects list
					objects = objects_from_detection.copy() if objects_from_detection else []
					recognized_names = []  # Track which names were actually recognized
					if recognized_persons:
						print(f"[DEBUG] Merging {len(recognized_persons)} recognized persons into objects list")
						# Update person objects with recognized names
						for rec_person in recognized_persons:
							rec_name = rec_person.get("recognized_name")
							if rec_name and rec_name.lower() != "unknown":
								recognized_names.append(rec_name.lower())
								# Find corresponding person object in objects list and update it
								rec_bbox = rec_person.get("bbox", [])
								for i, obj in enumerate(objects):
									if obj.get("class", "").lower() == "person":
										# Match by bbox position
										obj_bbox = obj.get("bbox", [])
										if (len(obj_bbox) == 4 and len(rec_bbox) == 4 and
										    abs(obj_bbox[0] - rec_bbox[0]) < 20):
											# Update object with recognized name
											objects[i] = rec_person.copy()  # Replace with recognized person data
											objects[i]["class"] = rec_name
											print(f"[DEBUG] Updated person object to recognized name: {rec_name}")
											break
					
					# Check if query is asking about a specific person by name
					# Extract person name from query if present
					query_lower = current_query.lower() if current_query else ""
					person_name_in_query = None
					if query_lower:
						# Common patterns for person queries
						person_query_patterns = [
							r"is\s+([a-z]+)\s+around",
							r"where\s+is\s+([a-z]+)",
							r"do\s+you\s+see\s+([a-z]+)",
							r"can\s+you\s+see\s+([a-z]+)",
							r"check\s+if\s+([a-z]+)\s+is",
							r"is\s+([a-z]+)\s+here",
							r"([a-z]+)\s+is\s+around",
						]
						for pattern in person_query_patterns:
							match = re.search(pattern, query_lower)
							if match:
								potential_name = match.group(1).strip()
								# Skip common words that aren't names
								if potential_name not in ["there", "here", "around", "where", "what", "who", "this", "that", "the", "a", "an"]:
									person_name_in_query = potential_name
									print(f"[DEBUG] Detected person name in query: '{person_name_in_query}'")
									break
					
					# If query asks about a specific person, check if they were recognized
					if person_name_in_query:
						person_found = False
						person_recognized = False
						for obj in objects:
							if obj.get("class", "").lower() == "person" or obj.get("class", "").lower() in recognized_names:
								person_found = True
								obj_class_lower = obj.get("class", "").lower()
								# Check if this person matches the name in query
								if obj_class_lower == person_name_in_query or person_name_in_query in obj_class_lower or obj_class_lower in person_name_in_query:
									person_recognized = True
									print(f"[DEBUG] Person '{person_name_in_query}' was recognized as '{obj_class_lower}'")
									break
						
						if person_found and not person_recognized:
							# Person detected but NOT recognized as the name in query
							enhanced_context_parts.append(f"IMPORTANT: I see a person, but face recognition did NOT identify them as '{person_name_in_query}'. The person was not recognized or is not enrolled in the system.")
							print(f"[DEBUG] Person query mismatch: Query asks for '{person_name_in_query}' but face recognition did not match that name")
						elif not person_found:
							# No person detected at all
							enhanced_context_parts.append(f"I do not see any person in the current view, so I cannot check if '{person_name_in_query}' is around.")
							print(f"[DEBUG] Person query: No person detected, cannot check for '{person_name_in_query}'")
					
					# Process merged objects
					if objects:
						objects_found_in_current = True  # Mark that objects were found in current detection
						# Objects were found - list them naturally (no timing info for current detections)
						obj_names = [obj.get("class", "object") for obj in objects[:5]]  # Limit to 5
						if len(objects) > 5:
							enhanced_context_parts.append(f"I can see {len(objects)} objects in the current view, including: {', '.join(obj_names)}.")
						else:
							enhanced_context_parts.append(f"I can see: {', '.join(obj_names)}.")
						
						# Add distance/direction info for found objects (with human-friendly directions)
						details = []
						for obj in objects[:3]:  # Limit details to 3 objects
							name = obj.get("class", "object")
							dist = obj.get("approxDistance", 0)
							angle_x = obj.get("angle_x", 0)
							angle_y = obj.get("angle_y", 0)
							if dist > 0:
								# Use human-friendly direction formatting
								location_str = format_location_response(name, dist, angle_x, angle_y)
								details.append(location_str)
						if details:
							enhanced_context_parts.append(" ".join(details) + ".")
						
						# Check if we were searching for a specific object and found it
						if objects_searched_for:
							target_lower = str(objects_searched_for[0]).lower() if objects_searched_for else ""
							for obj in objects:
								obj_class = str(obj.get("class", "")).lower()
								recognized_name = obj.get("recognized_name")
								if _fuzzy_match_class(target_lower, obj_class, recognized_name=recognized_name):
									# We found the target object in current detection - mark it
									objects_found_in_current = True
									print(f"[DEBUG] Target '{target_lower}' found in current objects as '{obj_class}' (recognized_name='{recognized_name}')")
									break
					else:
						# No objects found in step results - check if we have cached objects before saying "didn't detect"
						# Only add "didn't detect" message if we also don't have cached objects
						if not (cached_objects and len(cached_objects) > 0):
							# No objects in results AND no cached objects - use simple, clear message
							if objects_searched_for:
								searched = objects_searched_for[0] if objects_searched_for else "objects"
								# Don't mention "memory" - just say what was searched
								enhanced_context_parts.append(f"I searched for {searched} but didn't find it in the current view.")
							else:
								enhanced_context_parts.append("I looked around but didn't detect any objects in the current view.")
						else:
							# We have cached objects, so don't add "didn't detect" message yet
							# The cached objects will be added in the fallback section below
							print(f"[DEBUG] No objects in step results, but cached objects exist - will use cached objects instead")
					
					# Check for historical detection info (now using stored variable from loop)
					if historical_detection_info:
						hist = historical_detection_info
						print(f"[DEBUG] Processing historical detection info: {hist}")
						age_sec = hist.get('age_seconds', 0)
						
						# Only report historical if object was NOT found in current detection
						# AND historical detection is within memory window (2 minutes)
						# Note: Objects older than 2 minutes are automatically deleted from memory
						HISTORICAL_THRESHOLD_SEC = 120.0  # Report if found in memory (up to 2 minutes)
						
						if not objects_found_in_current and age_sec <= HISTORICAL_THRESHOLD_SEC:
							age_sec_int = int(age_sec)
							age_str = f"{age_sec_int} second{'s' if age_sec_int != 1 else ''}" if age_sec_int < 60 else f"{age_sec_int // 60} minute{'s' if age_sec_int // 60 != 1 else ''}"
							
							# Extract angle information from historical detection
							hist_angle_x = hist.get("angle_x", 0)
							hist_angle_y = hist.get("angle_y", 0)
							hist_distance = hist.get("distance", 0)
							
							# Format historical location with human-friendly directions
							if hist_distance > 0 and (hist_angle_x != 0 or hist_angle_y != 0):
								# Use format_location_response for historical objects too
								direction = angles_to_direction(hist_angle_x, hist_angle_y)
								historical_info = f"I don't see {hist['class']} right now, but I detected one about {age_str} ago. It was about {hist_distance:.1f} meters away, {direction}."
							else:
								# Fallback if angles not available
								historical_info = f"I don't see {hist['class']} right now, but I detected one about {age_str} ago at approximately {hist_distance:.1f} meters away."
							
							print(f"[DEBUG] Historical info formatted: {historical_info}")
						else:
							if objects_found_in_current:
								print(f"[DEBUG] Skipping historical info - object found in current detection")
							else:
								print(f"[DEBUG] Skipping historical info - too old ({age_sec:.1f}s > {HISTORICAL_THRESHOLD_SEC}s threshold)")
							historical_info = None
				
				# Fallback: If no objects found in results, check cached objects from continuous detection loop
				if objects_found_count == 0 and cached_objects and len(cached_objects) > 0:
					print(f"[DEBUG] No objects in results, but found {len(cached_objects)} objects in cache - using cached objects")
					objects_found_count = len(cached_objects)
					objects_found_in_current = True
					obj_names = [obj.get("class", "object") for obj in cached_objects[:5]]
					if len(cached_objects) > 5:
						enhanced_context_parts.append(f"I can see {len(cached_objects)} objects in the current view, including: {', '.join(obj_names)}.")
					else:
						enhanced_context_parts.append(f"I can see: {', '.join(obj_names)}.")
					details = []
					for obj in cached_objects[:3]:
						name = obj.get("class", "object")
						dist = obj.get("approxDistance", 0)
						angle_x = obj.get("angle_x", 0)
						angle_y = obj.get("angle_y", 0)
						if dist > 0:
							# Use human-friendly direction formatting
							location_str = format_location_response(name, dist, angle_x, angle_y)
							details.append(location_str)
					if details:
						enhanced_context_parts.append(" ".join(details) + ".")
				
				# Another fallback: Check detection history from continuous detection loop (if cache was empty)
				if objects_found_count == 0:
					recent_snapshots = context.get_recent_objects(max_age_sec=2.0)  # Get snapshots from last 2 seconds
					if recent_snapshots:
						# Get most recent snapshot
						latest_snapshot = recent_snapshots[0] if isinstance(recent_snapshots, list) and recent_snapshots else recent_snapshots
						recent_objects = latest_snapshot.get("objects", []) if isinstance(latest_snapshot, dict) else []
						if recent_objects:
							print(f"[DEBUG] No objects in results or cache, but found {len(recent_objects)} objects in recent snapshot - using recent objects")
							objects_found_count = len(recent_objects)
							objects_found_in_current = True
							obj_names = [obj.get("class", "object") for obj in recent_objects[:5]]
							if len(recent_objects) > 5:
								enhanced_context_parts.append(f"I can see {len(recent_objects)} objects in the current view, including: {', '.join(obj_names)}.")
							else:
								enhanced_context_parts.append(f"I can see: {', '.join(obj_names)}.")
							details = []
							for obj in recent_objects[:3]:
								name = obj.get("class", "object")
								dist = obj.get("approxDistance", 0)
								angle_x = obj.get("angle_x", 0)
								angle_y = obj.get("angle_y", 0)
								if dist > 0:
									# Use human-friendly direction formatting
									location_str = format_location_response(name, dist, angle_x, angle_y)
									details.append(location_str)
							if details:
								enhanced_context_parts.append(" ".join(details) + ".")
				
				# Only add temporal memory summary if objects were NOT found in current detection
				# (to avoid redundant timing info for currently visible objects)
				# NOTE: Temporal memory is only used for historical context, not for "not found" messages
				# We don't add it here to avoid confusing the LLM with "memory" references
				# Historical info is handled separately via historical_found flag
				
				# If we have scene description, use it directly
				if scene_description:
					enhanced_context = scene_description
					print(f"[DEBUG] Using Pixtral scene description as context")
				else:
					# Combine context parts for object queries
					enhanced_context = "\n".join(enhanced_context_parts) if enhanced_context_parts else "No visual information available."
					
					# If no objects were found and no context was built, make it clear
					if not enhanced_context_parts and not historical_info:
						enhanced_context = "I looked around but didn't detect any objects in the current view. No objects visible."
					
					# Add historical info if present
					if historical_info:
						print(f"[DEBUG] Adding historical_info to context: {historical_info}")
						enhanced_context = f"{historical_info}\n{enhanced_context}"
					else:
						print(f"[DEBUG] No historical_info to add (historical_info={historical_info})")
				
				print(f"[DEBUG] Final context being passed to chatbot:\n{enhanced_context}")
				print(f"[DEBUG] objects_found_count={objects_found_count}, objects_found_in_current={objects_found_in_current}")
				answer = self._chatbot.answer(step.params.get("query", ""), tool_data=tool_data, context=enhanced_context)
				results[step.id] = {"answer": answer}
			else:
				results[step.id] = {"status": "skipped"}
		return results


class ResponseSynthesizer:
	def compose(self, results: Dict[str, Any]) -> str:
		# Prefer chatbot answer if present
		for key, val in results.items():
			if isinstance(val, dict) and "answer" in val:
				return val["answer"]
		return "Done."


class AgentController:
	def __init__(self, event_bus: EventBus, registry: ToolRegistry, object_detector: ObjectDetectorAdapter, scene_analysis: SceneAnalysisAdapter, chatbot: ChatbotAdapter, document_scan_fn: Optional[Callable[..., Dict[str, Any]]] = None, pixtral_analysis: Optional[Any] = None, face_recognition: Optional[Any] = None, frame_provider: Optional[Callable[[], Any]] = None) -> None:
		self._bus = event_bus
		self._registry = registry
		# Enable clarification by default for all object-finding queries
		enable_clarification_llm = os.environ.get("AGENT_ENABLE_CLARIFICATION", "1") in ("1", "true", "True")
		self._clarifier = ClarificationChecker(enable_llm=enable_clarification_llm)
		self._planner = Planner(registry)
		self._context = ContextWindow()
		self._engine = ExecutionEngine(object_detector, scene_analysis, chatbot, document_scan=document_scan_fn, pixtral_analysis=pixtral_analysis, face_recognition=face_recognition)
		self._synth = ResponseSynthesizer()
		self._frame_provider = frame_provider
		# Store pending clarification: tracks most recent clarification request
		self._pending_clarification: Optional[Dict[str, Any]] = None
		asyncio.create_task(self._subscribe())

	async def _subscribe(self) -> None:
		async def on_transcript_ready(_topic: str, envelope) -> None:
			payload: schemas.TranscriptReady = envelope["payload"]
			query = payload.transcript
			request_id = envelope["request_id"]
			start_time = time.time()
			print(f"[agent] Processing transcript: \"{query}\"")
			
			# Initialize unified logging for this query
			query_id = None
			if UNIFIED_LOGGING_AVAILABLE:
				try:
					logger_integration = get_logger_integration()
					query_id = logger_integration.log_query_start(query, "audio")
				except:
					pass
			
			# Start performance measurement for this query
			try:
				from core.infrastructure.performance_monitor import get_performance_monitor
				perf_monitor = get_performance_monitor()
				# Classify query type based on content
				query_type = "Unknown"
				query_lower = query.lower() if query else ""
				if any(word in query_lower for word in ["where", "find", "locate"]):
					query_type = "Object Location"
				elif any(word in query_lower for word in ["describe", "what", "see"]):
					query_type = "Scene Description" 
				elif any(word in query_lower for word in ["who", "person"]):
					query_type = "Person Query"
				elif any(word in query_lower for word in ["read", "document", "text"]):
					query_type = "Document Reading"
				
				perf_monitor.start_query_measurement(request_id, query, query_type)
			except:
				pass  # Performance monitoring is optional
			
			# Start experimental metrics tracking
			try:
				from core.metrics.agent_bridge import get_agent_bridge
				bridge = get_agent_bridge()
				if bridge:
					bridge.start_query_tracking(request_id, query, query_type)
			except:
				pass  # Experimental metrics is optional
			
			# Check if transcript seems empty or didn't capture anything
			query_clean = query.strip() if query else ""
			is_empty_transcript = False
			
			# SPECIAL HANDLING: Check for "thank you" patterns
			# Remove punctuation and normalize
			query_normalized = re.sub(r'[^\w\s]', '', query_clean.lower()).strip()
			query_words = query_normalized.split()
			
			# Check if it's EXACTLY "thank you" (just those two words, any case/punctuation)
			is_exactly_thank_you = (
				len(query_words) == 2 and 
				query_words[0] == "thank" and 
				query_words[1] == "you"
			)
			
			# Check if it contains "thank you" but has other words
			contains_thank_you = "thank" in query_normalized and "you" in query_normalized
			has_other_words = len(query_words) > 2 or (len(query_words) == 2 and not is_exactly_thank_you)
			
			if is_exactly_thank_you:
				# EXACTLY "thank you" → treat as white noise, ask "Did you ask anything?"
				print(f"[agent] Exactly 'thank you' detected (likely white noise) - treating as empty transcript")
				is_empty_transcript = True
			elif contains_thank_you and has_other_words:
				# "Thank you" with other words (e.g., "thank you for helping", "i said thank you")
				# → Hardcode response to "You're welcome!" and return immediately
				print(f"[agent] 'Thank you' with other words detected - responding with hardcoded 'You're welcome!'")
				self._context.add_turn("user", query_clean)
				self._context.add_turn("assistant", "You're welcome!")
				await self._bus.publish("ResponseReady", schemas.ResponseReady(request_id=request_id, answer="You're welcome!"), request_id=request_id)
				return
			
			# More strict empty detection - catch truly empty, very short, or generic phrases that indicate no real speech
			if not query_clean or len(query_clean) == 0:
				is_empty_transcript = True
			elif len(query_clean) < 2:  # Very short text (likely noise)
				is_empty_transcript = True
			elif query_clean.lower() in [".", "..", "...", "[silence]", "[no speech]", "hmm", "um", "uh"]:
				is_empty_transcript = True
			# Check for other generic phrases that Whisper sometimes returns when there's no actual speech
			# (but skip "thank you" since we already handled it above)
			generic_phrases = [
				"you are welcome", "you're welcome", "welcome", 
				"these objects", "these", "these these", "these these these",
				"thanks",  # Note: "thank you" is handled separately above
				"okay", "ok", "all right", "alright",
				"yes", "no", "yeah", "nope"
			]
			
			query_lower_clean = query_clean.lower().strip('.,!?;:')
			if query_lower_clean in generic_phrases:
				turns = self._context.get_snapshot().get("turns", [])
				
				# Check if there was a recent assistant response with actual information
				# (not just a question or clarification)
				has_recent_informative_response = False
				if len(turns) > 0:
					# Look at last assistant turn
					last_turn = turns[-1]
					if last_turn.get("role") == "assistant":
						last_content = last_turn.get("content", "").lower()
						# Check if last response had actual information (not just questions)
						informative_phrases = [
							"i can see", "i see", "there is", "there are", "about", "meters away",
							"i don't see", "detected", "found", "ready when you need",
							"front", "left", "right", "behind", "away", "distance"
						]
						# If last response has informative content, generic phrase is likely valid
						if any(phrase in last_content for phrase in informative_phrases):
							has_recent_informative_response = True
							print(f"[agent] Generic phrase '{query_clean}' detected but recent informative response found - treating as valid")
				
				# Check if it's a response to a clarification question
				is_response_to_question = len(turns) > 0 and any(
					phrase in turns[-1].get("content", "").lower() 
					for phrase in ["did you", "do you", "are you", "can you", "question", "please repeat"]
				)
				
				# If there's no recent informative response AND it's not a response to a question,
				# treat as empty (likely false positive from white noise/hazy audio)
				if not has_recent_informative_response and not is_response_to_question:
					is_empty_transcript = True
					print(f"[agent] Generic phrase '{query_clean}' detected without recent informative context - treating as empty transcript (likely white noise/hazy audio)")
			
			if is_empty_transcript:
				# Check if we already asked about empty transcript recently - avoid loops
				turns = self._context.get_snapshot().get("turns", [])
				recently_asked_empty = False
				if len(turns) >= 2:
					# Check last 2 assistant turns for empty transcript question
					for turn in turns[-2:]:
						if turn.get("role") == "assistant" and ("did you ask anything" in turn.get("content", "").lower() or "did you say anything" in turn.get("content", "").lower()):
							recently_asked_empty = True
							break
				
				if recently_asked_empty:
					# Already asked recently - don't ask again, just ignore this empty transcript
					print(f"[agent] Empty transcript detected but already asked recently. Ignoring.")
					# Reset FSM state to IDLE so user can try recording again
					await self._bus.publish("EmptyTranscriptIgnored", schemas.EmptyTranscriptIgnored(request_id=request_id), request_id=request_id)
					return
				
				print(f"[agent] Empty or unclear transcript detected. Asking user for confirmation...")
				# Ask if user asked anything
				clar_event = schemas.ClarificationNeeded(
					request_id=request_id,
					question="Did you ask anything?",
					original_query="",
					ambiguity_reason="empty_or_unclear_transcript"
				)
				# Add clarification question to context
				self._context.add_turn("assistant", "Did you ask anything?")
				await self._bus.publish("ClarificationNeeded", clar_event, request_id=request_id)
				# Store pending clarification with special flag for empty transcript handling
				self._pending_clarification = {
					"original_query": "",
					"clarification": {"question": "Did you ask anything?", "reason": "empty_or_unclear_transcript"},
					"is_empty_transcript": True
				}
				return
			
			# Check if last assistant turn was a clarification question
			turns = self._context.get_snapshot().get("turns", [])
			is_clarification_response = False
			if len(turns) >= 2 and turns[-1].get("role") == "assistant":
				last_assistant = turns[-1].get("content", "")
				if any(phrase in last_assistant.lower() for phrase in ["do you mean", "did you mean", "are you looking for", "do you want me", "did you say anything"]):
					is_clarification_response = True
			
			# Check if this is a response to a pending clarification (including empty transcript check)
			# BUT first check if it looks like a valid new query (to avoid false clarification detection)
			query_lower = query.lower().strip()
			query_words = query_lower.split()
			looks_like_new_query = (
				len(query_words) > 3 or 
				any(keyword in query_lower for keyword in 
					["where", "what", "is there", "do you see", "can you", "describe", "find", "locate", "read", "scan", "analyze", "who", "when", "how"])
			)
			
			# If we have pending clarification for empty transcript, check response FIRST
			if self._pending_clarification and self._pending_clarification.get("is_empty_transcript"):
				pending = self._pending_clarification
				response_lower = query.lower().strip()
				response_words = response_lower.split()
				
				# FIRST: Check if user said "no" (they didn't say anything) - check this BEFORE checking for valid queries
				# This includes various ways of saying "no" and also generic phrases that are likely false positives
				no_responses = [
					"no", "nope", "nothing", "i didn't", "i didn't say", "i said nothing", 
					"no i didn't", "i didn't say anything", "no i didn't say anything",
					"i didn't ask", "no i didn't ask", "i didn't ask anything"
				]
				
				# Also check for generic phrases that are common false positives from hazy audio
				# These should be treated as "no" when responding to "Did you ask anything?"
				generic_false_phrases = [
					"thank you for watching", "thanks for watching", "thank you for", 
					"you're welcome", "you are welcome", "welcome",
					"these objects", "these", "these these"
				]
				
				is_no_response = (
					any(word in response_lower for word in no_responses) or
					any(phrase in response_lower for phrase in generic_false_phrases)
				)
				
				if is_no_response:
					# User didn't say anything (or it's a false positive) - acknowledge and return to idle
					print(f"[agent] User confirmed they didn't say anything (or false positive detected). Returning to idle state.")
					self._context.add_turn("user", response_lower)
					self._context.add_turn("assistant", "Okay, I'm ready when you need me.")
					# Clear pending clarification
					self._pending_clarification = None
					# Acknowledge and return to idle
					await self._bus.publish("ResponseReady", schemas.ResponseReady(request_id=request_id, answer="Okay, I'm ready when you need me."), request_id=request_id)
					# Reset FSM to IDLE
					await self._bus.publish("EmptyTranscriptIgnored", schemas.EmptyTranscriptIgnored(request_id=request_id), request_id=request_id)
					return
				
				# SECOND: Check if user said "yes" (they want to repeat)
				if any(word in response_lower for word in ["yes", "yeah", "yep", "correct", "right", "i did", "i said", "i did say"]):
					# User wants to repeat - ask them to repeat and wait for new recording
					print(f"[agent] User confirmed they said something. Asking them to repeat.")
					self._context.add_turn("user", response_lower)
					self._context.add_turn("assistant", "I didn't catch that. Please repeat what you said.")
					# Clear pending clarification
					self._pending_clarification = None
					# Tell user to repeat and return to idle so they can record again
					await self._bus.publish("ResponseReady", schemas.ResponseReady(request_id=request_id, answer="I didn't catch that. Please repeat what you said."), request_id=request_id)
					# Reset FSM to IDLE so user can record again
					await self._bus.publish("EmptyTranscriptIgnored", schemas.EmptyTranscriptIgnored(request_id=request_id), request_id=request_id)
					return
				
				# THIRD: Check if response looks like a valid query (not a yes/no answer)
				# Only treat as new query if it's clearly a real query, not a generic phrase
				if looks_like_new_query and not any(phrase in response_lower for phrase in generic_false_phrases):
					# This looks like a valid query, not a response to "did you say anything"
					# Clear empty transcript pending and treat as new query
					print(f"[agent] Empty transcript clarification response looks like a valid query. Treating as new query.")
					self._pending_clarification = None
					# Process as new query below (skip clarification handling)
				else:
					# Short unclear response - might be another empty transcript or unclear answer
					# Check if it's another empty transcript
					query_clean_check = query.strip() if query else ""
					if not query_clean_check or len(query_clean_check) < 3:
						# Another empty transcript - just ignore and clear state to prevent loop
						print(f"[agent] Another empty transcript after clarification. Clearing state to prevent loop.")
						self._pending_clarification = None
						# Reset FSM state to IDLE so user can try recording again
						await self._bus.publish("EmptyTranscriptIgnored", schemas.EmptyTranscriptIgnored(request_id=request_id), request_id=request_id)
						return
					# Otherwise, treat as unclear response but don't loop - clear state
					print(f"[agent] Unclear response to empty transcript question. Clearing state.")
					self._pending_clarification = None
					# Reset FSM state to IDLE so user can try recording again
					await self._bus.publish("EmptyTranscriptIgnored", schemas.EmptyTranscriptIgnored(request_id=request_id), request_id=request_id)
					return
			
			# Handle other clarifications (non-empty-transcript)
			if (self._pending_clarification or is_clarification_response) and not looks_like_new_query:
				if self._pending_clarification:
					pending = self._pending_clarification
					self._pending_clarification = None
				else:
					# Infer from conversation
					pending = {"original_query": turns[-2].get("content", "") if len(turns) >= 2 else query, "clarification": {"question": turns[-1].get("content", "")}}
				
				original = pending.get("original_query", query)
				
				# Regular clarification response handling
				confirmed_query = self._parse_clarification_response(query, original)
				if confirmed_query:
					# Proceed with confirmed query
					self._context.add_turn("user", confirmed_query)
					plan = self._planner.plan(confirmed_query, self._context.get_snapshot())
					plan_event = schemas.PlanReady(request_id=request_id, plan={"id": plan.id, "steps": [s.__dict__ for s in plan.steps]})
					await self._bus.publish("PlanReady", plan_event, request_id=request_id)
					plan_start = time.time()
					results = await self._engine.run(plan, self._context, frame_provider=self._frame_provider)
					exec_time = time.time() - plan_start
					if exec_time > 0.1:
						print(f"[agent] Execution took {exec_time:.2f}s")
					answer = self._synth.compose(results)
					self._context.add_turn("assistant", answer)
					total_time = time.time() - start_time
					print(f"[agent] Response ready: \"{answer}\" (total: {total_time:.2f}s)")
					await self._bus.publish("ResponseReady", schemas.ResponseReady(request_id=request_id, answer=answer), request_id=request_id)
				else:
					# User said no or unclear - ask again
					clar = pending.get("clarification", {})
					clar_event = schemas.ClarificationNeeded(request_id=request_id, question=f"I didn't understand. {clar.get('question', 'Could you clarify?')}", original_query=original, ambiguity_reason=clar.get("reason", ""))
					await self._bus.publish("ClarificationNeeded", clar_event, request_id=request_id)
					self._pending_clarification = pending  # Keep pending
				return
			
			# New query - check for clarification needs
			self._context.add_turn("user", query)
			clarification = self._clarifier.check_and_clarify(query)
			
			if clarification:
				# Need clarification - ask user
				clar_event = schemas.ClarificationNeeded(request_id=request_id, question=clarification["question"], original_query=query, ambiguity_reason=clarification["reason"])
				# Add clarification question to context (so it's tracked in conversation)
				self._context.add_turn("assistant", clarification["question"])
				await self._bus.publish("ClarificationNeeded", clar_event, request_id=request_id)
				self._pending_clarification = {"original_query": query, "clarification": clarification}
				# Don't proceed yet - wait for user response
				return
			
			# Query is clear - proceed directly
			plan_start = time.time()
			plan = self._planner.plan(query, self._context.get_snapshot())
			plan_time = time.time() - plan_start
			
			# Record planning metrics if performance monitor is available
			try:
				from core.infrastructure.performance_monitor import get_performance_monitor
				perf_monitor = get_performance_monitor()
				# Check if planner has cache information
				cache_hit = hasattr(self._planner, '_last_cache_hit') and getattr(self._planner, '_last_cache_hit', False)
				fast_path = hasattr(self._planner, '_last_fast_path') and getattr(self._planner, '_last_fast_path', False)
				perf_monitor.record_planning_metrics(request_id, plan_time * 1000, cache_hit, fast_path)
			except:
				pass  # Performance monitoring is optional
			
			# Record planning in experimental metrics
			try:
				from core.metrics.agent_bridge import get_agent_bridge
				bridge = get_agent_bridge()
				if bridge:
					bridge.mark_planning_start(request_id)
					# If this was an LLM-based plan (not fast path), mark LLM start
					cache_hit = hasattr(self._planner, '_last_cache_hit') and getattr(self._planner, '_last_cache_hit', False)
					fast_path = hasattr(self._planner, '_last_fast_path') and getattr(self._planner, '_last_fast_path', False)
					if not fast_path:  # LLM was used for planning
						bridge.mark_llm_start(request_id, cache_hit)
			except:
				pass  # Experimental metrics is optional
			
			if plan_time > 0.1:
				print(f"[agent] Planning took {plan_time:.2f}s")
			print(f"[DEBUG] Plan created with {len(plan.steps)} steps:")
			for s in plan.steps:
				print(f"[DEBUG]   Step {s.id}: tool={s.tool}, params={s.params}, deps={s.deps}")
			plan_event = schemas.PlanReady(request_id=request_id, plan={"id": plan.id, "steps": [s.__dict__ for s in plan.steps]})
			await self._bus.publish("PlanReady", plan_event, request_id=request_id)
			exec_start = time.time()
			results = await self._engine.run(plan, self._context, frame_provider=self._frame_provider)
			exec_time = time.time() - exec_start
			
			# Record execution metrics if performance monitor is available
			try:
				from core.infrastructure.performance_monitor import get_performance_monitor
				perf_monitor = get_performance_monitor()
				perf_monitor.record_execution_metrics(request_id, exec_time * 1000)
			except:
				pass  # Performance monitoring is optional
			
			if exec_time > 0.1:
				print(f"[agent] Execution took {exec_time:.2f}s")
			print(f"[DEBUG] Execution results: {list(results.keys())}")
			for step_id, result in results.items():
				if isinstance(result, dict) and "objects" in result:
					print(f"[DEBUG]   Step {step_id}: {len(result.get('objects', []))} objects")
			answer = self._synth.compose(results)
			self._context.add_turn("assistant", answer)
			total_time = time.time() - start_time
			
			# Record final query metrics if performance monitor is available
			try:
				from core.infrastructure.performance_monitor import get_performance_monitor
				perf_monitor = get_performance_monitor()
				perf_monitor.finish_query_measurement(request_id, total_time * 1000)
			except:
				pass  # Performance monitoring is optional
			
			# End experimental metrics tracking
			try:
				from core.metrics.agent_bridge import get_agent_bridge
				bridge = get_agent_bridge()
				if bridge:
					# Calculate a simple accuracy score based on response length and success
					accuracy_score = 0.9 if len(answer) > 10 and "error" not in answer.lower() else 0.5
					bridge.end_query_tracking(request_id, success=True, accuracy_score=accuracy_score)
			except:
				pass  # Experimental metrics is optional
			
			print(f"[agent] Response ready: \"{answer}\" (total: {total_time:.2f}s)")
			
			# Log query completion for unified logging
			if UNIFIED_LOGGING_AVAILABLE and query_id:
				try:
					logger_integration = get_logger_integration()
					logger_integration.log_query_end(query_id, True, answer, total_time * 1000)
				except:
					pass
			
			await self._bus.publish("ResponseReady", schemas.ResponseReady(request_id=request_id, answer=answer), request_id=request_id)
		
		await self._bus.subscribe("TranscriptReady", on_transcript_ready)
	
	def _parse_clarification_response(self, response: str, original_query: str) -> Optional[str]:
		"""Parse user response to clarification (yes/no/confirmation). Returns confirmed query or None."""
		response_lower = response.lower().strip()
		# Check for affirmative responses
		if any(word in response_lower for word in ["yes", "yeah", "yep", "correct", "right", "that's right", "that is correct", "confirmed", "confirm"]):
			return original_query  # Use original query
		# Check for corrections ("no, I meant pens" or "pens" directly)
		if any(word in response_lower for word in ["no", "nope", "incorrect", "wrong"]):
			# User corrected - extract the correction
			# Simple heuristic: if response contains object names, use those
			return response  # Use the correction
		# If response is short and doesn't contain yes/no, assume it's a correction
		if len(response.split()) <= 5:
			return response
		return None

	@property
	def context(self) -> ContextWindow:
		return self._context
