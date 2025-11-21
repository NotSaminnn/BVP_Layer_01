from __future__ import annotations
from typing import Any, Dict, Optional
import os

from core.modules.vlm.mistral_chat import MistralChat  # type: ignore
from core.infrastructure.location_formatter import format_location_response  # type: ignore


class ChatbotAdapter:
	def __init__(self, api_key: Optional[str] = None, verbose: bool = False) -> None:
		api_key = api_key or os.environ.get("MISTRAL_API_KEY") or ""
		# Use lower max_tokens (80) for faster responses (reduced from 100)
		# Lower temperature (0.3) for faster, more deterministic responses
		self._chat = MistralChat(api_key=api_key, verbose=verbose, max_tokens=80, temperature=0.3)

	def answer(self, query: str, tool_data: Optional[Dict[str, Any]] = None, context: Optional[str] = None) -> str:
		# PRIORITY 1: Extract document scan text if available (prioritize full_text over summary)
		doc_text = None
		has_document_scan = False
		ocr_intent = None
		document_type = None
		document_analysis = None
		
		if tool_data:
			for step_id, step_result in tool_data.items():
				if isinstance(step_result, dict):
					# Check for document scan results - check multiple possible fields
					if "full_text" in step_result:
						doc_text = step_result["full_text"]
						has_document_scan = True
					elif "summary" in step_result and step_result.get("blocks"):
						# Reconstruct from blocks if full_text not available
						blocks = step_result.get("blocks", [])
						if blocks:
							doc_text = "\n".join([b.get("text", "") for b in blocks if b.get("text")])
							has_document_scan = True
					elif "text" in step_result:
						# OCR might return text directly
						doc_text = step_result["text"]
						has_document_scan = True
					
					# Extract intent and document type from scan results
					if "intent" in step_result:
						ocr_intent = step_result["intent"]
					if "document_type" in step_result:
						document_type = step_result["document_type"]
					if "analysis" in step_result:
						document_analysis = step_result["analysis"]
		
		# If document scan was performed, handle based on intent
		if has_document_scan:
			if not doc_text or len(doc_text.strip()) < 10:
				return "I scanned the document but couldn't find any readable text. Please ensure the document is clearly visible and well-lit."
			
			# Route based on intent
			intent = ocr_intent or "read"  # Default to "read" if intent not specified
			doc_type = document_type or "general"
			
			if intent == "read":
				# READ INTENT: Read text as-is (like main.py)
				query_lower = query.lower()
				if len(doc_text) <= 500:
					# Short text - return directly
					return f"Here's what I read: {doc_text}"
				else:
					# Longer text - format nicely using LLM
					enhanced_context = f"""Document/Text Content (extracted from image):
{doc_text[:4000]}"""
					system_prompt = """You are a helpful assistant that reads documents aloud.
Read the document content clearly and naturally. Preserve the structure and flow.
If the text is long, format it nicely for reading aloud with appropriate pauses."""
					return self._chat.chat(query, system_prompt=system_prompt, context=enhanced_context)
			
			elif intent == "summarize":
				# SUMMARIZE INTENT: Create summary using MistralChat
				enhanced_context = f"""Document/Text Content (extracted from image):
{doc_text[:4000]}"""
				system_prompt = """You are a helpful assistant that summarizes documents.
Create a concise summary (2-3 sentences) covering the main points.
Focus on key information: who, what, when, where, why.
Keep it natural and conversational."""
				return self._chat.chat(
					"Summarize this document", 
					system_prompt=system_prompt, 
					context=enhanced_context
				)
			
			elif intent == "analyze":
				# ANALYZE INTENT: Structured analysis based on document type
				if doc_type == "receipt":
					return self._analyze_receipt(doc_text, document_analysis, query)
				elif doc_type == "letter":
					return self._analyze_letter(doc_text, query)
				elif doc_type == "form":
					return self._analyze_form(doc_text, query)
				elif doc_type == "label":
					return self._analyze_label(doc_text, query)
				else:
					# General document analysis
					return self._analyze_document(doc_text, document_analysis, query)
			
			elif intent == "extract":
				# EXTRACT INTENT: Return structured format
				enhanced_context = f"""Document/Text Content (extracted from image):
{doc_text[:4000]}"""
				system_prompt = """You are a helpful assistant that extracts key information from documents.
Extract and organize the important information clearly.
Keep it structured and easy to understand."""
				return self._chat.chat(query, system_prompt=system_prompt, context=enhanced_context)
			
			else:
				# Fallback: treat as read
				return f"Here's what I read: {doc_text[:500]}"
		
		# If document scan was performed but no text found
		if has_document_scan and not doc_text:
			return "I scanned the document but couldn't find any readable text. Please ensure the document is clearly visible and well-lit."
		
		# For non-document queries, check if this is an object detection query
		# Determine if we're looking for objects that weren't found
		is_object_query = False
		objects_found = []
		objects_searched = []
		pixtral_descriptions = []
		
		if tool_data:
			for step_id, step_result in tool_data.items():
				if isinstance(step_result, dict):
					if "objects" in step_result:
						objects_found = step_result.get("objects", [])
						is_object_query = True
					# Check for Pixtral enhanced detections
					if "enhanced_detections" in step_result:
						for obj in step_result["enhanced_detections"]:
							if obj.get("pixtral_description"):
								pixtral_descriptions.append(f"- {obj.get('class', 'object')}: {obj['pixtral_description']}")
		
		# Use a user-friendly system prompt for object queries (shortened for speed)
		if context and is_object_query:
			# Add Pixtral descriptions to context if available
			if pixtral_descriptions:
				enhanced_context = f"{context}\n\nAI Descriptions:\n" + "\n".join(pixtral_descriptions)
			else:
				enhanced_context = context
			
			# Check if query is asking about appearance/description
			query_lower = query.lower()
			is_appearance_query = any(phrase in query_lower for phrase in [
				"how does", "how do", "what does", "what do", "look like", "look", 
				"describe", "what is", "tell me about", "how is"
			])
			
			# Optimized system prompt - prioritize Pixtral descriptions for appearance queries
			if pixtral_descriptions and is_appearance_query:
				# For appearance queries with Pixtral descriptions: focus on visual description
				system_prompt = """Blind person assistant. The user is asking about how objects LOOK.
IMPORTANT: Use the AI Descriptions provided to describe the visual appearance of objects.
Describe colors, shapes, sizes, textures, and other visual features from the AI Descriptions.
If location is mentioned, include it briefly, but focus on visual description.
Only report objects in context. Don't invent objects.
First person, 2-3 sentences. Natural and descriptive."""
			else:
				# For location queries: focus on location information
				system_prompt = """Blind person assistant. Use location from context (front/left/right/top/down). 
If object found: say location with distance. If not found: "I don't see any [object] right now."
For historical: "I don't see [object] now, but detected one X seconds ago."
CRITICAL FOR PERSON QUERIES: 
- If context says "face recognition did NOT identify them as [name]", you MUST say the person was NOT recognized as that name.
- Only use person names that are explicitly listed in the context (e.g., "I can see: reshad" means reshad was recognized).
- If context says "Person" (not a name), and query asks about a specific name, say "I see a person but I don't recognize them as [name]".
- NEVER assume a "Person" in context is the person mentioned in the query unless explicitly recognized.
Only report objects in context. Don't invent objects or names.
First person, 1-2 sentences. Simple and natural."""
			
			# Truncate context if too long to speed up LLM processing (but preserve Pixtral descriptions)
			max_context_length = 1500 if pixtral_descriptions else 1000  # Allow more context if Pixtral descriptions exist
			if len(enhanced_context) > max_context_length:
				# Try to preserve Pixtral descriptions even if truncating
				if pixtral_descriptions:
					# Keep Pixtral descriptions, truncate other context
					pixtral_section = "\n\nAI Descriptions:\n" + "\n".join(pixtral_descriptions)
					remaining_length = max_context_length - len(pixtral_section) - 50
					truncated_context = context[:remaining_length] + "..." if len(context) > remaining_length else context
					enhanced_context = f"{truncated_context}{pixtral_section}"
				else:
					enhanced_context = enhanced_context[:max_context_length] + "..."
				print(f"[DEBUG] Context truncated to {max_context_length} chars for faster processing")
			
			try:
				return self._chat.chat(f"Context: {enhanced_context}\nQ: {query}", system_prompt=system_prompt)
			except Exception as e:
				# If chat fails, return a user-friendly error message
				error_str = str(e)
				if "429" in error_str or "rate limit" in error_str.lower() or "capacity exceeded" in error_str.lower():
					return ("I'm experiencing high demand right now and can't process your request immediately. "
						   "Please wait a moment and try again.")
				return f"I encountered an error processing your request. Please try again."
		
		# For non-document queries without object data, use answer_with_context or regular chat
		if context:
			# Use a simple, user-friendly system prompt for general queries
			system_prompt = """You are a friendly assistant helping a blind person. Keep messages simple, natural, and clear. 
IMPORTANT: Only report objects that are explicitly listed in the context. Do NOT invent, assume, or guess objects that aren't in the context.
CRITICAL: If the context contains BOTH "I looked around but didn't detect any objects" AND "I can see X objects", prioritize the "I can see" information - objects ARE present.
Only say "I don't see any objects around you right now" if the context ONLY says "No visual information available" or "I looked around but didn't detect any objects" or "No objects visible" WITHOUT any "I can see" statements.
Do NOT make assumptions about room types or objects that aren't mentioned in the context.
If objects are listed, describe them based on what's actually in the context.
Keep answers under 2 sentences."""
			return self._chat.chat(f"Context: {context}\nQ: {query}", system_prompt=system_prompt)
		return self._chat.chat(query, system_prompt="You are a concise assistant helping a blind person. Only report what is explicitly provided. Do NOT invent objects. Keep messages simple and natural. Keep answers under 2 sentences.")
	
	def _analyze_receipt(self, text: str, analysis: Optional[str], query: str) -> str:
		"""Analyze receipt and extract structured information."""
		enhanced_context = f"""Receipt Text Content:
{text[:4000]}

{analysis or ''}"""
		system_prompt = """You are a helpful assistant analyzing a receipt.
Extract and present:
- Store name
- Date and time
- Items purchased (list each item with price if available)
- Subtotal, tax, and total
- Payment method if visible
Format it clearly and naturally for reading aloud."""
		return self._chat.chat(query, system_prompt=system_prompt, context=enhanced_context)
	
	def _analyze_letter(self, text: str, query: str) -> str:
		"""Analyze letter and extract key information."""
		enhanced_context = f"""Letter Text Content:
{text[:4000]}"""
		system_prompt = """You are a helpful assistant analyzing a letter.
Extract and present:
- Sender
- Recipient
- Date
- Subject/main topic
- Key points or requests
Format it clearly for reading aloud."""
		return self._chat.chat(query, system_prompt=system_prompt, context=enhanced_context)
	
	def _analyze_form(self, text: str, query: str) -> str:
		"""Analyze form and extract fields."""
		enhanced_context = f"""Form Text Content:
{text[:4000]}"""
		system_prompt = """You are a helpful assistant analyzing a form.
Extract and present:
- Form type/name
- Fields and their values
- Any instructions or requirements
Format it clearly and structured for reading aloud."""
		return self._chat.chat(query, system_prompt=system_prompt, context=enhanced_context)
	
	def _analyze_label(self, text: str, query: str) -> str:
		"""Analyze label and extract product information."""
		enhanced_context = f"""Label Text Content:
{text[:4000]}"""
		system_prompt = """You are a helpful assistant analyzing a product label.
Extract and present:
- Product name
- Brand
- Key information (ingredients, nutrition facts, instructions, etc.)
- Any warnings or important notes
Format it clearly for reading aloud."""
		return self._chat.chat(query, system_prompt=system_prompt, context=enhanced_context)
	
	def _analyze_document(self, text: str, analysis: Optional[str], query: str) -> str:
		"""General document analysis."""
		enhanced_context = f"""Document Text Content:
{text[:4000]}

{analysis or ''}"""
		system_prompt = """You are a helpful assistant analyzing a document.
Extract and present the key information clearly.
Identify the document type, main topics, and important details.
Format it naturally for reading aloud."""
		return self._chat.chat(query, system_prompt=system_prompt, context=enhanced_context)
