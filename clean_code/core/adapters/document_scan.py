from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np
import cv2
import os

try:
	from core.modules.vlm.ocr_processor import OCRProcessor  # type: ignore
	from core.modules.vlm.pixtral_analyzer import PixtralAnalyzer  # type: ignore
	from core.modules.vlm.mistral_chat import MistralChat  # type: ignore
except Exception:
	OCRProcessor = None  # type: ignore
	PixtralAnalyzer = None  # type: ignore
	MistralChat = None  # type: ignore


def _preprocess_frame_for_ocr(frame: np.ndarray) -> np.ndarray:
	"""Preprocess frame for better OCR accuracy."""
	# Convert to grayscale if color
	if len(frame.shape) == 3:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	else:
		gray = frame.copy()
	
	# Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	enhanced = clahe.apply(gray)
	
	# Optional: Apply slight sharpening
	kernel = np.array([[-1, -1, -1],
	                   [-1,  9, -1],
	                   [-1, -1, -1]])
	sharpened = cv2.filter2D(enhanced, -1, kernel)
	
	# Convert back to BGR for compatibility
	return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


class DocumentScanAdapter:
	def __init__(self, api_key: Optional[str] = None, verbose: bool = False) -> None:
		api_key = api_key or os.environ.get("MISTRAL_API_KEY") or ""
		self._ocr = OCRProcessor(api_key=api_key, verbose=verbose) if OCRProcessor and api_key else None
		self._pixtral: Optional[Any] = None
		self._mistral_chat: Optional[Any] = None
		if PixtralAnalyzer and api_key:
			try:
				self._pixtral = PixtralAnalyzer(api_key=api_key, verbose=verbose)
			except Exception as e:
				if verbose:
					print(f"[WARNING] PixtralAnalyzer initialization failed: {e}")
		if MistralChat and api_key:
			try:
				self._mistral_chat = MistralChat(api_key=api_key, verbose=verbose)
			except Exception as e:
				if verbose:
					print(f"[WARNING] MistralChat initialization failed: {e}")

	def scan(self, frame: Optional[np.ndarray]) -> Dict[str, Any]:
		if frame is None:
			return {"blocks": [], "summary": "No frame provided"}
		if self._ocr is None:
			return {"blocks": [], "summary": "OCR not available"}
		
		# Preprocess frame for better OCR
		processed_frame = _preprocess_frame_for_ocr(frame)
		
		# Use detailed OCR prompt for better accuracy
		res = self._ocr.extract_text(processed_frame, prompt_type="detailed")
		
		if not res or not res.get("success"):
			# Fallback to default prompt if detailed fails
			res = self._ocr.extract_text(frame, prompt_type="default")
		
		text = (res or {}).get("text", "") or (res or {}).get("raw_text", "")
		
		if not text:
			return {"blocks": [], "summary": "No text detected in the image"}
		
		# Split into lines and preserve more context
		lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
		
		# Create blocks with full text preservation (not just first 50 lines)
		blocks = [{"text": ln, "bbox": None, "confidence": None} for ln in lines]
		
		# Create a more comprehensive summary (first 1000 chars, multiple paragraphs)
		if len(lines) > 0:
			# Try to preserve paragraph structure
			full_text = "\n".join(lines)
			summary = full_text[:2000]  # Keep more text for better context
			if len(full_text) > 2000:
				summary += "..."
		else:
			summary = text[:500]
		
		return {
			"blocks": blocks,
			"summary": summary or "(no text)",
			"full_text": text[:5000],  # Keep full text for LLM context
			"success": res.get("success", False)
		}
	
	def scan_with_detection(self, frame: Optional[np.ndarray], detect_first: bool = True) -> Dict[str, Any]:
		"""
		Scan document with optional document detection step (like main.py).
		
		Args:
			frame: Input frame
			detect_first: If True, use Pixtral to detect document before OCR
		
		Returns:
			Dict with blocks, summary, full_text, success, document_detected
		"""
		if frame is None:
			return {"blocks": [], "summary": "No frame provided", "document_detected": False}
		
		# Step 1: Document detection (if enabled)
		document_detected = True
		if detect_first and self._pixtral:
			try:
				scene_analysis = self._pixtral.analyze_scene(frame, "document_detection")
				document_keywords = ['document', 'paper', 'text', 'letter', 'page', 'book', 
				                     'magazine', 'newspaper', 'form', 'contract', 'receipt', 
				                     'invoice', 'label', 'sign']
				has_document = any(keyword in scene_analysis.lower() for keyword in document_keywords) if scene_analysis else False
				
				if not has_document:
					return {
						"blocks": [],
						"summary": "No document detected in the current view. Please place a document in front of the camera.",
						"full_text": "",
						"document_detected": False,
						"success": False
					}
			except Exception as e:
				if self._ocr and hasattr(self._ocr, 'verbose') and self._ocr.verbose:
					print(f"[WARNING] Document detection failed: {e}, proceeding with OCR anyway")
				# Continue with OCR even if detection fails
		
		# Step 2: Perform OCR (existing logic)
		result = self.scan(frame)
		result["document_detected"] = document_detected
		return result
	
	def detect_document_type(self, frame: Optional[np.ndarray], extracted_text: Optional[str] = None) -> str:
		"""
		Detect document type using Pixtral analysis.
		
		Args:
			frame: Input frame
			extracted_text: Optional extracted text for better classification
		
		Returns:
			Document type: "receipt", "letter", "form", "label", "general", or "unknown"
		"""
		if not self._pixtral or frame is None:
			return "unknown"
		
		try:
			# Use Pixtral to classify document type
			prompt = """What type of document is this? Choose one: receipt, letter, form, label, invoice, contract, or general document.
			Respond with only the document type word."""
			
			# Analyze scene for document type
			scene_desc = self._pixtral.analyze_scene(frame, "document_detection")
			
			# Simple keyword-based classification
			scene_lower = scene_desc.lower()
			if any(word in scene_lower for word in ['receipt', 'purchase', 'payment', 'total', 'tax']):
				return "receipt"
			elif any(word in scene_lower for word in ['letter', 'dear', 'signature', 'sincerely']):
				return "letter"
			elif any(word in scene_lower for word in ['form', 'application', 'field', 'checkbox']):
				return "form"
			elif any(word in scene_lower for word in ['label', 'ingredient', 'nutrition', 'product']):
				return "label"
			elif any(word in scene_lower for word in ['invoice', 'bill', 'charge']):
				return "invoice"
			elif any(word in scene_lower for word in ['contract', 'agreement', 'terms']):
				return "contract"
			else:
				return "general"
		except Exception as e:
			return "unknown"
	
	def scan_with_intent(self, frame: Optional[np.ndarray], intent: str = "read",
	                    document_type: Optional[str] = None) -> Dict[str, Any]:
		"""
		Scan document based on user intent.
		
		Args:
			frame: Input frame
			intent: One of "read", "summarize", "analyze", "extract"
			document_type: Optional document type hint (receipt, letter, form, etc.)
		
		Returns:
			Dict with text, analysis, format, recommendation, intent, document_type
		"""
		if frame is None:
			return {
				"blocks": [],
				"summary": "No frame provided",
				"full_text": "",
				"intent": intent,
				"success": False
			}
		
		# Step 1: Detect document (with detection)
		scan_result = self.scan_with_detection(frame, detect_first=True)
		
		if not scan_result.get("document_detected", False):
			return scan_result
		
		# Step 2: Detect document type if not provided
		if not document_type:
			document_type = self.detect_document_type(frame, scan_result.get("full_text"))
		
		# Step 3: Extract text (already done in scan_with_detection)
		extracted_text = scan_result.get("full_text", "")
		
		result = {
			"blocks": scan_result.get("blocks", []),
			"summary": scan_result.get("summary", ""),
			"full_text": extracted_text,
			"intent": intent,
			"document_type": document_type,
			"success": scan_result.get("success", False)
		}
		
		# Step 4: Process based on intent
		if intent == "analyze" and self._ocr and extracted_text:
			# Use OCRProcessor.analyze_text_content() for detailed analysis
			try:
				analysis_result = self._ocr.analyze_text_content(frame, prompt_type="detailed")
				if analysis_result.get("success"):
					result["analysis"] = analysis_result.get("analysis", "")
			except Exception as e:
				if self._ocr and hasattr(self._ocr, 'verbose') and self._ocr.verbose:
					print(f"[WARNING] Text analysis failed: {e}")
		
		# Step 5: Add format recommendation based on document type and intent
		if document_type == "receipt" and intent == "analyze":
			result["format"] = "structured"  # Receipt should be structured
			result["recommendation"] = "extract_items_total_date"
		elif intent == "read":
			result["format"] = "verbatim"  # Read as-is
			result["recommendation"] = "read_aloud"
		elif intent == "summarize":
			result["format"] = "summary"  # Summarize
			result["recommendation"] = "create_summary"
		else:
			result["format"] = "structured"
			result["recommendation"] = "extract_key_info"
		
		return result
