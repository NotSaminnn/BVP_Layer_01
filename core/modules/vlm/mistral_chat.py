#!/usr/bin/env python3
"""
Mistral Chat - Text-based chat and analysis using Mistral models
"""

import json
import re
import time
from typing import Dict, Any, Optional, List
from mistralai import Mistral

from core.config import (
    DEFAULT_CHAT_MODEL, DEFAULT_MAX_TOKENS, ERROR_MESSAGES
)

class MistralChat:
    """
    Text-based chat system using Mistral models
    """
    
    def __init__(self, api_key: str, model_name: str = DEFAULT_CHAT_MODEL,
                 max_tokens: int = DEFAULT_MAX_TOKENS, verbose: bool = True, temperature: float = 0.7):
        """
        Initialize the Mistral chat system
        
        Args:
            api_key: Mistral API key
            model_name: Chat model name
            max_tokens: Maximum tokens for responses
            verbose: Whether to print status messages
            temperature: Sampling temperature (0.0-1.0, lower = more deterministic, faster)
        """
        if not api_key:
            raise ValueError(ERROR_MESSAGES["api_key_missing"])
        
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Increased to 1.0 seconds between requests to avoid rate limits
        
        # Initialize Mistral client
        try:
            self.client = Mistral(api_key=api_key)
            if self.verbose:
                print("🤖 Mistral Chat initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mistral client: {str(e)}")
    
    def chat(self, message: str, system_prompt: str = None, 
             context: str = None, max_retries: int = 3) -> str:
        """
        Send a chat message to Mistral with retry logic for rate limiting
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            context: Optional context information
            max_retries: Maximum number of retries for rate limit errors (default: 3)
            
        Returns:
            Chat response
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add context if provided
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        
        # Add user message
        messages.append({"role": "user", "content": message})
        
        # Retry logic with exponential backoff for rate limiting
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Rate limiting - ensure minimum interval between requests
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                if time_since_last_request < self.min_request_interval:
                    sleep_time = self.min_request_interval - time_since_last_request
                    time.sleep(sleep_time)
                
                # Make API call with temperature for faster, more deterministic responses
                response = self.client.chat.complete(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                # Update last request time
                self.last_request_time = time.time()
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_exception = e
                error_str = str(e)
                
                # Check if it's a rate limit error (429) - check multiple patterns
                is_rate_limit = (
                    "429" in error_str or 
                    "Status 429" in error_str or
                    "rate limit" in error_str.lower() or
                    "service_tier_capacity_exceeded" in error_str.lower() or
                    "capacity exceeded" in error_str.lower() or
                    '"code":"3505"' in error_str or  # Mistral API error code for capacity exceeded
                    "3505" in error_str
                )
                
                if is_rate_limit and attempt < max_retries - 1:
                    # Calculate exponential backoff: 2^attempt seconds (1s, 2s, 4s, etc.)
                    # Add jitter to avoid thundering herd
                    backoff_time = (2 ** attempt) + (0.1 * attempt)
                    if self.verbose:
                        print(f"⚠️ Rate limit exceeded (attempt {attempt + 1}/{max_retries}). Waiting {backoff_time:.1f}s before retry...")
                    time.sleep(backoff_time)
                    continue  # Retry
                else:
                    # Not a rate limit error, or max retries reached
                    if not is_rate_limit:
                        # For non-rate-limit errors, don't retry
                        break
                    # Rate limit but max retries reached - will fall through to error message
        
        # All retries failed or non-retryable error
        if last_exception is None:
            return "An unknown error occurred. Please try again."
        
        error_msg = self._format_error_message(last_exception)
        if self.verbose:
            print(f"⚠️ {error_msg}")
        return error_msg
    
    def _format_error_message(self, exception: Exception) -> str:
        """
        Format user-friendly error messages from API exceptions
        
        Args:
            exception: The exception that occurred
            
        Returns:
            User-friendly error message
        """
        error_str = str(exception)
        
        # Check for rate limit errors (429) - including specific Mistral error codes
        if ("429" in error_str or 
            "Status 429" in error_str or
            "rate limit" in error_str.lower() or 
            "service_tier_capacity_exceeded" in error_str.lower() or
            '"code":"3505"' in error_str or
            "3505" in error_str):
            return ("I'm experiencing high demand right now and can't process your request immediately. "
                   "Please wait a moment and try again. If this continues, you may need to wait a few minutes "
                   "or check your API service tier limits.")
        
        # Check for API key errors
        if "401" in error_str or "unauthorized" in error_str.lower() or "api key" in error_str.lower():
            return "There's an issue with the API authentication. Please check your API key configuration."
        
        # Check for model errors
        if "model" in error_str.lower() and "not available" in error_str.lower():
            return "The requested AI model is currently unavailable. Please try again later."
        
        # Check for timeout errors
        if "timeout" in error_str.lower() or "timed out" in error_str.lower():
            return "The request took too long to process. Please try again."
        
        # Generic error message
        if "Chat error:" in error_str:
            # Remove "Chat error:" prefix for cleaner message
            error_str = error_str.replace("Chat error: ", "")
        
        return f"I encountered an error: {error_str}. Please try again."
    
    def classify_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Classify user intent using Mistral
        
        Args:
            user_message: User message to classify
            
        Returns:
            Dictionary with intent classification
        """
        system_prompt = (
            "You are an intent classifier. Map the user's question to one of: "
            "DESCRIBE_SCENE, WHAT_IS_the_object, WHERE_IS_the_object, HOW_IS_the_object, "
            "IS_THERE_a/an_OBJECT, HOW_MANY_the_object_are_there, SCAN_DOCUMENT, OCR, LIST_OBJECTS. "
            "If an object is mentioned (e.g., pen, mug, laptop), return it as 'object'. "
            "Respond ONLY in JSON like {\"intent\":\"...\", \"object\":\"...\"}."
        )
        
        examples = (
            "Examples:\n"
            "what is the pen -> {\"intent\":\"WHAT_IS_the_object\", \"object\":\"pen\"}\n"
            "where is the mug -> {\"intent\":\"WHERE_IS_the_object\", \"object\":\"mug\"}\n"
            "how is the phone -> {\"intent\":\"HOW_IS_the_object\", \"object\":\"phone\"}\n"
            "is there a laptop -> {\"intent\":\"IS_THERE_a/an_OBJECT\", \"object\":\"laptop\"}\n"
            "how many mugs are there -> {\"intent\":\"HOW_MANY_the_object_are_there\", \"object\":\"mug\"}\n"
            "describe the scene -> {\"intent\":\"DESCRIBE_SCENE\", \"object\":\"\"}\n"
            "scan document -> {\"intent\":\"SCAN_DOCUMENT\", \"object\":\"\"}\n"
            "read the document -> {\"intent\":\"SCAN_DOCUMENT\", \"object\":\"\"}\n"
            "scan the document -> {\"intent\":\"SCAN_DOCUMENT\", \"object\":\"\"}\n"
            "what objects are in the scene -> {\"intent\":\"LIST_OBJECTS\", \"object\":\"\"}"
        )
        
        try:
            response = self.chat(
                message=f"{examples}\n\nQuestion: {user_message}",
                system_prompt=system_prompt
            )
            
            # Parse JSON response
            data = json.loads(response)
            intent = str(data.get("intent", "")).upper()
            obj = data.get("object") or None
            
            # Validate intent
            valid_intents = {
                "DESCRIBE_SCENE", "WHAT_IS_the_object", "WHERE_IS_the_object", 
                "HOW_IS_the_object", "IS_THERE_a/an_OBJECT", "HOW_MANY_the_object_are_there",
                "SCAN_DOCUMENT", "OCR", "LIST_OBJECTS"
            }
            
            if intent not in valid_intents:
                intent = "UNKNOWN"
            
            return {
                "intent": intent,
                "object": obj,
                "confidence": 1.0
            }
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Intent classification failed: {str(e)}")
            
            # Fallback to regex-based classification
            return self._fallback_intent_classification(user_message)
    
    def _fallback_intent_classification(self, user_message: str) -> Dict[str, Any]:
        """
        Fallback intent classification using regex patterns
        
        Args:
            user_message: User message to classify
            
        Returns:
            Dictionary with intent classification
        """
        text = user_message.lower()
        
        # Define regex patterns
        patterns = {
            "SCAN_DOCUMENT": re.compile(r"\b(scan\s+document|read\s+document|scan\s+the\s+document|read\s+the\s+document)\b", re.IGNORECASE),
            "OCR": re.compile(r"\b(scan|ocr)(?:\b|\s)", re.IGNORECASE),
            "DESCRIBE_SCENE": re.compile(r"\b(describe\s+the\s+scene|what\s+do\s+you\s+see|what\s+is\s+in\s+the\s+scene)\b", re.IGNORECASE),
            "WHAT_IS_the_object": re.compile(r"\bwhat(?:\s+is|'s)?\s+(?:the\s+|a\s+)?([a-zA-Z_\-]+)", re.IGNORECASE),
            "WHERE_IS_the_object": re.compile(r"\bwhere(?:\s+is|'s)?\s+(?:the\s+|a\s+)?([a-zA-Z_\-]+)", re.IGNORECASE),
            "HOW_IS_the_object": re.compile(r"\bhow(?:\s+is|'s)?\s+(?:the\s+|a\s+)?([a-zA-Z_\-]+)", re.IGNORECASE),
            "IS_THERE_a/an_OBJECT": re.compile(r"\bis\s+there\s+(?:an|a|the)?\s*([a-zA-Z_\-]+)", re.IGNORECASE),
            "HOW_MANY_the_object_are_there": re.compile(r"\bhow\s+many(?:\s+(?:of\s+)?(?:the\s+|a\s+)?)?\s*([a-zA-Z_\-]*)", re.IGNORECASE)
        }
        
        # Check patterns in order of priority
        for intent, pattern in patterns.items():
            match = pattern.search(text)
            if match:
                obj = match.group(1) if match.groups() and match.group(1) else None
                return {
                    "intent": intent,
                    "object": obj,
                    "confidence": 0.8  # Lower confidence for regex fallback
                }
        
        # Default to LIST_OBJECTS if no pattern matches
        return {
            "intent": "LIST_OBJECTS",
            "object": None,
            "confidence": 0.5
        }
    
    def answer_with_context(self, user_message: str, context: str) -> str:
        """
        Answer user question with provided context
        
        Args:
            user_message: User question
            context: Context information
            
        Returns:
            Answer string
        """
        system_prompt = (
            "You are a concise assistant. Use provided context to answer the user's question. "
            "Do not invent objects that are not present. Keep answers under 2 sentences."
        )
        
        return self.chat(
            message=f"Context:\n{context}\n\nQuestion: {user_message}",
            system_prompt=system_prompt
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and configuration
        
        Returns:
            Dictionary with system information
        """
        return {
            'model_name': self.model_name,
            'max_tokens': self.max_tokens,
            'verbose': self.verbose,
            'api_key_configured': bool(self.api_key),
            'client_initialized': hasattr(self, 'client') and self.client is not None
        }
