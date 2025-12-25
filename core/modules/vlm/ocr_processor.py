#!/usr/bin/env python3
"""
OCR Processor - Text extraction and analysis using Pixtral
"""

import base64
import time
import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from mistralai import Mistral
from PIL import Image

from core.config import (
    DEFAULT_PIXTRAL_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMP_DIR,
    OCR_PROMPTS, ERROR_MESSAGES, TEMP_FILE_EXTENSION
)

class OCRProcessor:
    """
    OCR processor using Pixtral for text extraction and analysis
    """
    
    def __init__(self, api_key: str, model_name: str = DEFAULT_PIXTRAL_MODEL,
                 max_tokens: int = DEFAULT_MAX_TOKENS, temp_dir: str = DEFAULT_TEMP_DIR,
                 verbose: bool = True):
        """
        Initialize the OCR processor
        
        Args:
            api_key: Mistral API key
            model_name: Pixtral model name
            max_tokens: Maximum tokens for responses
            temp_dir: Temporary directory for image processing
            verbose: Whether to print status messages
        """
        if not api_key:
            raise ValueError(ERROR_MESSAGES["api_key_missing"])
        
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temp_dir = temp_dir
        self.verbose = verbose
        
        # Initialize Mistral client
        try:
            self.client = Mistral(api_key=api_key)
            if self.verbose:
                print("🤖 OCR Processor initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mistral client: {str(e)}")
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def extract_text(self, frame: np.ndarray, prompt_type: str = "default") -> Dict[str, Any]:
        """
        Extract text from image using Pixtral
        
        Args:
            frame: Input frame
            prompt_type: Type of OCR prompt to use
            
        Returns:
            Dictionary with extracted text and metadata
        """
        temp_path = None
        try:
            # Save temporary image
            temp_path = self._save_temp_image(frame, "ocr")
            
            # Get OCR prompt
            prompt = OCR_PROMPTS.get(prompt_type, OCR_PROMPTS["default"])
            
            # Extract text with Pixtral
            extracted_text = self._extract_text_with_pixtral(temp_path, prompt)
            
            # Process and clean text
            processed_text = self._process_extracted_text(extracted_text)
            
            return {
                'text': processed_text,
                'raw_text': extracted_text,
                'timestamp': time.time(),
                'prompt_type': prompt_type,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"OCR extraction failed: {str(e)[:50]}"
            if self.verbose:
                print(f"⚠️ {error_msg}")
            
            return {
                'text': '',
                'raw_text': '',
                'timestamp': time.time(),
                'prompt_type': prompt_type,
                'success': False,
                'error': error_msg
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    
    def extract_text_from_region(self, frame: np.ndarray, bbox: List[int], 
                               prompt_type: str = "default") -> Dict[str, Any]:
        """
        Extract text from a specific region of the image
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            prompt_type: Type of OCR prompt to use
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if len(bbox) != 4:
            return {
                'text': '',
                'raw_text': '',
                'timestamp': time.time(),
                'prompt_type': prompt_type,
                'success': False,
                'error': 'Invalid bounding box format'
            }
        
        try:
            # Crop the region
            x1, y1, x2, y2 = map(int, bbox)
            cropped_frame = frame[y1:y2, x1:x2]
            
            if cropped_frame.size == 0:
                return {
                    'text': '',
                    'raw_text': '',
                    'timestamp': time.time(),
                    'prompt_type': prompt_type,
                    'success': False,
                    'error': 'Empty crop region'
                }
            
            # Extract text from cropped region
            return self.extract_text(cropped_frame, prompt_type)
            
        except Exception as e:
            error_msg = f"Region OCR extraction failed: {str(e)[:50]}"
            if self.verbose:
                print(f"⚠️ {error_msg}")
            
            return {
                'text': '',
                'raw_text': '',
                'timestamp': time.time(),
                'prompt_type': prompt_type,
                'success': False,
                'error': error_msg
            }
    
    def analyze_text_content(self, frame: np.ndarray, prompt_type: str = "detailed") -> Dict[str, Any]:
        """
        Analyze text content in the image
        
        Args:
            frame: Input frame
            prompt_type: Type of analysis prompt to use
            
        Returns:
            Dictionary with text analysis results
        """
        temp_path = None
        try:
            # Save temporary image
            temp_path = self._save_temp_image(frame, "text_analysis")
            
            # Get analysis prompt
            prompt = OCR_PROMPTS.get(prompt_type, OCR_PROMPTS["detailed"])
            
            # Analyze text with Pixtral
            analysis_result = self._extract_text_with_pixtral(temp_path, prompt)
            
            return {
                'analysis': analysis_result,
                'timestamp': time.time(),
                'prompt_type': prompt_type,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Text analysis failed: {str(e)[:50]}"
            if self.verbose:
                print(f"⚠️ {error_msg}")
            
            return {
                'analysis': '',
                'timestamp': time.time(),
                'prompt_type': prompt_type,
                'success': False,
                'error': error_msg
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    
    def _save_temp_image(self, image: np.ndarray, prefix: str) -> str:
        """
        Save image to temporary file
        
        Args:
            image: Image array
            prefix: Filename prefix
            
        Returns:
            Path to temporary file
        """
        timestamp = int(time.time() * 1000)
        filename = f"temp_{prefix}_{timestamp}{TEMP_FILE_EXTENSION}"
        temp_path = os.path.join(self.temp_dir, filename)
        
        # Ensure image is in correct format for Pixtral (RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB for Pixtral model
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Save as RGB using PIL to ensure proper color format
            pil_image = Image.fromarray(image_rgb)
            pil_image.save(temp_path, 'JPEG', quality=95)
            success = True
        else:
            # For grayscale images, convert to RGB
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                pil_image = Image.fromarray(image_rgb)
                pil_image.save(temp_path, 'JPEG', quality=95)
                success = True
            else:
                success = cv2.imwrite(temp_path, image)
        
        if not success:
            raise RuntimeError("Failed to save temporary image")
        
        return temp_path
    
    def _extract_text_with_pixtral(self, image_path: str, prompt: str) -> str:
        """
        Extract text using Pixtral API
        
        Args:
            image_path: Path to image file
            prompt: OCR prompt
            
        Returns:
            Extracted text string
        """
        try:
            # Read and encode image
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Create messages for Pixtral
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }]
            
            # Make API call
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise RuntimeError(f"Pixtral OCR API call failed: {str(e)}")
    
    def _process_extracted_text(self, text: str) -> str:
        """
        Process and clean extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Processed text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = text.strip()
        
        # Remove common OCR artifacts
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Remove multiple spaces
        
        # Limit length
        if len(text) > 1000:
            text = text[:997] + "..."
        
        return text
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and configuration
        
        Returns:
            Dictionary with system information
        """
        return {
            'model_name': self.model_name,
            'max_tokens': self.max_tokens,
            'temp_dir': self.temp_dir,
            'verbose': self.verbose,
            'api_key_configured': bool(self.api_key),
            'client_initialized': hasattr(self, 'client') and self.client is not None
        }
