#!/usr/bin/env python3
"""
Pixtral Analyzer - Core vision analysis using Pixtral model
"""

import base64
import time
import os
import cv2
import numpy as np
import hashlib
import json
import threading
from typing import List, Dict, Any, Optional, Union, Set
from mistralai import Mistral
from PIL import Image

from core.config import (
    DEFAULT_PIXTRAL_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMP_DIR,
    OBJECT_ANALYSIS_PROMPTS, SCENE_ANALYSIS_PROMPTS, ERROR_MESSAGES,
    TEMP_FILE_EXTENSION, MAX_CONCURRENT_ANALYSES, REQUEST_TIMEOUT, MAX_RETRIES
)

class PixtralAnalyzer:
    """
    Core Pixtral vision analyzer for object detection analysis
    """
    
    def __init__(self, api_key: str, model_name: str = DEFAULT_PIXTRAL_MODEL, 
                 max_tokens: int = DEFAULT_MAX_TOKENS, temp_dir: str = DEFAULT_TEMP_DIR,
                 verbose: bool = True):
        """
        Initialize the Pixtral analyzer
        
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
                print("🤖 Pixtral Analyzer initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mistral client: {str(e)}")
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # Initialize caching system
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        # Analysis queue for background processing
        self.analysis_queue = []
        self.analysis_thread = None
        self.analysis_running = False
    
    def analyze_object(self, frame: np.ndarray, detection: Dict[str, Any], 
                      prompt_type: str = "default") -> str:
        """
        Analyze a single detected object using Pixtral
        
        Args:
            frame: Input frame
            detection: Detection dictionary with bbox, class_name, etc.
            prompt_type: Type of prompt to use for analysis
            
        Returns:
            Analysis description string
        """
        temp_path = None
        try:
            # Extract bounding box and crop object
            bbox = detection.get('bbox', [])
            if len(bbox) != 4:
                return "Invalid bounding box format"
            
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return "Empty crop region"
            
            # Save temporary image
            temp_path = self._save_temp_image(crop, detection)
            
            # Get analysis prompt
            class_name = detection.get('class_name', 'object')
            prompt = OBJECT_ANALYSIS_PROMPTS.get(prompt_type, OBJECT_ANALYSIS_PROMPTS["default"])
            prompt = prompt.format(class_name=class_name)
            
            # Analyze with Pixtral
            description = self._analyze_image_with_pixtral(temp_path, prompt)
            
            # Clean up description
            description = self._clean_description(description)
            
            return description
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)[:50]}"
            if self.verbose:
                print(f"⚠️ {error_msg}")
            return error_msg
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    
    def analyze_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                          prompt_type: str = "default") -> List[Dict[str, Any]]:
        """
        Analyze multiple detections from object detection results
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            prompt_type: Type of prompt to use for analysis
            
        Returns:
            List of analysis results with descriptions
        """
        if not detections:
            return []
        
        results = []
        for detection in detections:
            try:
                description = self.analyze_object(frame, detection, prompt_type)
                
                # Create enhanced detection result
                enhanced_detection = detection.copy()
                enhanced_detection['pixtral_description'] = description
                enhanced_detection['analysis_timestamp'] = time.time()
                
                results.append(enhanced_detection)
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Failed to analyze detection: {str(e)}")
                
                # Add detection with error message
                enhanced_detection = detection.copy()
                enhanced_detection['pixtral_description'] = f"Analysis failed: {str(e)[:50]}"
                enhanced_detection['analysis_timestamp'] = time.time()
                results.append(enhanced_detection)
        
        return results
    
    def analyze_scene(self, frame: np.ndarray, prompt_type: str = "default") -> str:
        """
        Analyze the entire scene using Pixtral
        
        Args:
            frame: Input frame
            prompt_type: Type of prompt to use for analysis
            
        Returns:
            Scene description string
        """
        temp_path = None
        try:
            # Save temporary image
            temp_path = self._save_temp_image(frame, {"class_name": "scene"})
            
            # Get scene analysis prompt
            prompt = SCENE_ANALYSIS_PROMPTS.get(prompt_type, SCENE_ANALYSIS_PROMPTS["default"])
            
            # Analyze with Pixtral
            description = self._analyze_image_with_pixtral(temp_path, prompt)
            
            return description.strip()
            
        except Exception as e:
            error_msg = f"Scene analysis error: {str(e)[:50]}"
            if self.verbose:
                print(f"⚠️ {error_msg}")
            return error_msg
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    
    def _save_temp_image(self, image: np.ndarray, detection: Dict[str, Any]) -> str:
        """
        Save image to temporary file with retry mechanism and better error handling
        
        Args:
            image: Image array
            detection: Detection info for filename
            
        Returns:
            Path to temporary file
        """
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                timestamp = int(time.time() * 1000)
                class_name = detection.get('class_name', 'object').replace(' ', '_')
                class_id = detection.get('class_id', 0)
                track_id = detection.get('track_id', 'unknown')
                confidence = detection.get('confidence', 0)
                
                # Create more descriptive filename
                filename = f"temp_analysis_{timestamp}_{class_id}_{class_name}_{track_id}_{confidence:.2f}{TEMP_FILE_EXTENSION}"
                temp_path = os.path.join(self.temp_dir, filename)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                
                # Validate image
                if image is None or image.size == 0:
                    raise ValueError("Invalid image data")
                
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
                    raise RuntimeError("cv2.imwrite returned False")
                
                # Verify file was created and has content
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                    raise RuntimeError("File was not created or is empty")
                
                if self.verbose:
                    print(f"💾 Saved temp image: {filename}")
                
                return temp_path
                
            except Exception as e:
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"⚠️ Attempt {attempt + 1} failed to save temp image: {str(e)}, retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    if self.verbose:
                        print(f"❌ Failed to save temp image after {max_retries} attempts: {str(e)}")
                    raise RuntimeError(f"Failed to save temporary image after {max_retries} attempts: {str(e)}")
    
    def _analyze_image_with_pixtral(self, image_path: str, prompt: str) -> str:
        """
        Analyze image using Pixtral API
        
        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            
        Returns:
            Analysis result string
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
            
            # Make API call with retries
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.client.chat.complete(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens
                    )
                    
                    description = response.choices[0].message.content.strip()
                    return description
                    
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        raise e
                    time.sleep(1)  # Wait before retry
            
        except Exception as e:
            raise RuntimeError(f"Pixtral API call failed: {str(e)}")
    
    def _clean_description(self, description: str) -> str:
        """
        Clean and format description text
        
        Args:
            description: Raw description from Pixtral
            
        Returns:
            Cleaned description
        """
        if not description:
            return "No description available"
        
        # Remove extra whitespace
        description = description.strip()
        
        # Ensure it ends with proper punctuation
        if description and not description.endswith(('.', '!', '?')):
            description = description.split('.')[0] + '.' if '.' in description else description + '.'
        
        # Limit length
        if len(description) > 200:
            description = description[:197] + "..."
        
        return description
    
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
    
    def _generate_cache_key(self, frame: np.ndarray, detection: Dict[str, Any], prompt_type: str) -> str:
        """Generate a cache key for the analysis request"""
        # Create a hash based on the cropped image and detection info
        bbox = detection.get('bbox', [])
        if len(bbox) != 4:
            return None
        
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        # Create hash from image content and detection metadata
        image_hash = hashlib.md5(crop.tobytes()).hexdigest()
        detection_hash = hashlib.md5(
            f"{detection.get('class_name', '')}_{detection.get('confidence', 0):.3f}_{prompt_type}".encode()
        ).hexdigest()
        
        return f"{image_hash}_{detection_hash}"
    
    def _get_cached_analysis(self, cache_key: str) -> Optional[str]:
        """Get cached analysis result"""
        if not cache_key:
            return None
        
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Try to load from disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    result = cache_data.get('description', '')
                    with self.cache_lock:
                        self.cache[cache_key] = result
                    return result
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to load cache file {cache_file}: {str(e)}")
        
        return None
    
    def _cache_analysis(self, cache_key: str, description: str):
        """Cache analysis result"""
        if not cache_key or not description:
            return
        
        with self.cache_lock:
            self.cache[cache_key] = description
        
        # Save to disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            cache_data = {
                'description': description,
                'timestamp': time.time()
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to save cache file {cache_file}: {str(e)}")
    
    def analyze_object_with_cache(self, frame: np.ndarray, detection: Dict[str, Any], 
                                 prompt_type: str = "default") -> str:
        """
        Analyze a single detected object using Pixtral with caching
        
        Args:
            frame: Input frame
            detection: Detection dictionary with bbox, class_name, etc.
            prompt_type: Type of prompt to use for analysis
            
        Returns:
            Analysis description string
        """
        # Generate cache key
        cache_key = self._generate_cache_key(frame, detection, prompt_type)
        
        # Check cache first
        if cache_key:
            cached_result = self._get_cached_analysis(cache_key)
            if cached_result:
                if self.verbose:
                    print(f"📋 Using cached analysis for {detection.get('class_name', 'object')}")
                return cached_result
        
        # Perform analysis
        description = self.analyze_object(frame, detection, prompt_type)
        
        # Cache the result
        if cache_key and description:
            self._cache_analysis(cache_key, description)
        
        return description
    
    def analyze_detections_with_tracking(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                                       new_track_ids: Set[int] = None) -> List[Dict[str, Any]]:
        """
        Analyze detections with tracking support - only analyze new objects
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            new_track_ids: Set of new track IDs to analyze
            
        Returns:
            List of enhanced detections with Pixtral descriptions
        """
        if new_track_ids is None:
            new_track_ids = set()
        
        enhanced_detections = []
        
        for detection in detections:
            enhanced_detection = detection.copy()
            
            # Only analyze new objects or objects without descriptions
            track_id = detection.get('track_id')
            should_analyze = (
                track_id in new_track_ids or 
                'pixtral_description' not in detection or 
                not detection.get('pixtral_description')
            )
            
            if should_analyze:
                try:
                    description = self.analyze_object_with_cache(frame, detection)
                    enhanced_detection['pixtral_description'] = description
                    enhanced_detection['analysis_timestamp'] = time.time()
                    
                    if self.verbose and track_id:
                        print(f"🤖 Analyzed new object [ID: {track_id}] {detection.get('class_name', 'object')}")
                        
                except Exception as e:
                    enhanced_detection['pixtral_description'] = f"Analysis failed: {str(e)[:100]}"
                    if self.verbose:
                        print(f"⚠️ Analysis failed for {detection.get('class_name', 'object')}: {str(e)}")
            else:
                # Keep existing description
                enhanced_detection['pixtral_description'] = detection.get('pixtral_description', 'No description available')
            
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
