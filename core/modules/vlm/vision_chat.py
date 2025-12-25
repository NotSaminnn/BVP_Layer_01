#!/usr/bin/env python3
"""
Vision Chat - Combined vision and text chat system
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Callable

from core.pixtral_analyzer import PixtralAnalyzer
from core.mistral_chat import MistralChat
from core.ocr_processor import OCRProcessor
from core.config import ERROR_MESSAGES

class VisionChat:
    """
    Combined vision and text chat system that integrates Pixtral and Mistral
    """
    
    def __init__(self, api_key: str, 
                 pixtral_model: str = "pixtral-12b-2409",
                 chat_model: str = "mistral-small-latest",
                 verbose: bool = True):
        """
        Initialize the vision chat system
        
        Args:
            api_key: Mistral API key
            pixtral_model: Pixtral model name
            chat_model: Chat model name
            verbose: Whether to print status messages
        """
        if not api_key:
            raise ValueError(ERROR_MESSAGES["api_key_missing"])
        
        self.api_key = api_key
        self.verbose = verbose
        
        # Initialize components
        self.pixtral_analyzer = PixtralAnalyzer(
            api_key=api_key, 
            model_name=pixtral_model,
            verbose=verbose
        )
        
        self.mistral_chat = MistralChat(
            api_key=api_key,
            model_name=chat_model,
            verbose=verbose
        )
        
        self.ocr_processor = OCRProcessor(
            api_key=api_key,
            verbose=verbose
        )
        
        if self.verbose:
            print("🤖 Vision Chat system initialized successfully")
    
    def handle_query(self, user_message: str, frame: np.ndarray = None, 
                    detections: List[Dict[str, Any]] = None) -> str:
        """
        Handle user query with vision and text capabilities
        
        Args:
            user_message: User query
            frame: Current frame (optional)
            detections: List of detections (optional)
            
        Returns:
            Response string
        """
        if not user_message:
            return ""
        
        # Classify intent
        intent_result = self.mistral_chat.classify_intent(user_message)
        intent = intent_result["intent"]
        obj = intent_result["object"]
        
        if self.verbose:
            print(f"🎯 Intent: {intent}, Object: {obj}")
        
        # Route based on intent
        if intent == "OCR":
            return self._handle_ocr_query(user_message, frame)
        
        elif intent == "DESCRIBE_SCENE":
            return self._handle_scene_description(user_message, frame)
        
        elif intent == "SCAN_DOCUMENT":
            return self._handle_scan_document(user_message, frame)
        
        elif intent == "LIST_OBJECTS":
            return self._handle_list_objects(user_message, detections)
        
        elif intent in ["WHAT_IS_the_object", "WHERE_IS_the_object", "HOW_IS_the_object", 
                        "IS_THERE_a/an_OBJECT", "HOW_MANY_the_object_are_there"]:
            return self._handle_object_query(intent, obj, user_message, frame, detections)
        
        else:
            # Fallback to general chat
            return self.mistral_chat.chat(
                user_message,
                system_prompt="You are a concise assistant. Keep answers under 2 sentences."
            )
    
    def _handle_ocr_query(self, user_message: str, frame: np.ndarray) -> str:
        """Handle OCR-related queries"""
        if frame is None:
            return "No video frame available. Start webcam or video first."
        
        try:
            ocr_result = self.ocr_processor.extract_text(frame)
            extracted_text = ocr_result.get('text', '(no text)')
            
            context = f"OCR extracted text: {extracted_text[:2000]}"
            return self.mistral_chat.answer_with_context(user_message, context)
            
        except Exception as e:
            return f"OCR analysis failed: {str(e)[:50]}"
    
    def _handle_scene_description(self, user_message: str, frame: np.ndarray) -> str:
        """Handle scene description queries"""
        if frame is None:
            return "No video frame available. Start webcam or video first."
        
        try:
            description = self.pixtral_analyzer.analyze_scene(frame)
            return description or "No scene description available"
            
        except Exception as e:
            return f"Scene analysis failed: {str(e)[:50]}"
    
    def _handle_scan_document(self, user_message: str, frame: np.ndarray) -> str:
        """Handle document scanning queries - detects and reads documents using OCR"""
        if frame is None:
            return "No video frame available. Start webcam or video first."
        
        try:
            # First, check if there's a document-like object in the scene using Pixtral
            scene_analysis = self.pixtral_analyzer.analyze_scene(frame, "document_detection")
            
            # Check if the scene contains document-like objects
            document_keywords = ['document', 'paper', 'text', 'letter', 'page', 'book', 'magazine', 'newspaper', 'form', 'contract']
            has_document = any(keyword in scene_analysis.lower() for keyword in document_keywords) if scene_analysis else False
            
            if not has_document:
                return "No document detected in the current view. Please place a document in front of the camera."
            
            # If document is detected, perform OCR
            ocr_result = self.ocr_processor.extract_text(frame)
            extracted_text = ocr_result.get('text', '')
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                return "Document detected but no readable text found. Please ensure the document is clearly visible and well-lit."
            
            # Format the extracted text for better readability
            formatted_text = self._format_extracted_text(extracted_text)
            
            # Create a comprehensive response
            response = f"Document detected and scanned successfully!\n\n"
            response += f"Extracted text:\n{formatted_text}"
            
            return response
            
        except Exception as e:
            return f"Document scanning failed: {str(e)[:100]}"
    
    def _format_extracted_text(self, text: str) -> str:
        """Format extracted text for better readability"""
        if not text:
            return "No text extracted"
        
        # Clean up the text
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:  # Filter out very short lines
                formatted_lines.append(line)
        
        # Join lines and limit length
        formatted_text = '\n'.join(formatted_lines)
        
        # Limit to reasonable length (first 1000 characters)
        if len(formatted_text) > 1000:
            formatted_text = formatted_text[:1000] + "\n... (text truncated)"
        
        return formatted_text
    
    def _handle_list_objects(self, user_message: str, detections: List[Dict[str, Any]]) -> str:
        """Handle object listing queries"""
        if not detections:
            return "No objects detected in the current view."
        
        # Extract unique object names
        object_names = []
        for detection in detections:
            class_name = detection.get('class_name', 'Unknown')
            if class_name not in object_names:
                object_names.append(class_name)
        
        context = f"Detected objects: {', '.join(object_names)}"
        return self.mistral_chat.answer_with_context(user_message, context)
    
    def _handle_object_query(self, intent: str, obj: str, user_message: str, 
                           frame: np.ndarray, detections: List[Dict[str, Any]]) -> str:
        """Handle object-specific queries"""
        if not obj:
            if intent == "HOW_MANY_the_object_are_there":
                count = len(detections) if detections else 0
                context = f"Total detected objects: {count}."
                return self.mistral_chat.answer_with_context(user_message, context)
            return self.mistral_chat.answer_with_context(user_message, "No object specified.")
        
        # Find the object in detections
        target_detection = self._find_object_by_name(obj, detections)
        
        if not target_detection:
            if intent == "IS_THERE_a/an_OBJECT":
                return "No"
            elif intent == "HOW_MANY_the_object_are_there":
                return "0"
            else:
                context = f"Object '{obj}' not detected in current view."
                return self.mistral_chat.answer_with_context(user_message, context)
        
        if intent == "WHAT_IS_the_object":
            return self._handle_what_is_query(target_detection, user_message, frame)
        
        elif intent == "WHERE_IS_the_object":
            return self._handle_where_is_query(target_detection, user_message, frame)
        
        elif intent == "HOW_IS_the_object":
            return self._handle_how_is_query(target_detection, user_message, frame)
        
        elif intent == "IS_THERE_a/an_OBJECT":
            return "Yes"
        
        elif intent == "HOW_MANY_the_object_are_there":
            return self._handle_how_many_query(obj, user_message, detections)
        
        return "Unknown intent"
    
    def _handle_what_is_query(self, detection: Dict[str, Any], user_message: str, frame: np.ndarray) -> str:
        """Handle 'what is' queries - provides detailed object descriptions"""
        try:
            class_name = detection.get('class_name', 'object')
            
            if frame is None:
                return f"It is a {class_name}."
            
            # Get detailed analysis of the object using Pixtral
            description = self.pixtral_analyzer.analyze_object(frame, detection, "detailed")
            
            # Create comprehensive description
            if description and description != "No description available":
                return f"It is a {class_name}. {description}"
            else:
                return f"It is a {class_name}."
            
        except Exception as e:
            return f"It is a {detection.get('class_name', 'object')}."
    
    def _handle_where_is_query(self, detection: Dict[str, Any], user_message: str, frame: np.ndarray) -> str:
        """Handle 'where is' queries using distance and angle from YOLO"""
        try:
            # Get distance and angle information from YOLO detection
            distance = detection.get('distance', 0)
            angle_x = detection.get('angle_x', 0)
            angle_y = detection.get('angle_y', 0)
            class_name = detection.get('class_name', 'object')
            
            # Create detailed location description using YOLO data
            location_info = f"The {class_name} is located at {distance:.2f} meters away"
            
            if angle_x != 0 or angle_y != 0:
                location_info += f", with horizontal angle of {angle_x:+.1f}° and vertical angle of {angle_y:+.1f}°"
            
            # Add relative position for context
            relative_pos = self._get_object_location(detection, frame.shape if frame is not None else (480, 640))
            location_info += f", positioned in the {relative_pos} area of the frame"
            
            return location_info
            
        except Exception as e:
            return f"The {detection.get('class_name', 'object')} is in the frame."
    
    def _handle_how_is_query(self, detection: Dict[str, Any], user_message: str, frame: np.ndarray) -> str:
        """Handle 'how is' queries - provides object condition/state descriptions"""
        try:
            class_name = detection.get('class_name', 'object')
            
            if frame is None:
                return f"The {class_name} appears to be in good condition."
            
            # Get condition/state analysis of the object using Pixtral
            description = self.pixtral_analyzer.analyze_object(frame, detection, "condition")
            
            # Create condition description
            if description and description != "No description available":
                return f"The {class_name} {description}"
            else:
                return f"The {class_name} appears to be in good condition."
            
        except Exception as e:
            return f"The {detection.get('class_name', 'object')} appears to be in good condition."
    
    def _handle_how_many_query(self, obj: str, user_message: str, detections: List[Dict[str, Any]]) -> str:
        """Handle 'how many' queries - returns the number of objects"""
        if not detections:
            return "0"
        
        # Count matching objects with improved matching logic
        obj_lower = obj.lower().strip()
        count = 0
        matched_objects = []
        
        for detection in detections:
            class_name = detection.get('class_name', '').lower().strip()
            class_id = detection.get('class_id', '')
            
            # Check for exact match or partial match
            if (obj_lower == class_name or 
                obj_lower in class_name or 
                class_name in obj_lower or
                # Handle plural/singular variations
                (obj_lower + 's') == class_name or
                (class_name + 's') == obj_lower or
                # Handle common variations
                self._is_object_match(obj_lower, class_name)):
                
                count += 1
                matched_objects.append({
                    'class_name': detection.get('class_name', 'Unknown'),
                    'class_id': class_id,
                    'confidence': detection.get('confidence', 0)
                })
        
        # Return count as string
        if count == 0:
            return "0"
        elif count == 1:
            return "1"
        else:
            return str(count)
    
    def _is_object_match(self, query_obj: str, class_name: str) -> bool:
        """Check if query object matches class name with common variations"""
        # Common object name mappings
        object_mappings = {
            'flower': ['flower', 'flowers', 'plant', 'plants', 'vase', 'vases'],
            'tv': ['tv', 'television', 'screen', 'monitor', 'display'],
            'fan': ['fan', 'ceiling fan', 'ventilator'],
            'cabinet': ['cabinet', 'shelf', 'shelves', 'cupboard', 'storage'],
            'laptop': ['laptop', 'computer', 'notebook', 'pc'],
            'phone': ['phone', 'mobile', 'cellphone', 'smartphone'],
            'mug': ['mug', 'cup', 'coffee cup', 'drinking vessel'],
            'pen': ['pen', 'pencil', 'writing instrument'],
            'book': ['book', 'books', 'novel', 'magazine'],
            'chair': ['chair', 'seat', 'stool'],
            'table': ['table', 'desk', 'surface'],
            'car': ['car', 'vehicle', 'automobile', 'auto'],
            'person': ['person', 'people', 'human', 'man', 'woman', 'child']
        }
        
        # Check if query object is in any mapping group
        for key, variations in object_mappings.items():
            if query_obj in variations and class_name in variations:
                return True
        
        return False
    
    def _find_object_by_name(self, name: str, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find object by name in detections with improved matching"""
        if not detections:
            return None
        
        name_lower = name.lower().strip()
        
        # First try exact match
        for detection in detections:
            class_name = detection.get('class_name', '').lower().strip()
            if name_lower == class_name:
                return detection
        
        # Then try partial match
        for detection in detections:
            class_name = detection.get('class_name', '').lower().strip()
            if (name_lower in class_name or 
                class_name in name_lower or
                # Handle plural/singular variations
                (name_lower + 's') == class_name or
                (class_name + 's') == name_lower or
                # Handle common variations
                self._is_object_match(name_lower, class_name)):
                return detection
        
        return None
    
    def _get_object_location(self, detection: Dict[str, Any], frame_shape: tuple) -> str:
        """Get object location description"""
        if not detection or not frame_shape:
            return "unknown location"
        
        bbox = detection.get('bbox', [])
        if len(bbox) != 4:
            return "unknown location"
        
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        
        # Determine horizontal position
        if cx < w/3:
            horiz = "left"
        elif cx > 2*w/3:
            horiz = "right"
        else:
            horiz = "center"
        
        # Determine vertical position
        if cy < h/3:
            vert = "top"
        elif cy > 2*h/3:
            vert = "bottom"
        else:
            vert = "middle"
        
        return f"{vert}-{horiz}"
    
    def analyze_detections_with_descriptions(self, frame: np.ndarray, 
                                          detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze detections and add Pixtral descriptions
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            List of enhanced detections with descriptions
        """
        return self.pixtral_analyzer.analyze_detections(frame, detections)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and configuration
        
        Returns:
            Dictionary with system information
        """
        return {
            'pixtral_analyzer': self.pixtral_analyzer.get_system_info(),
            'mistral_chat': self.mistral_chat.get_system_info(),
            'ocr_processor': self.ocr_processor.get_system_info(),
            'verbose': self.verbose
        }
