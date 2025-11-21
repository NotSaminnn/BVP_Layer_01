#!/usr/bin/env python3
"""
Object Analyzer - Integration with object detection systems
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Union

from core.pixtral_analyzer import PixtralAnalyzer
from core.config import ERROR_MESSAGES

class ObjectAnalyzer:
    """
    Object analyzer that integrates with object detection systems
    """
    
    def __init__(self, api_key: str, **kwargs):
        """
        Initialize the object analyzer
        
        Args:
            api_key: Mistral API key
            **kwargs: Additional configuration options
        """
        if not api_key:
            raise ValueError(ERROR_MESSAGES["api_key_missing"])
        
        self.api_key = api_key
        self.pixtral_analyzer = PixtralAnalyzer(api_key=api_key, **kwargs)
    
    def analyze_detection_results(self, frame: np.ndarray, 
                                detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze detection results from object detection system
        
        Args:
            frame: Input frame
            detection_results: Detection results from object detection system
            
        Returns:
            Enhanced detection results with Pixtral analysis
        """
        if not detection_results:
            return detection_results
        
        # Extract detections from different possible formats
        detections = self._extract_detections(detection_results)
        
        if not detections:
            return detection_results
        
        # Analyze detections with Pixtral
        enhanced_detections = self.pixtral_analyzer.analyze_detections(frame, detections)
        
        # Create enhanced results
        enhanced_results = detection_results.copy()
        enhanced_results['detections'] = enhanced_detections
        enhanced_results['pixtral_analysis'] = {
            'total_objects': len(enhanced_detections),
            'analyzed_objects': len([d for d in enhanced_detections if 'pixtral_description' in d]),
            'analysis_timestamp': enhanced_detections[0].get('analysis_timestamp') if enhanced_detections else None
        }
        
        return enhanced_results
    
    def analyze_single_detection(self, frame: np.ndarray, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single detection
        
        Args:
            frame: Input frame
            detection: Single detection dictionary
            
        Returns:
            Enhanced detection with Pixtral analysis
        """
        if not detection:
            return detection
        
        # Analyze with Pixtral
        description = self.pixtral_analyzer.analyze_object(frame, detection)
        
        # Create enhanced detection
        enhanced_detection = detection.copy()
        enhanced_detection['pixtral_description'] = description
        enhanced_detection['analysis_timestamp'] = self._get_timestamp()
        
        return enhanced_detection
    
    def get_object_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of detected objects
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Summary dictionary
        """
        if not detections:
            return {
                'total_objects': 0,
                'object_classes': [],
                'descriptions_available': 0
            }
        
        # Extract object classes
        object_classes = []
        descriptions_available = 0
        
        for detection in detections:
            class_name = detection.get('class_name', 'Unknown')
            if class_name not in object_classes:
                object_classes.append(class_name)
            
            if 'pixtral_description' in detection:
                descriptions_available += 1
        
        return {
            'total_objects': len(detections),
            'object_classes': object_classes,
            'descriptions_available': descriptions_available,
            'unique_classes': len(object_classes)
        }
    
    def filter_detections_by_class(self, detections: List[Dict[str, Any]], 
                                 target_class: str) -> List[Dict[str, Any]]:
        """
        Filter detections by class name
        
        Args:
            detections: List of detection dictionaries
            target_class: Target class name to filter by
            
        Returns:
            Filtered list of detections
        """
        if not detections or not target_class:
            return detections
        
        target_class_lower = target_class.lower()
        filtered = []
        
        for detection in detections:
            class_name = detection.get('class_name', '').lower()
            if target_class_lower in class_name or class_name in target_class_lower:
                filtered.append(detection)
        
        return filtered
    
    def get_detections_with_descriptions(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get only detections that have Pixtral descriptions
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of detections with descriptions
        """
        if not detections:
            return []
        
        return [d for d in detections if 'pixtral_description' in d and d['pixtral_description']]
    
    def _extract_detections(self, detection_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract detections from various detection result formats
        
        Args:
            detection_results: Detection results dictionary
            
        Returns:
            List of detection dictionaries
        """
        if not detection_results:
            return []
        
        # Try different possible keys for detections
        possible_keys = ['detections', 'objects', 'results', 'boxes']
        
        for key in possible_keys:
            if key in detection_results:
                detections = detection_results[key]
                if isinstance(detections, list):
                    return detections
        
        # If no detections found, return empty list
        return []
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and configuration
        
        Returns:
            Dictionary with system information
        """
        return {
            'pixtral_analyzer': self.pixtral_analyzer.get_system_info(),
            'api_key_configured': bool(self.api_key)
        }
