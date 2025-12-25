#!/usr/bin/env python3
"""
Advanced 3D Mapping System using Camera Calibration Parameters
Integrates with YOLO detection results for real-world distance and angle estimation
"""

import cv2
import numpy as np
import math
try:
    from core.config import (CAMERA_MATRIX, DISTORTION_COEFFICIENTS, CAMERA_FX, CAMERA_FY, 
                       CAMERA_CX, CAMERA_CY, REAL_OBJECT_HEIGHT, DISTANCE_CORRECTION_FACTOR, MEAN_REPROJECTION_ERROR)
except ImportError:
    from config import (CAMERA_MATRIX, DISTORTION_COEFFICIENTS, CAMERA_FX, CAMERA_FY, 
                       CAMERA_CX, CAMERA_CY, REAL_OBJECT_HEIGHT, DISTANCE_CORRECTION_FACTOR, MEAN_REPROJECTION_ERROR)

class Advanced3DMapper:
    """
    Advanced 3D mapping system using camera calibration parameters
    Provides distance and angle estimation for YOLO detection results
    """
    
    def __init__(self):
        # Convert camera matrix to numpy array
        self.camera_matrix = np.array(CAMERA_MATRIX, dtype=np.float32)
        self.dist_coeffs = np.array(DISTORTION_COEFFICIENTS, dtype=np.float32)
        
        # Extract camera parameters
        self.fx = CAMERA_FX
        self.fy = CAMERA_FY
        self.cx = CAMERA_CX
        self.cy = CAMERA_CY
        self.mean_reprojection_error = MEAN_REPROJECTION_ERROR
        
        # Object dimension database (in meters)
        self.object_dimensions = {
            # People
            'person': {'height': 1.7, 'width': 0.4},
            'man': {'height': 1.75, 'width': 0.4},
            'woman': {'height': 1.65, 'width': 0.4},
            'child': {'height': 1.2, 'width': 0.3},
            
            # Vehicles
            'car': {'height': 1.5, 'width': 1.8},
            'truck': {'height': 2.5, 'width': 2.5},
            'bus': {'height': 3.0, 'width': 2.5},
            'motorcycle': {'height': 1.1, 'width': 0.8},
            'bicycle': {'height': 1.0, 'width': 0.6},
            'van': {'height': 2.0, 'width': 2.0},
            'suv': {'height': 1.8, 'width': 1.8},
            
            # Animals
            'dog': {'height': 0.6, 'width': 0.3},
            'cat': {'height': 0.3, 'width': 0.2},
            'horse': {'height': 1.8, 'width': 0.6},
            'cow': {'height': 1.5, 'width': 0.8},
            
            # Furniture
            'chair': {'height': 0.9, 'width': 0.4},
            'table': {'height': 0.75, 'width': 1.2},
            'sofa': {'height': 0.8, 'width': 2.0},
            'bed': {'height': 0.6, 'width': 2.0},
            
            # Electronics
            'tv': {'height': 0.6, 'width': 1.0},
            'laptop': {'height': 0.3, 'width': 0.3},
            'monitor': {'height': 0.4, 'width': 0.5},
            'phone': {'height': 0.15, 'width': 0.07},
            
            # Common objects
            'bottle': {'height': 0.25, 'width': 0.08},
            'cup': {'height': 0.12, 'width': 0.08},
            'book': {'height': 0.025, 'width': 0.2},
            'backpack': {'height': 0.4, 'width': 0.3},
            
            # Default fallback
            'default': {'height': REAL_OBJECT_HEIGHT, 'width': 0.5}
        }
        
        print(f"Advanced 3D Mapper initialized:")
        print(f"  Camera Matrix: fx={self.fx:.2f}, fy={self.fy:.2f}")
        print(f"  Principal Point: cx={self.cx:.2f}, cy={self.cy:.2f}")
        print(f"  Distortion Coefficients: {len(self.dist_coeffs)} parameters")
        print(f"  Object database: {len(self.object_dimensions)} objects")
    
    def get_object_dimensions(self, class_name):
        """
        Get real-world dimensions for a detected object class
        
        Args:
            class_name: YOLO class name
            
        Returns:
            tuple: (height, width) in meters
        """
        # Try exact match first
        if class_name in self.object_dimensions:
            dims = self.object_dimensions[class_name]
            return dims['height'], dims['width']
        
        # Try partial matches
        for key, dims in self.object_dimensions.items():
            if key in class_name.lower() or class_name.lower() in key:
                return dims['height'], dims['width']
        
        # Return default dimensions
        dims = self.object_dimensions['default']
        return dims['height'], dims['width']
    
    def correct_lens_distortion(self, points):
        """
        Correct lens distortion for given pixel points
        
        Args:
            points: numpy array of shape (N, 2) containing (x, y) pixel coordinates
            
        Returns:
            numpy array: Undistorted pixel coordinates
        """
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
        
        # Convert to the format expected by cv2.undistortPoints
        points_float = points.astype(np.float32)
        
        # Undistort the points
        undistorted_points = cv2.undistortPoints(
            points_float.reshape(-1, 1, 2), 
            self.camera_matrix, 
            self.dist_coeffs, 
            P=self.camera_matrix
        )
        
        return undistorted_points.reshape(-1, 2)
    
    def pixel_to_angle(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to angles relative to camera optical center
        
        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            
        Returns:
            tuple: (horizontal_angle, vertical_angle) in degrees
        """
        # Calculate pixel offsets from principal point
        dx = pixel_x - self.cx
        dy = pixel_y - self.cy
        
        # Convert to angles using pinhole camera model
        horizontal_angle = math.degrees(math.atan(dx / self.fx))
        vertical_angle = math.degrees(math.atan(dy / self.fy))
        
        return horizontal_angle, vertical_angle
    
    def calculate_distance_from_height(self, bbox_height_pixels, real_height):
        """
        Calculate distance using object height and bounding box height
        
        Args:
            bbox_height_pixels: Height of bounding box in pixels
            real_height: Real-world height of object in meters
            
        Returns:
            float: Distance in meters
        """
        if bbox_height_pixels <= 0:
            return float('inf')
        
        # Use similar triangles: distance = (real_height * focal_length) / pixel_height
        raw_distance = (real_height * self.fy) / bbox_height_pixels
        
        # Apply correction factor
        corrected_distance = raw_distance * DISTANCE_CORRECTION_FACTOR
        
        return corrected_distance
    
    def calculate_distance_from_width(self, bbox_width_pixels, real_width):
        """
        Calculate distance using object width and bounding box width
        
        Args:
            bbox_width_pixels: Width of bounding box in pixels
            real_width: Real-world width of object in meters
            
        Returns:
            float: Distance in meters
        """
        if bbox_width_pixels <= 0:
            return float('inf')
        
        # Use similar triangles: distance = (real_width * focal_length) / pixel_width
        raw_distance = (real_width * self.fx) / bbox_width_pixels
        
        # Apply correction factor
        corrected_distance = raw_distance * DISTANCE_CORRECTION_FACTOR
        
        return corrected_distance
    
    def calculate_distance_combined(self, bbox, real_height, real_width):
        """
        Calculate distance using both height and width, return the most reliable estimate
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            real_height: Real-world height in meters
            real_width: Real-world width in meters
            
        Returns:
            dict: Distance calculation results
        """
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        
        # Calculate distances using both methods
        distance_height = self.calculate_distance_from_height(bbox_height, real_height)
        distance_width = self.calculate_distance_from_width(bbox_width, real_width)
        
        # Calculate aspect ratio to determine which method is more reliable
        bbox_aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1
        expected_aspect_ratio = real_width / real_height if real_height > 0 else 1
        
        # Choose the more reliable method based on aspect ratio similarity
        aspect_ratio_diff = abs(bbox_aspect_ratio - expected_aspect_ratio)
        
        if aspect_ratio_diff < 0.3:  # Similar aspect ratios, use average
            final_distance = (distance_height + distance_width) / 2
            method = "combined"
        elif bbox_aspect_ratio < expected_aspect_ratio:  # Object appears taller, prefer height
            final_distance = distance_height
            method = "height"
        else:  # Object appears wider, prefer width
            final_distance = distance_width
            method = "width"
        
        return {
            'distance': final_distance,
            'distance_height': distance_height,
            'distance_width': distance_width,
            'method': method,
            'bbox_height': bbox_height,
            'bbox_width': bbox_width,
            'aspect_ratio': bbox_aspect_ratio,
            'expected_aspect_ratio': expected_aspect_ratio
        }
    
    def process_yolo_detection(self, bbox, class_name, confidence, frame_shape):
        """
        Process a single YOLO detection and return 3D mapping results
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            class_name: YOLO class name
            confidence: Detection confidence
            frame_shape: (height, width) of the frame
            
        Returns:
            dict: Complete 3D mapping results
        """
        frame_height, frame_width = frame_shape[:2]
        
        # Get object dimensions
        real_height, real_width = self.get_object_dimensions(class_name)
        
        # Calculate bounding box center
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Correct lens distortion for center point
        center_point = np.array([[center_x, center_y]], dtype=np.float32)
        undistorted_center = self.correct_lens_distortion(center_point)[0]
        
        # Calculate angles
        horizontal_angle, vertical_angle = self.pixel_to_angle(
            undistorted_center[0], undistorted_center[1]
        )
        
        # Calculate distance
        distance_results = self.calculate_distance_combined(bbox, real_height, real_width)
        
        # Check for edge cases
        edge_case_warnings = self._check_edge_cases(bbox, frame_width, frame_height, distance_results)
        
        return {
            'class_name': class_name,
            'confidence': confidence,
            'bbox': bbox,
            'center_pixel': (center_x, center_y),
            'center_undistorted': undistorted_center,
            'distance': distance_results['distance'],
            'distance_height': distance_results['distance_height'],
            'distance_width': distance_results['distance_width'],
            'distance_method': distance_results['method'],
            'horizontal_angle': horizontal_angle,
            'vertical_angle': vertical_angle,
            'real_dimensions': {'height': real_height, 'width': real_width},
            'bbox_dimensions': {'height': distance_results['bbox_height'], 'width': distance_results['bbox_width']},
            'aspect_ratios': {
                'bbox': distance_results['aspect_ratio'],
                'expected': distance_results['expected_aspect_ratio']
            },
            'edge_case_warnings': edge_case_warnings
        }
    
    def _check_edge_cases(self, bbox, frame_width, frame_height, distance_results):
        """
        Check for edge cases and potential issues
        
        Args:
            bbox: Bounding box coordinates
            frame_width: Frame width
            frame_height: Frame height
            distance_results: Distance calculation results
            
        Returns:
            list: List of warning messages
        """
        warnings = []
        x1, y1, x2, y2 = bbox
        
        # Check if bounding box is near frame edges
        margin = 50  # pixels
        if x1 < margin or y1 < margin or x2 > frame_width - margin or y2 > frame_height - margin:
            warnings.append("Bounding box near frame edge - reduced accuracy")
        
        # Check for very small bounding boxes
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area < 1000:  # Less than 1000 pixels
            warnings.append("Very small bounding box - distance may be inaccurate")
        
        # Check for extreme aspect ratios
        aspect_ratio = distance_results['aspect_ratio']
        if aspect_ratio > 3 or aspect_ratio < 0.3:
            warnings.append("Extreme aspect ratio - object may be partially occluded")
        
        # Check for very large distance differences between methods
        height_dist = distance_results['distance_height']
        width_dist = distance_results['distance_width']
        if height_dist != float('inf') and width_dist != float('inf'):
            diff_ratio = abs(height_dist - width_dist) / max(height_dist, width_dist)
            if diff_ratio > 0.5:  # More than 50% difference
                warnings.append("Large difference between height/width distance estimates")
        
        # Check for unrealistic distances
        final_distance = distance_results['distance']
        if final_distance > 50:  # More than 50 meters
            warnings.append("Very large distance estimate - may be inaccurate")
        elif final_distance < 0.1:  # Less than 10 cm
            warnings.append("Very small distance estimate - object may be too close")
        
        return warnings
    
    def process_multiple_detections(self, detections, frame_shape):
        """
        Process multiple YOLO detections
        
        Args:
            detections: List of detection dictionaries
            frame_shape: (height, width) of the frame
            
        Returns:
            list: List of 3D mapping results
        """
        results = []
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            result = self.process_yolo_detection(bbox, class_name, confidence, frame_shape)
            results.append(result)
        
        return results
    
    def get_distance_category(self, distance):
        """
        Categorize distance into ranges with appropriate colors
        
        Args:
            distance: Distance in meters
            
        Returns:
            tuple: (category_name, color_bgr)
        """
        if distance < 0.5:
            return "Very Close", (0, 0, 255)  # Red
        elif distance < 1.0:
            return "Close", (0, 165, 255)  # Orange
        elif distance < 2.0:
            return "Medium", (0, 255, 255)  # Yellow
        elif distance < 5.0:
            return "Far", (0, 255, 0)  # Green
        else:
            return "Very Far", (255, 0, 0)  # Blue
