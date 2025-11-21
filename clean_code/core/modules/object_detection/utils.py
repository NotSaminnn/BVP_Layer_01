#!/usr/bin/env python3
"""
Standalone Utility Functions for Object Detection with Distance and Angle Mapping

This module provides standalone utility functions that can be imported and used
individually without requiring the full detection system.

Usage:
    from object_detection_distance_angle.utils import calculate_distance, calculate_angles
    
    distance = calculate_distance(bbox_height_pixels, real_height=1.7)
    angle_x, angle_y = calculate_angles(center_x, center_y, frame_width, frame_height)
"""

import cv2
import numpy as np
import math
from typing import Tuple, List, Dict, Any, Optional, Union
# Handle both relative and absolute imports
try:
    from core.config import CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY, REAL_OBJECT_HEIGHT, DISTANCE_CORRECTION_FACTOR
except ImportError:
    # Fallback for direct execution
    from core.config import CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY, REAL_OBJECT_HEIGHT, DISTANCE_CORRECTION_FACTOR

# ============================================================================
# Distance and Angle Calculation Functions
# ============================================================================

def calculate_distance(bbox_height_pixels: float, 
                      real_height: float = REAL_OBJECT_HEIGHT,
                      focal_length: float = CAMERA_FY,
                      correction_factor: float = DISTANCE_CORRECTION_FACTOR) -> float:
    """
    Calculate distance using similar triangles with correction factor.
    
    Args:
        bbox_height_pixels: Height of bounding box in pixels
        real_height: Real-world height of object in meters
        focal_length: Camera focal length (default: CAMERA_FY)
        correction_factor: Distance correction factor
        
    Returns:
        Distance in meters (corrected)
        
    Raises:
        ValueError: If bbox_height_pixels is invalid
    """
    if bbox_height_pixels <= 0:
        return float('inf')
    
    # Calculate raw distance using similar triangles
    raw_distance = (real_height * focal_length) / bbox_height_pixels
    
    # Apply correction factor
    corrected_distance = raw_distance * correction_factor
    
    return corrected_distance

def calculate_angles(center_x: float, center_y: float, 
                    frame_width: int, frame_height: int,
                    focal_x: float = CAMERA_FX, focal_y: float = CAMERA_FY,
                    principal_x: float = CAMERA_CX, principal_y: float = CAMERA_CY) -> Tuple[float, float]:
    """
    Calculate horizontal and vertical angles from pixel coordinates.
    
    Args:
        center_x: X coordinate of object center
        center_y: Y coordinate of object center
        frame_width: Width of the frame
        frame_height: Height of the frame
        focal_x: Focal length in x direction
        focal_y: Focal length in y direction
        principal_x: Principal point x coordinate
        principal_y: Principal point y coordinate
        
    Returns:
        Tuple of (horizontal_angle, vertical_angle) in degrees
        
    Raises:
        ValueError: If frame dimensions are invalid
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"Invalid frame dimensions: {frame_width}x{frame_height}")
    
    # Calculate pixel offsets from principal point
    pixel_offset_x = center_x - principal_x
    pixel_offset_y = center_y - principal_y
    
    # Calculate angles using pinhole camera model
    horizontal_angle = math.degrees(math.atan(pixel_offset_x / focal_x))
    vertical_angle = math.degrees(math.atan(pixel_offset_y / focal_y))
    
    return horizontal_angle, vertical_angle

def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Get center coordinates of bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple of (center_x, center_y)
        
    Raises:
        ValueError: If bbox format is invalid
    """
    if len(bbox) != 4:
        raise ValueError(f"Bbox must have 4 elements [x1, y1, x2, y2], got {len(bbox)}")
    
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    return center_x, center_y

def get_bbox_dimensions(bbox: List[float]) -> Tuple[float, float]:
    """
    Get width and height of bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple of (width, height) in pixels
        
    Raises:
        ValueError: If bbox format is invalid
    """
    if len(bbox) != 4:
        raise ValueError(f"Bbox must have 4 elements [x1, y1, x2, y2], got {len(bbox)}")
    
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    return width, height

# ============================================================================
# Object Dimension Database
# ============================================================================

def get_object_dimensions(class_name: str) -> Tuple[float, float]:
    """
    Get real-world dimensions for a detected object class.
    
    Args:
        class_name: YOLO class name
        
    Returns:
        Tuple of (height, width) in meters
    """
    # Object dimension database (in meters)
    object_dimensions = {
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
    }
    
    # Try exact match first
    if class_name in object_dimensions:
        dims = object_dimensions[class_name]
        return dims['height'], dims['width']
    
    # Try partial matches
    for key, dims in object_dimensions.items():
        if key in class_name.lower() or class_name.lower() in key:
            return dims['height'], dims['width']
    
    # Return default dimensions
    return REAL_OBJECT_HEIGHT, 0.5

# ============================================================================
# Visualization Functions
# ============================================================================

def get_distance_color(distance: float) -> Tuple[str, Tuple[int, int, int]]:
    """
    Get color based on distance category.
    
    Args:
        distance: Distance in meters
        
    Returns:
        Tuple of (category_name, color_bgr)
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

def draw_simple_detection(frame: np.ndarray, bbox: List[float], 
                         class_name: str, confidence: float, 
                         distance: float, angle_x: float, angle_y: float,
                         color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Draw simple detection information on frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        class_name: Object class name
        confidence: Detection confidence
        distance: Distance in meters
        angle_x: Horizontal angle in degrees
        angle_y: Vertical angle in degrees
        color: Optional color override
        
    Returns:
        Annotated frame
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Get color based on distance if not provided
    if color is None:
        _, color = get_distance_color(distance)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Prepare text
    info_text = f"{class_name} | {distance:.1f}m | H:{angle_x:.0f}° V:{angle_y:.0f}°"
    
    # Draw text above bounding box
    text_x = x1
    text_y = y1 - 10
    
    # Ensure text doesn't go off screen
    if text_y < 20:
        text_y = y2 + 20
    
    # Draw text with outline for better visibility
    cv2.putText(frame, info_text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # Black outline
    cv2.putText(frame, info_text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  # Colored text
    
    return frame

# ============================================================================
# Data Processing Functions
# ============================================================================

def process_yolo_detection(bbox: List[float], class_name: str, confidence: float,
                          frame_shape: Tuple[int, int]) -> Dict[str, Any]:
    """
    Process a single YOLO detection and return 3D mapping results.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        class_name: YOLO class name
        confidence: Detection confidence
        frame_shape: (height, width) of the frame
        
    Returns:
        Dictionary with 3D mapping results
    """
    frame_height, frame_width = frame_shape[:2]
    
    # Get object dimensions
    real_height, real_width = get_object_dimensions(class_name)
    
    # Get bounding box properties
    center_x, center_y = get_bbox_center(bbox)
    bbox_width, bbox_height = get_bbox_dimensions(bbox)
    
    # Calculate distance
    distance = calculate_distance(bbox_height, real_height)
    
    # Calculate angles
    angle_x, angle_y = calculate_angles(center_x, center_y, frame_width, frame_height)
    
    return {
        'class_name': class_name,
        'confidence': confidence,
        'bbox': bbox,
        'center_pixel': (center_x, center_y),
        'distance': distance,
        'horizontal_angle': angle_x,
        'vertical_angle': angle_y,
        'real_dimensions': {'height': real_height, 'width': real_width},
        'bbox_dimensions': {'height': bbox_height, 'width': bbox_width}
    }

def create_detection_summary(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of detection results.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary with summary statistics
    """
    if not detections:
        return {'total_detections': 0}
    
    distances = [d['distance'] for d in detections if d['distance'] != float('inf')]
    angles_h = [d['horizontal_angle'] for d in detections]
    angles_v = [d['vertical_angle'] for d in detections]
    
    summary = {
        'total_detections': len(detections),
        'valid_distances': len(distances),
        'distance_stats': {
            'min': min(distances) if distances else 0,
            'max': max(distances) if distances else 0,
            'mean': np.mean(distances) if distances else 0,
            'std': np.std(distances) if distances else 0
        },
        'angle_stats': {
            'horizontal_range': (min(angles_h), max(angles_h)),
            'vertical_range': (min(angles_v), max(angles_v))
        },
        'object_classes': list(set([d['class_name'] for d in detections]))
    }
    
    return summary

# ============================================================================
# File I/O Functions
# ============================================================================

def save_detection_data(detections: List[Dict[str, Any]], filename: str) -> None:
    """
    Save detection data to file.
    
    Args:
        detections: List of detection dictionaries
        filename: Output filename
    """
    with open(filename, 'w') as f:
        f.write("Detection Data\n")
        f.write("=" * 50 + "\n")
        for i, detection in enumerate(detections):
            f.write(f"Detection {i+1}:\n")
            f.write(f"  Class: {detection['class_name']}\n")
            f.write(f"  Confidence: {detection['confidence']:.3f}\n")
            f.write(f"  Distance: {detection['distance']:.2f}m\n")
            f.write(f"  Angle X: {detection['horizontal_angle']:.1f}°\n")
            f.write(f"  Angle Y: {detection['vertical_angle']:.1f}°\n")
            f.write(f"  Bounding Box: {detection['bbox']}\n")
            f.write("-" * 30 + "\n")

def load_detection_data(filename: str) -> List[Dict[str, Any]]:
    """
    Load detection data from file.
    
    Args:
        filename: Input filename
        
    Returns:
        List of detection dictionaries
    """
    detections = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    current_detection = {}
    for line in lines:
        line = line.strip()
        if line.startswith("Detection"):
            if current_detection:
                detections.append(current_detection)
            current_detection = {}
        elif line.startswith("Class:"):
            current_detection['class_name'] = line.split(":", 1)[1].strip()
        elif line.startswith("Confidence:"):
            current_detection['confidence'] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Distance:"):
            current_detection['distance'] = float(line.split(":", 1)[1].strip().replace("m", ""))
        elif line.startswith("Angle X:"):
            current_detection['horizontal_angle'] = float(line.split(":", 1)[1].strip().replace("°", ""))
        elif line.startswith("Angle Y:"):
            current_detection['vertical_angle'] = float(line.split(":", 1)[1].strip().replace("°", ""))
        elif line.startswith("Bounding Box:"):
            bbox_str = line.split(":", 1)[1].strip()
            # Parse bounding box from string representation
            bbox_str = bbox_str.replace("[", "").replace("]", "")
            bbox = [float(x.strip()) for x in bbox_str.split(",")]
            current_detection['bbox'] = bbox
    
    if current_detection:
        detections.append(current_detection)
    
    return detections

# ============================================================================
# Validation Functions
# ============================================================================

def validate_bbox(bbox: List[float]) -> bool:
    """
    Validate bounding box format and values.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        True if valid, False otherwise
    """
    if len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    
    # Check if coordinates are valid
    if x1 >= x2 or y1 >= y2:
        return False
    
    # Check if coordinates are non-negative
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return False
    
    return True

def validate_frame(frame: np.ndarray) -> bool:
    """
    Validate frame format and content.
    
    Args:
        frame: Input frame
        
    Returns:
        True if valid, False otherwise
    """
    if frame is None:
        return False
    
    if frame.size == 0:
        return False
    
    if len(frame.shape) != 3:
        return False
    
    if frame.shape[2] != 3:
        return False
    
    return True

# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of how to use the utility functions."""
    
    # Example bounding box
    bbox = [100, 100, 200, 300]  # [x1, y1, x2, y2]
    
    # Calculate distance
    distance = calculate_distance(bbox[3] - bbox[1], real_height=1.7)
    print(f"Distance: {distance:.2f}m")
    
    # Calculate angles
    center_x, center_y = get_bbox_center(bbox)
    angle_x, angle_y = calculate_angles(center_x, center_y, 1280, 720)
    print(f"Angles: H={angle_x:.1f}°, V={angle_y:.1f}°")
    
    # Get object dimensions
    height, width = get_object_dimensions("person")
    print(f"Person dimensions: {height}m x {width}m")
    
    # Process detection
    detection = process_yolo_detection(bbox, "person", 0.85, (720, 1280))
    print(f"Detection result: {detection}")

if __name__ == "__main__":
    example_usage()
