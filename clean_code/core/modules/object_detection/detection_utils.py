import cv2
import numpy as np
import math
try:
    from core.config import CAMERA_FX, CAMERA_FY, REAL_OBJECT_HEIGHT, DISTANCE_CORRECTION_FACTOR
except ImportError:
    from config import CAMERA_FX, CAMERA_FY, REAL_OBJECT_HEIGHT, DISTANCE_CORRECTION_FACTOR

class DistanceAngleCalculator:
    """Utility class for calculating distance and angles from camera calibration data"""
    
    def __init__(self, fx=CAMERA_FX, fy=CAMERA_FY, real_height=REAL_OBJECT_HEIGHT):
        self.fx = fx
        self.fy = fy
        self.real_height = real_height
        self.distance_correction_factor = DISTANCE_CORRECTION_FACTOR
    
    def calculate_distance(self, bbox_height_pixels):
        """
        Calculate distance using similar triangles with correction factor
        
        Args:
            bbox_height_pixels: Height of bounding box in pixels
            
        Returns:
            distance: Distance in meters (corrected)
        """
        if bbox_height_pixels <= 0:
            return float('inf')
        
        # Calculate raw distance
        raw_distance = (self.real_height * self.fy) / bbox_height_pixels
        
        # Apply correction factor (camera maps 30cm actual to 3m detected)
        corrected_distance = raw_distance * self.distance_correction_factor
        
        return corrected_distance
    
    def calculate_angles(self, center_x, center_y, frame_width, frame_height):
        """
        Calculate horizontal and vertical angles
        
        Args:
            center_x: X coordinate of object center
            center_y: Y coordinate of object center
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            angle_x: Horizontal angle in degrees
            angle_y: Vertical angle in degrees
        """
        # Principal point (center of image)
        cx = frame_width / 2
        cy = frame_height / 2
        
        # Calculate pixel offsets
        pixel_offset_x = center_x - cx
        pixel_offset_y = center_y - cy
        
        # Calculate angles
        angle_x = math.degrees(math.atan(pixel_offset_x / self.fx))
        angle_y = math.degrees(math.atan(pixel_offset_y / self.fy))
        
        return angle_x, angle_y
    
    def get_object_center(self, bbox):
        """
        Get center coordinates of bounding box
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            center_x, center_y: Center coordinates
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y
    
    def get_bbox_height(self, bbox):
        """
        Get height of bounding box
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            height: Height in pixels
        """
        x1, y1, x2, y2 = bbox
        return y2 - y1

def draw_info_panel(frame, detections, frame_width, frame_height):
    """
    Draw information panel with detection summary
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        Annotated frame
    """
    if not detections:
        return frame
    
    # Create info panel
    panel_height = 100
    panel_width = 300
    panel_x = frame_width - panel_width - 10
    panel_y = 10
    
    # Draw background panel
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (255, 255, 255), 2)
    
    # Draw title
    cv2.putText(frame, "Detection Summary", 
               (panel_x + 10, panel_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw detection info
    y_offset = 45
    for i, detection in enumerate(detections[:3]):  # Show max 3 detections
        info = f"{detection['class']}: {detection['distance']:.1f}m"
        cv2.putText(frame, info, 
                   (panel_x + 10, panel_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 20
    
    return frame

def save_detection_data(detections, filename="detection_data.txt"):
    """
    Save detection data to file
    
    Args:
        detections: List of detection dictionaries
        filename: Output filename
    """
    with open(filename, 'w') as f:
        f.write("Detection Data\n")
        f.write("=" * 50 + "\n")
        for i, detection in enumerate(detections):
            f.write(f"Detection {i+1}:\n")
            f.write(f"  Class: {detection['class']}\n")
            f.write(f"  Confidence: {detection['confidence']:.3f}\n")
            f.write(f"  Distance: {detection['distance']:.2f}m\n")
            f.write(f"  Angle X: {detection['angle_x']:.1f}°\n")
            f.write(f"  Angle Y: {detection['angle_y']:.1f}°\n")
            f.write(f"  Bounding Box: {detection['bbox']}\n")
            f.write("-" * 30 + "\n")
