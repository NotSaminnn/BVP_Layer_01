#!/usr/bin/env python3
"""
Visualization utilities for 3D mapping results
"""

import cv2
import numpy as np
try:
    from core.config import TEXT_COLOR, TEXT_SCALE, TEXT_THICKNESS
except ImportError:
    from config import TEXT_COLOR, TEXT_SCALE, TEXT_THICKNESS

class VisualizationUtils:
    """Utilities for visualizing 3D mapping results"""
    
    def __init__(self):
        self.colors = {
            'very_close': (0, 0, 255),    # Red
            'close': (0, 165, 255),       # Orange
            'medium': (0, 255, 255),      # Yellow
            'far': (0, 255, 0),           # Green
            'very_far': (255, 0, 0),      # Blue
            'warning': (0, 0, 255),       # Red for warnings
            'info': (255, 255, 255)       # White for info
        }
    
    def draw_detection_with_3d_info(self, frame, detection_result):
        """
        Draw bounding box and simple 3D information for a single detection
        
        Args:
            frame: Input frame
            detection_result: Result from Advanced3DMapper.process_yolo_detection
            
        Returns:
            numpy array: Annotated frame
        """
        bbox = detection_result['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get distance category and color
        distance_category, category_color = self._get_distance_category_color(
            detection_result['distance']
        )
        
        # Draw bounding box with distance-based color
        cv2.rectangle(frame, (x1, y1), (x2, y2), category_color, 2)
        
        # Prepare simple information text (just class name and confidence)
        info_text = f"{detection_result['class_name']} ({detection_result.get('confidence', 0.0):.2f})"
        
        # Draw text above bounding box (simple, no background)
        text_x = x1
        text_y = y1 - 10
        
        # Ensure text doesn't go off screen
        if text_y < 20:
            text_y = y2 + 20
        
        # Draw text with outline for better visibility
        cv2.putText(frame, info_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # Black outline
        cv2.putText(frame, info_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, category_color, 2)  # Colored text
        
        return frame
    
    def _get_distance_category_color(self, distance):
        """Get color based on distance category"""
        if distance < 0.5:
            return "Very Close", self.colors['very_close']
        elif distance < 1.0:
            return "Close", self.colors['close']
        elif distance < 2.0:
            return "Medium", self.colors['medium']
        elif distance < 5.0:
            return "Far", self.colors['far']
        else:
            return "Very Far", self.colors['very_far']
    
    def _draw_info_panel(self, frame, info_lines, position, color):
        """Draw information panel with background"""
        x, y = position
        text_height = 20
        text_spacing = 5
        
        # Calculate panel dimensions
        max_width = 0
        for line in info_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            max_width = max(max_width, text_size[0])
        
        panel_height = len(info_lines) * (text_height + text_spacing) + text_spacing
        panel_width = max_width + 20
        
        # Position panel (above bounding box, or below if not enough space)
        panel_x = x
        panel_y = y - panel_height - 10
        
        if panel_y < 0:
            panel_y = y + 10
        
        # Draw background
        cv2.rectangle(frame, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     color, 2)
        
        # Draw text
        for i, line in enumerate(info_lines):
            text_y = panel_y + (i + 1) * (text_height + text_spacing)
            
            # Use warning color for warning lines
            text_color = self.colors['warning'] if line.startswith('⚠️') else color
            
            cv2.putText(frame, line, 
                       (panel_x + 10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    def draw_distance_legend(self, frame):
        """Draw distance category legend"""
        legend_items = [
            ("Very Close (< 0.5m)", self.colors['very_close']),
            ("Close (0.5-1.0m)", self.colors['close']),
            ("Medium (1.0-2.0m)", self.colors['medium']),
            ("Far (2.0-5.0m)", self.colors['far']),
            ("Very Far (> 5.0m)", self.colors['very_far'])
        ]
        
        start_x = 10
        start_y = 30
        item_height = 25
        
        for i, (text, color) in enumerate(legend_items):
            y = start_y + i * item_height
            
            # Draw color box
            cv2.rectangle(frame, (start_x, y - 15), (start_x + 20, y - 5), color, -1)
            
            # Draw text
            cv2.putText(frame, text, (start_x + 25, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['info'], 2)
    
    def draw_camera_info(self, frame, mapper):
        """Draw camera calibration information"""
        info_lines = [
            f"Camera: fx={mapper.fx:.1f}, fy={mapper.fy:.1f}",
            f"Principal: cx={mapper.cx:.1f}, cy={mapper.cy:.1f}",
            f"Reprojection Error: {mapper.mean_reprojection_error:.3f}"
        ]
        
        frame_height = frame.shape[0]
        start_y = frame_height - 80
        
        for i, line in enumerate(info_lines):
            y = start_y + i * 20
            cv2.putText(frame, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info'], 1)
    
    def create_summary_panel(self, frame, detection_results):
        """Create summary panel with detection statistics"""
        if not detection_results:
            return frame
        
        # Calculate statistics
        distances = [r['distance'] for r in detection_results if r['distance'] != float('inf')]
        avg_distance = np.mean(distances) if distances else 0
        min_distance = min(distances) if distances else 0
        max_distance = max(distances) if distances else 0
        
        # Create summary text
        summary_lines = [
            f"Detections: {len(detection_results)}",
            f"Avg Distance: {avg_distance:.2f}m",
            f"Range: {min_distance:.2f}m - {max_distance:.2f}m"
        ]
        
        # Draw summary panel
        frame_width = frame.shape[1]
        panel_width = 200
        panel_height = len(summary_lines) * 25 + 20
        panel_x = frame_width - panel_width - 10
        panel_y = 10
        
        # Background
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (255, 255, 255), 2)
        
        # Text
        for i, line in enumerate(summary_lines):
            y = panel_y + (i + 1) * 25
            cv2.putText(frame, line, (panel_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['info'], 2)
        
        return frame
