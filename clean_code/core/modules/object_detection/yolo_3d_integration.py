#!/usr/bin/env python3
"""
YOLO 3D Integration Module
Production-ready integration function for YOLO inference pipeline
"""

import cv2
import numpy as np
from typing import Optional
try:
    from core.advanced_3d_mapping import Advanced3DMapper
    from core.infrastructure.visualization_utils import VisualizationUtils
except ImportError:
    from advanced_3d_mapping import Advanced3DMapper
    from visualization_utils import VisualizationUtils
try:
    from core.config import CONFIDENCE_THRESHOLD
except ImportError:
    from config import CONFIDENCE_THRESHOLD

class YOLO3DIntegration:
    """
    Production-ready integration class for YOLO + 3D mapping
    """
    
    def __init__(self, yolo_model_path="yolo11n_object365.pt", device: Optional[str] = None):
        """
        Initialize the YOLO 3D integration system
        
        Args:
            yolo_model_path: Path to YOLO model file
            device: Device to use ('cuda', 'mps', or 'cpu'). If None, auto-detect.
        """
        from ultralytics import YOLO
        import torch
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Move model to device if GPU available (for PyTorch models)
        if self.device != 'cpu' and hasattr(self.yolo_model, 'model'):
            try:
                if hasattr(self.yolo_model.model, 'to'):
                    self.yolo_model.model.to(self.device)
                self.yolo_model.device = self.device
            except Exception as e:
                print(f"⚠️  Warning: Could not move model to {self.device}: {e}")
                print(f"   Continuing with CPU...")
                self.device = 'cpu'
        
        # Initialize 3D mapper
        self.mapper = Advanced3DMapper()
        
        # Initialize visualization utils
        self.visualizer = VisualizationUtils()
        
        print("YOLO 3D Integration initialized successfully")
        print(f"YOLO model: {yolo_model_path}")
        print(f"Device: {self.device.upper()}")
        print(f"Camera calibration loaded with reprojection error: {self.mapper.mean_reprojection_error:.3f}")
    
    def process_frame(self, frame, confidence_threshold=CONFIDENCE_THRESHOLD, 
                     show_visualization=True, show_legend=True, show_camera_info=True):
        """
        Process a single frame with YOLO detection and 3D mapping
        
        Args:
            frame: Input frame (numpy array)
            confidence_threshold: Minimum confidence for detections
            show_visualization: Whether to draw visualization
            show_legend: Whether to show distance legend
            show_camera_info: Whether to show camera calibration info
            
        Returns:
            dict: Processing results
        """
        # Run YOLO detection with device specification
        results = self.yolo_model(frame, device=self.device, verbose=False)
        
        # Extract detections
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get detection data
                    bbox = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_name = self.yolo_model.names[class_id]
                    
                    # Filter by confidence
                    if confidence > confidence_threshold:
                        detections.append({
                            'bbox': bbox,
                            'class': class_name,
                            'confidence': confidence
                        })
        
        # Process with 3D mapping
        frame_shape = frame.shape
        mapping_results = self.mapper.process_multiple_detections(detections, frame_shape)
        
        # Create output frame
        output_frame = frame.copy()
        
        if show_visualization:
            # Draw each detection with simple visualization
            for result in mapping_results:
                output_frame = self.visualizer.draw_detection_with_3d_info(output_frame, result)
        
        return {
            'frame': output_frame,
            'detections': detections,
            'mapping_results': mapping_results,
            'frame_shape': frame_shape
        }
    
    def process_video_stream(self, video_source=0, confidence_threshold=CONFIDENCE_THRESHOLD):
        """
        Process video stream with real-time 3D mapping
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
            confidence_threshold: Minimum confidence for detections
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Set video resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting video stream processing...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'd' - Toggle distance legend")
        print("  'c' - Toggle camera info")
        print("  'v' - Toggle visualization")
        
        show_legend = True
        show_camera_info = True
        show_visualization = True
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            result = self.process_frame(
                frame, 
                confidence_threshold=confidence_threshold,
                show_visualization=show_visualization,
                show_legend=show_legend,
                show_camera_info=show_camera_info
            )
            
            # Display frame
            cv2.imshow('YOLO 3D Mapping', result['frame'])
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"yolo_3d_frame_{frame_count}.jpg"
                cv2.imwrite(filename, result['frame'])
                print(f"Frame saved as {filename}")
            elif key == ord('d'):
                show_legend = not show_legend
                print(f"Distance legend: {'ON' if show_legend else 'OFF'}")
            elif key == ord('c'):
                show_camera_info = not show_camera_info
                print(f"Camera info: {'ON' if show_camera_info else 'OFF'}")
            elif key == ord('v'):
                show_visualization = not show_visualization
                print(f"Visualization: {'ON' if show_visualization else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")

def integrate_with_yolo_pipeline(yolo_results, frame_shape):
    """
    Standalone function to integrate 3D mapping with existing YOLO pipeline
    
    Args:
        yolo_results: YOLO model results
        frame_shape: (height, width) of the frame
        
    Returns:
        list: 3D mapping results
    """
    mapper = Advanced3DMapper()
    
    # Extract detections from YOLO results
    detections = []
    for result in yolo_results:
        if result.boxes is not None:
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = result.names[class_id]
                
                detections.append({
                    'bbox': bbox,
                    'class': class_name,
                    'confidence': confidence
                })
    
    # Process with 3D mapping
    mapping_results = mapper.process_multiple_detections(detections, frame_shape)
    
    return mapping_results

def create_detection_summary(mapping_results):
    """
    Create a summary of detection results
    
    Args:
        mapping_results: List of 3D mapping results
        
    Returns:
        dict: Summary statistics
    """
    if not mapping_results:
        return {'total_detections': 0}
    
    distances = [r['distance'] for r in mapping_results if r['distance'] != float('inf')]
    angles_h = [r['horizontal_angle'] for r in mapping_results]
    angles_v = [r['vertical_angle'] for r in mapping_results]
    
    summary = {
        'total_detections': len(mapping_results),
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
        'object_classes': list(set([r['class_name'] for r in mapping_results])),
        'warnings': [w for r in mapping_results for w in r['edge_case_warnings']]
    }
    
    return summary

# Example usage function
def example_usage():
    """Example of how to use the YOLO 3D integration"""
    
    # Initialize the integration system
    yolo_3d = YOLO3DIntegration("yolo11n_object365.pt")
    
    # Option 1: Process video stream
    yolo_3d.process_video_stream(video_source=0, confidence_threshold=0.5)
    
    # Option 2: Process single frame
    # frame = cv2.imread("test_image.jpg")
    # result = yolo_3d.process_frame(frame)
    # cv2.imshow("Result", result['frame'])
    # cv2.waitKey(0)
    
    # Option 3: Integrate with existing YOLO pipeline
    # yolo_model = YOLO("yolo11n_object365.pt")
    # results = yolo_model(frame)
    # mapping_results = integrate_with_yolo_pipeline(results, frame.shape)
    # summary = create_detection_summary(mapping_results)

if __name__ == "__main__":
    example_usage()
