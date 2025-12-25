#!/usr/bin/env python3
"""
Object Detection with Distance and Angle Calculation - Main Module

This module provides the main ObjectDetectionWithDistanceAngle class for
real-time object detection with 3D mapping capabilities.

Usage:
    from object_detection_distance_angle import ObjectDetectionWithDistanceAngle
    
    detector = ObjectDetectionWithDistanceAngle()
    detector.process_video()
"""

import cv2
import torch
import numpy as np
import os
import sys
import time
import threading
import queue
from typing import List, Dict, Any, Optional, Union, Set

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found. Please install with: pip install ultralytics")
    sys.exit(1)

# Handle both relative and absolute imports
try:
    from core.advanced_3d_mapping import Advanced3DMapper
    from core.infrastructure.visualization_utils import VisualizationUtils
    from core.detection_utils import save_detection_data
    from core.config import (CAMERA_FX, CAMERA_FY, CONFIDENCE_THRESHOLD, BOUNDING_BOX_COLOR, 
                       TEXT_COLOR, TEXT_SCALE, TEXT_THICKNESS, DEFAULT_VIDEO_SOURCE, 
                       VIDEO_WIDTH, VIDEO_HEIGHT, SHOW_FPS, DISTANCE_CORRECTION_FACTOR,
                       DEFAULT_MODEL_PATH, GUI_SUPPORT_REQUIRED)
except ImportError:
    # Fallback for direct execution
    from advanced_3d_mapping import Advanced3DMapper
    from visualization_utils import VisualizationUtils
    from detection_utils import save_detection_data
    from config import (CAMERA_FX, CAMERA_FY, CONFIDENCE_THRESHOLD, BOUNDING_BOX_COLOR, 
                       TEXT_COLOR, TEXT_SCALE, TEXT_THICKNESS, DEFAULT_VIDEO_SOURCE, 
                       VIDEO_WIDTH, VIDEO_HEIGHT, SHOW_FPS, DISTANCE_CORRECTION_FACTOR,
                       DEFAULT_MODEL_PATH, GUI_SUPPORT_REQUIRED)

class ObjectDetectionWithDistanceAngle:
    """
    Main class for object detection with distance and angle calculation.
    
    This class integrates YOLO object detection with advanced 3D mapping
    capabilities for real-world distance and angle estimation.
    
    Attributes:
        model: YOLO model instance
        mapper: Advanced3DMapper instance for 3D calculations
        visualizer: VisualizationUtils instance for drawing
        device: Computing device (cuda/cpu)
    """
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, 
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 verbose: bool = True):
        """
        Initialize the object detection system with distance and angle calculation.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
            verbose: Whether to print initialization information
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ImportError: If required packages are not installed
        """
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Validate confidence threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {confidence_threshold}")
        
        try:
            # Validate system requirements
            if verbose:
                self._validate_system_requirements()
            
            # Set computing device with MPS support
            self.device = self._get_optimal_device()
            
            # Initialize YOLO model with device support
            # Ultralytics YOLO automatically uses GPU if available when device is specified
            self.model = YOLO(model_path)
            # Move model to detected device (if PyTorch model)
            if self.device != 'cpu' and hasattr(self.model, 'model'):
                try:
                    # For PyTorch models, move to device
                    if hasattr(self.model.model, 'to'):
                        self.model.model.to(self.device)
                    # Also set device for inference
                    self.model.device = self.device
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Warning: Could not move model to {self.device}: {e}")
                        print(f"   Continuing with CPU...")
                    self.device = 'cpu'
            
            self.confidence_threshold = confidence_threshold
            
            # Initialize 3D mapping components
            self.mapper = Advanced3DMapper()
            self.visualizer = VisualizationUtils()
            
            # Object tracking state
            self.seen_track_ids: Set[int] = set()
            self.current_tracks: Dict[int, Dict[str, Any]] = {}
            self.tracking_enabled = True
            
            # AI analysis queue for background processing
            self.ai_queue = queue.Queue(maxsize=10)
            self.ai_thread = None
            self.ai_processing = False
            
            if verbose:
                self._print_initialization_info()
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize detection system: {str(e)}")
    
    def _validate_system_requirements(self):
        """Validate system requirements and dependencies."""
        print("Validating system requirements...")
        
        # Check OpenCV GUI support
        if GUI_SUPPORT_REQUIRED:
            try:
                # Test if OpenCV has GUI support
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow("test", test_img)
                cv2.destroyAllWindows()
                print("✅ OpenCV GUI support: Available")
            except cv2.error as e:
                if "not implemented" in str(e).lower():
                    print("❌ OpenCV GUI support: Missing")
                    print("   Solution: pip uninstall opencv-python-headless -y && pip install opencv-python")
                    raise RuntimeError("OpenCV GUI support is required but not available. Please install opencv-python instead of opencv-python-headless.")
                else:
                    print("⚠️  OpenCV GUI support: Warning - " + str(e))
        
        # Check Python version
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"✅ Python version: {python_version}")
        
        # Check device availability
        device_info = self._get_device_info()
        if device_info['type'] == 'cuda':
            print(f"✅ CUDA support: Available (GPU: {device_info['name']})")
        elif device_info['type'] == 'mps':
            print(f"✅ MPS support: Available (Apple Silicon GPU)")
        else:
            print("ℹ️  GPU support: Not available (using CPU)")
        
        print("✅ System validation completed")
    
    def _print_initialization_info(self):
        """Print initialization information."""
        print("=" * 60)
        print("Object Detection with Distance and Angle Calculation")
        print("=" * 60)
        print(f"Device: {self.device.upper()}")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"Advanced 3D Mapping System initialized")
        print(f"Camera calibration - FX: {CAMERA_FX}, FY: {CAMERA_FY}")
        print(f"Principal point - CX: {self.mapper.cx}, CY: {self.mapper.cy}")
        print(f"Distance correction factor: {DISTANCE_CORRECTION_FACTOR}")
        print(f"Reprojection error: {self.mapper.mean_reprojection_error:.3f} pixels")
        print("=" * 60)
    
    def process_detections(self, results: Any, frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        """
        Process YOLO detection results and calculate distance/angles.
        
        Args:
            results: YOLO detection results
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            List of detection dictionaries with 3D mapping information
            
        Raises:
            ValueError: If frame dimensions are invalid
        """
        # Validate frame dimensions
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"Invalid frame dimensions: {frame_width}x{frame_height}")
        
        detections = []
        
        try:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection data
                        bbox = box.xyxy[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Filter by confidence threshold
                        if confidence > self.confidence_threshold:
                            # Process with advanced 3D mapping
                            mapping_result = self.mapper.process_yolo_detection(
                                bbox, class_name, confidence, (frame_height, frame_width)
                            )
                            
                            # Store detection data
                            detection = {
                                'bbox': bbox,
                                'class': class_name,
                                'confidence': confidence,
                                'distance': mapping_result['distance'],
                                'angle_x': mapping_result['horizontal_angle'],
                                'angle_y': mapping_result['vertical_angle'],
                                'center': mapping_result['center_pixel'],
                                'mapping_result': mapping_result  # Store full mapping result
                            }
                            detections.append(detection)
                            
        except Exception as e:
            print(f"Warning: Error processing detections: {str(e)}")
            
        return detections
    
    def process_detections_with_tracking(self, results: Any, frame_width: int, frame_height: int, 
                                       enable_tracking: bool = True) -> List[Dict[str, Any]]:
        """
        Process YOLO detection results with tracking information.
        
        Args:
            results: YOLO detection results
            frame_width: Width of the frame
            frame_height: Height of the frame
            enable_tracking: Whether tracking is enabled
            
        Returns:
            List of detection dictionaries with 3D mapping and tracking information
        """
        # Validate frame dimensions
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"Invalid frame dimensions: {frame_width}x{frame_height}")
        
        detections = []
        
        try:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get detection data
                        bbox = box.xyxy[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Get track ID if available
                        track_id = None
                        if enable_tracking and self.tracking_enabled and hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id[0].cpu().numpy())
                        
                        # Filter by confidence threshold
                        if confidence > self.confidence_threshold:
                            # Process with advanced 3D mapping
                            mapping_result = self.mapper.process_yolo_detection(
                                bbox, class_name, confidence, (frame_height, frame_width)
                            )
                            
                            # Store detection data
                            detection = {
                                'bbox': bbox,
                                'class': class_name,
                                'class_name': class_name,  # For compatibility
                                'class_id': class_id,
                                'confidence': confidence,
                                'distance': mapping_result['distance'],
                                'angle_x': mapping_result['horizontal_angle'],
                                'angle_y': mapping_result['vertical_angle'],
                                'center': mapping_result['center_pixel'],
                                'track_id': track_id,
                                'mapping_result': mapping_result  # Store full mapping result
                            }
                            detections.append(detection)
                            
        except Exception as e:
            print(f"Warning: Error processing detections with tracking: {str(e)}")
            
        return detections
    
    def draw_detection_info(self, frame, detection):
        """
        Draw advanced 3D mapping information for a single detection
        
        Args:
            frame: Input frame
            detection: Detection dictionary with mapping_result
            
        Returns:
            Annotated frame
        """
        # Use the advanced visualizer
        mapping_result = detection['mapping_result']
        frame = self.visualizer.draw_detection_with_3d_info(frame, mapping_result)
        
        return frame
    
    def process_video(self, video_source: Union[int, str] = DEFAULT_VIDEO_SOURCE,
                     show_fps: bool = SHOW_FPS,
                     save_frames: bool = True,
                     save_detections: bool = True) -> Dict[str, Any]:
        """
        Process video stream and display detections with distance and angle.
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
            show_fps: Whether to display FPS counter
            save_frames: Whether to allow frame saving with 's' key
            save_detections: Whether to allow detection data saving with 'd' key
            
        Returns:
            Dictionary with processing statistics
            
        Raises:
            RuntimeError: If video source cannot be opened
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")
        
        try:
            # Set video resolution from config
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Video resolution: {frame_width}x{frame_height}")
            print(f"FPS: {fps}")
            print("Controls:")
            print("  'q' - Quit")
            if save_frames:
                print("  's' - Save current frame")
            if save_detections:
                print("  'd' - Save detection data")
            print("  'r' - Reset detection counter")
            
            return self._run_video_loop(cap, frame_width, frame_height, show_fps, 
                                      save_frames, save_detections)
                                      
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _run_video_loop(self, cap: cv2.VideoCapture, frame_width: int, frame_height: int,
                       show_fps: bool, save_frames: bool, save_detections: bool) -> Dict[str, Any]:
        """Run the main video processing loop."""
        frame_count = 0
        detection_count = 0
        start_time = cv2.getTickCount()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run YOLO detection with device specification
                results = self.model(frame, device=self.device, verbose=False)
                
                # Process detections
                detections = self.process_detections(results, frame_width, frame_height)
                
                # Draw detections with advanced 3D mapping
                for detection in detections:
                    frame = self.draw_detection_info(frame, detection)
                    detection_count += 1
                
                # Calculate and display FPS
                if show_fps:
                    current_time = cv2.getTickCount()
                    elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
                    if elapsed_time > 0:
                        current_fps = frame_count / elapsed_time
                        cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Detections: {detection_count}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Object Detection with Distance and Angle', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and save_frames:
                    # Save current frame
                    filename = f"detection_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('d') and save_detections:
                    # Save detection data
                    if detections:
                        save_detection_data(detections, f"detection_data_{frame_count}.txt")
                        print(f"Detection data saved for frame {frame_count}")
                elif key == ord('r'):
                    # Reset detection counter
                    detection_count = 0
                    print("Detection counter reset")
                    
        except KeyboardInterrupt:
            print("\nVideo processing interrupted by user")
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
        
        # Return processing statistics
        stats = {
            'total_frames': frame_count,
            'total_detections': detection_count,
            'average_detections_per_frame': detection_count / max(frame_count, 1)
        }
        
        print(f"Total frames processed: {frame_count}")
        print(f"Total detections: {detection_count}")
        
        return stats
    
    def process_single_frame(self, frame: np.ndarray, enable_tracking: bool = True) -> Dict[str, Any]:
        """
        Process a single frame and return detection results with optional tracking.
        
        Args:
            frame: Input frame as numpy array
            enable_tracking: Whether to enable object tracking
            
        Returns:
            Dictionary containing detection results and statistics
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided")
        
        frame_height, frame_width = frame.shape[:2]
        
        # Run YOLO detection with or without tracking
        if enable_tracking and self.tracking_enabled:
            # Use ByteTrack tracker (built into YOLO)
            results = self.model.track(
                frame, 
                device=self.device,
                conf=self.confidence_threshold, 
                persist=True,
                verbose=False
            )
        else:
            results = self.model(frame, device=self.device, conf=self.confidence_threshold, verbose=False)
        
        # Process detections with tracking information
        detections = self.process_detections_with_tracking(results, frame_width, frame_height, enable_tracking)
        
        # Create output frame with visualizations
        output_frame = frame.copy()
        for detection in detections:
            output_frame = self.draw_detection_info(output_frame, detection)
        
        # Extract tracking information
        current_track_ids = set()
        new_track_ids = set()
        
        if enable_tracking and self.tracking_enabled:
            for detection in detections:
                if 'track_id' in detection and detection['track_id'] is not None:
                    track_id = detection['track_id']
                    current_track_ids.add(track_id)
                    
                    if track_id not in self.seen_track_ids:
                        new_track_ids.add(track_id)
                        self.seen_track_ids.add(track_id)
        
        return {
            'frame': output_frame,
            'detections': detections,
            'frame_shape': (frame_height, frame_width),
            'detection_count': len(detections),
            'new_track_ids': new_track_ids,
            'current_track_ids': current_track_ids,
            'tracking_enabled': enable_tracking and self.tracking_enabled
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and configuration.
        
        Returns:
            Dictionary with system information
        """
        return {
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'camera_calibration': {
                'fx': CAMERA_FX,
                'fy': CAMERA_FY,
                'cx': self.mapper.cx,
                'cy': self.mapper.cy,
                'reprojection_error': self.mapper.mean_reprojection_error
            },
            'video_settings': {
                'width': VIDEO_WIDTH,
                'height': VIDEO_HEIGHT,
                'default_source': DEFAULT_VIDEO_SOURCE
            }
        }
    
    def _get_optimal_device(self) -> str:
        """
        Get the optimal computing device (CUDA > MPS > CPU).
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        # Check CUDA availability first (NVIDIA GPUs)
        if torch.cuda.is_available():
            return 'cuda'
        
        # Check MPS availability (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        
        # Fallback to CPU
        return 'cpu'
    
    def _get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed device information.
        
        Returns:
            Dictionary with device type and name
        """
        if torch.cuda.is_available():
            return {
                'type': 'cuda',
                'name': torch.cuda.get_device_name(0),
                'memory': torch.cuda.get_device_properties(0).total_memory
            }
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return {
                'type': 'mps',
                'name': 'Apple Silicon GPU',
                'memory': 'Unknown'
            }
        else:
            return {
                'type': 'cpu',
                'name': 'CPU',
                'memory': 'N/A'
            }

def main():
    """
    Main function to run the object detection system.
    
    This function can be used as a standalone script or imported as a module.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Object Detection with Distance and Angle Calculation")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to YOLO model file")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD, 
                       help="Confidence threshold for detections")
    parser.add_argument("--source", default=DEFAULT_VIDEO_SOURCE, 
                       help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--no-fps", action="store_true", help="Disable FPS display")
    parser.add_argument("--no-save", action="store_true", help="Disable frame/detection saving")
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = ObjectDetectionWithDistanceAngle(
            model_path=args.model,
            confidence_threshold=args.confidence
        )
        
        # Start video processing
        stats = detector.process_video(
            video_source=args.source,
            show_fps=not args.no_fps,
            save_frames=not args.no_save,
            save_detections=not args.no_save
        )
        
        print("\nProcessing completed successfully!")
        print(f"Statistics: {stats}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
