"""
Face Recognition Adapter - Integrates FaceNet face recognition into the agent system.

This adapter wraps the facenet_multi.py functionality and provides:
- Face recognition on person detections
- Name resolution for recognized faces
- Integration with temporal memory
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import sys
import numpy as np
import cv2
import pickle
import platform

# Fix subprocess errors on Windows
if platform.system() == "Windows":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from core.infrastructure.device_manager import get_device_manager, get_torch_device


class FaceRecognitionAdapter:
    """
    Adapter for FaceNet face recognition that integrates with the agent system.
    Recognizes faces in person detections and returns names from the gallery.
    """
    
    def __init__(self, gallery_path: Optional[str] = None, threshold: float = 0.82, 
                 force_cpu: bool = False, verbose: bool = False):
        """
        Initialize face recognition adapter.
        
        Args:
            gallery_path: Path to gallery.pkl file (auto-detected if None)
            threshold: Similarity threshold for recognition (default: 0.82)
            force_cpu: If True, force CPU usage even if GPU is available
            verbose: If True, print debug messages
        """
        self.verbose = verbose
        self.threshold = threshold
        self.force_cpu = force_cpu
        
        # Resolve gallery path
        if gallery_path is None:
            # Try to find gallery.pkl in Facenet directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            facenet_dir = os.path.join(os.path.dirname(current_dir), "Facenet")
            gallery_path = os.path.join(facenet_dir, "gallery.pkl")
            
            # If not found, try current directory
            if not os.path.exists(gallery_path):
                gallery_path = "gallery.pkl"
        
        self.gallery_path = gallery_path
        
        # Get device from device manager
        device_mgr = get_device_manager(force_cpu=force_cpu)
        self.device_str = get_torch_device()
        
        # Initialize models
        self.mtcnn: Optional[MTCNN] = None
        self.net: Optional[InceptionResnetV1] = None
        self.gallery: Dict[str, Dict[str, Any]] = {}
        self.gallery_embeddings: Optional[np.ndarray] = None
        self.gallery_names: List[str] = []
        
        # Load gallery
        self._load_gallery()
        
        # Initialize models (lazy loading)
        self._initialized = False
    
    def _load_gallery(self) -> None:
        """Load face gallery from pickle file."""
        if os.path.exists(self.gallery_path):
            try:
                with open(self.gallery_path, "rb") as f:
                    self.gallery = pickle.load(f)
                
                if self.gallery:
                    # Build embedding matrix for fast lookup
                    self.gallery_names = list(self.gallery.keys())
                    embeddings = [self.gallery[n]["emb"] for n in self.gallery_names]
                    self.gallery_embeddings = np.stack(embeddings, axis=0)  # [K, 512]
                    if self.verbose:
                        print(f"[FaceRecognition] Loaded gallery with {len(self.gallery_names)} people: {', '.join(self.gallery_names)}")
                else:
                    self.gallery_embeddings = None
                    self.gallery_names = []
                    if self.verbose:
                        print(f"[FaceRecognition] Gallery is empty")
            except Exception as e:
                print(f"[WARNING] Failed to load gallery from {self.gallery_path}: {e}")
                self.gallery = {}
                self.gallery_embeddings = None
                self.gallery_names = []
        else:
            if self.verbose:
                print(f"[FaceRecognition] Gallery file not found at {self.gallery_path}")
            self.gallery = {}
            self.gallery_embeddings = None
            self.gallery_names = []
    
    def _initialize_models(self) -> None:
        """Initialize MTCNN and InceptionResnetV1 models (lazy loading)."""
        if self._initialized:
            return
        
        try:
            if self.verbose:
                print(f"[FaceRecognition] Initializing models on device: {self.device_str}")
            
            # Initialize MTCNN
            try:
                self.mtcnn = MTCNN(
                    image_size=160,
                    margin=20,
                    post_process=True,
                    device=self.device_str,
                    keep_all=False,
                    min_face_size=40,
                    thresholds=[0.7, 0.8, 0.8],
                    factor=0.709
                )
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] MTCNN initialization issue: {e}")
                # Fallback with minimal settings
                self.mtcnn = MTCNN(image_size=160, margin=20, device=self.device_str, post_process=True)
            
            # Initialize InceptionResnetV1
            self.net = InceptionResnetV1(pretrained="vggface2").eval()
            self.net = self.net.to(self.device_str)
            
            self._initialized = True
            
            if self.verbose:
                print(f"[FaceRecognition] Models initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize face recognition models: {e}")
            self.mtcnn = None
            self.net = None
            self._initialized = False
    
    def _bgr2rgb(self, img: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB."""
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _resize_for_detection(self, img: np.ndarray, max_size: int = 800) -> np.ndarray:
        """Resize image for faster face detection while maintaining aspect ratio."""
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
    
    @torch.no_grad()
    def _embed_faces_bgr(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract face embeddings from BGR frame.
        
        Returns:
            Tuple of (embeddings, boxes) or (None, None) if no faces detected
        """
        if not self._initialized:
            self._initialize_models()
        
        if self.mtcnn is None or self.net is None:
            return None, None
        
        try:
            rgb = self._bgr2rgb(frame_bgr)
            
            # Detect faces
            boxes, probs = self.mtcnn.detect(rgb)
            if boxes is None:
                return None, None
            
            # Get aligned face images
            faces = self.mtcnn(rgb)
            if faces is None:
                return None, None
            
            # Handle single face vs multiple faces
            if faces.dim() == 3:
                faces = faces.unsqueeze(0)
            
            # Extract embeddings
            emb = self.net(faces.to(self.device_str)).cpu()
            emb = torch.nn.functional.normalize(emb, dim=1)  # L2-normalized
            return emb.numpy(), boxes
        except (OSError, RuntimeError, AttributeError) as e:
            if "subprocess" in str(e).lower() or "multiprocessing" in str(e).lower():
                if self.verbose:
                    print(f"[WARNING] Face detection subprocess issue: {e}")
            return None, None
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Face detection error: {e}")
            return None, None
    
    def recognize_faces_in_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recognize all faces in a frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            List of recognized faces, each with: {
                "name": str,  # Name from gallery or "Unknown"
                "confidence": float,  # Similarity score
                "bbox": [x1, y1, x2, y2],  # Bounding box
                "embedding": np.ndarray  # Face embedding (optional)
            }
        """
        if self.gallery_embeddings is None or len(self.gallery_names) == 0:
            # No gallery loaded, return empty
            return []
        
        # Resize for faster processing
        small_frame = self._resize_for_detection(frame, max_size=640)
        scale = frame.shape[1] / small_frame.shape[1] if small_frame.shape[1] > 0 else 1.0
        
        # Extract face embeddings
        embs, boxes = self._embed_faces_bgr(small_frame)
        
        if embs is None or boxes is None:
            return []
        
        # Scale boxes back to original frame size
        boxes = boxes * scale
        
        # Recognize each face
        recognized = []
        for emb, box in zip(embs, boxes):
            # Normalize embedding
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            
            # Compute cosine similarity with gallery
            sims = self.gallery_embeddings @ emb  # [K] cosine similarities
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            best_name = self.gallery_names[best_idx]
            
            # Check threshold
            if best_score >= self.threshold:
                name = best_name
            else:
                name = "Unknown"
            
            recognized.append({
                "name": name,
                "confidence": best_score,
                "bbox": box.tolist() if hasattr(box, 'tolist') else box,
                "embedding": emb
            })
        
        return recognized
    
    def recognize_face_in_roi(self, frame: np.ndarray, roi: List[float]) -> Optional[Dict[str, Any]]:
        """
        Recognize a single face in a region of interest (ROI).
        
        Args:
            frame: Input frame (BGR format)
            roi: Region of interest [x1, y1, x2, y2] (normalized or pixel coordinates)
        
        Returns:
            Dict with "name", "confidence", "bbox" or None if no face detected
        """
        if self.gallery_embeddings is None or len(self.gallery_names) == 0:
            return None
        
        # Extract ROI from frame
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        
        # Clamp to frame bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(x1, min(x2, w))
        y2 = max(y1, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract ROI
        roi_frame = frame[y1:y2, x1:x2]
        if roi_frame.size == 0:
            return None
        
        # Resize ROI for face detection
        roi_frame = self._resize_for_detection(roi_frame, max_size=400)
        
        # Extract face embedding
        embs, boxes = self._embed_faces_bgr(roi_frame)
        
        if embs is None or boxes is None or len(embs) == 0:
            return None
        
        # Use the largest face if multiple detected
        if len(boxes) > 1:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            best_idx = int(np.argmax(areas))
        else:
            best_idx = 0
        
        emb = embs[best_idx]
        box = boxes[best_idx]
        
        # Adjust box coordinates back to original frame
        roi_scale_x = (x2 - x1) / roi_frame.shape[1] if roi_frame.shape[1] > 0 else 1.0
        roi_scale_y = (y2 - y1) / roi_frame.shape[0] if roi_frame.shape[0] > 0 else 1.0
        box[0] = x1 + box[0] * roi_scale_x
        box[1] = y1 + box[1] * roi_scale_y
        box[2] = x1 + box[2] * roi_scale_x
        box[3] = y1 + box[3] * roi_scale_y
        
        # Normalize embedding
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        
        # Compute cosine similarity with gallery
        sims = self.gallery_embeddings @ emb
        best_idx_gallery = int(np.argmax(sims))
        best_score = float(sims[best_idx_gallery])
        best_name = self.gallery_names[best_idx_gallery]
        
        # Check threshold
        if best_score >= self.threshold:
            name = best_name
        else:
            name = "Unknown"
        
        return {
            "name": name,
            "confidence": best_score,
            "bbox": box.tolist() if hasattr(box, 'tolist') else box
        }
    
    def recognize_person_objects(self, frame: np.ndarray, person_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Recognize faces in person object detections.
        
        Args:
            frame: Input frame (BGR format)
            person_objects: List of person detection objects with bbox
        
        Returns:
            List of updated person objects with "recognized_name" field if recognized
        """
        if self.gallery_embeddings is None or len(self.gallery_names) == 0:
            # No gallery, return objects as-is
            return person_objects
        
        updated_objects = []
        for obj in person_objects:
            bbox = obj.get("bbox", [])
            if not bbox or len(bbox) != 4:
                # No valid bbox, skip recognition
                updated_objects.append(obj)
                continue
            
            # Try to recognize face in ROI
            recognition = self.recognize_face_in_roi(frame, bbox)
            
            if recognition and recognition["name"] != "Unknown":
                # Update object with recognized name
                obj = obj.copy()
                obj["recognized_name"] = recognition["name"]
                obj["face_confidence"] = recognition["confidence"]
                # Update class name to use recognized name instead of "person"
                obj["class"] = recognition["name"]
                if self.verbose:
                    print(f"[FaceRecognition] Recognized {recognition['name']} with confidence {recognition['confidence']:.2f}")
            else:
                # Keep original object
                obj = obj.copy()
            
            updated_objects.append(obj)
        
        return updated_objects
    
    def get_gallery_names(self) -> List[str]:
        """Get list of enrolled person names."""
        return self.gallery_names.copy()
    
    def reload_gallery(self) -> None:
        """Reload gallery from disk (useful if gallery was updated externally)."""
        self._load_gallery()
        if self.verbose:
            print(f"[FaceRecognition] Gallery reloaded: {len(self.gallery_names)} people")

