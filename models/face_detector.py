"""
Face detection module using YOLOv8 for the Emotion Detection App.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

from utils.config import FACE_DETECTION_MODEL_PATH, FACE_PADDING

class FaceDetector:
    """YOLOv8 face detector wrapper class."""
    
    def __init__(
        self,
        model_path: str = FACE_DETECTION_MODEL_PATH,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the face detector.
        
        Args:
            model_path: Path to YOLOv8 model
            confidence_threshold: Minimum detection confidence
        """
        self.confidence_threshold = confidence_threshold
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face detection model not found at: {model_path}. "
                "Please download the YOLOv8 face detection model first."
            )
        
        # Load the model
        try:
            self.model = YOLO(model_path)
            self._model_loaded = True
            print(f"Face detector loaded from: {model_path}")
        except Exception as e:
            print(f"Failed to load face detection model: {str(e)}")
            self._model_loaded = False
    
    @property
    def is_ready(self) -> bool:
        """Check if the model is ready to use."""
        return self._model_loaded
    
    def detect_faces(
        self,
        frame: np.ndarray,
        expand_bbox: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Input image frame
            expand_bbox: Whether to expand the bounding box
            
        Returns:
            List of dictionaries with face bounding boxes and confidence scores
        """
        if not self._model_loaded:
            print("Model not loaded. Cannot detect faces.")
            return []
        
        # Run YOLOv8 inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Process results
        faces = []
        h, w = frame.shape[:2]
        
        # For each detection result
        for result in results:
            # Extract the detected bounding boxes
            for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = box
                
                # Get confidence score
                conf = float(result.boxes.conf[i].item())
                
                # If needed, expand bounding box to include more context
                if expand_bbox:
                    # Expand by FACE_PADDING pixels in each direction
                    x1 = max(0, x1 - FACE_PADDING)
                    y1 = max(0, y1 - FACE_PADDING)
                    x2 = min(w, x2 + FACE_PADDING)
                    y2 = min(h, y2 + FACE_PADDING)
                
                # Add face detection info
                faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                })
        
        return faces
    
    def extract_face_regions(
        self,
        frame: np.ndarray,
        faces: List[Dict[str, Any]],
        target_size: Tuple[int, int] = (48, 48)
    ) -> List[Dict[str, Any]]:
        """
        Extract face regions from detected faces.
        
        Args:
            frame: Input image frame
            faces: List of detected faces
            target_size: Target size for extracted faces
            
        Returns:
            Updated faces list with extracted face regions
        """
        updated_faces = []
        
        for face in faces:
            x1, y1, x2, y2 = map(int, face["bbox"])
            
            # Extract face region
            face_img = frame[y1:y2, x1:x2]
            
            # Handle empty face region (e.g., at the edge of the frame)
            if face_img.size == 0:
                continue
                
            # Resize to target size
            try:
                face_img_resized = cv2.resize(face_img, target_size)
                
                # Convert to grayscale if needed for emotion classifier
                face_img_gray = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2GRAY)
                
                # Create a copy of the face dict and add the face image
                face_with_img = face.copy()
                face_with_img["face_img"] = face_img_resized
                face_with_img["face_img_gray"] = face_img_gray
                
                updated_faces.append(face_with_img)
                
            except Exception as e:
                print(f"Error extracting face: {str(e)}")
                continue
                
        return updated_faces 