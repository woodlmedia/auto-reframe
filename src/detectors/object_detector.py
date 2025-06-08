"""
Object detection using YOLOv8 for fallback when no people detected
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Detect and track objects using YOLOv8"""
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        preferred_classes: Optional[List[str]] = None
    ):
        """
        Initialize object detector
        
        Args:
            model_name: YOLOv8 model to use
            confidence_threshold: Minimum detection confidence
            preferred_classes: List of preferred class names
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
        # Default preferred classes
        if preferred_classes is None:
            preferred_classes = [
                'person', 'dog', 'cat', 'bird', 'horse',
                'car', 'motorcycle', 'bicycle',
                'tv', 'laptop', 'cell phone'
            ]
        
        self.preferred_classes = preferred_classes
        
        # Get class indices
        self.class_names = self.model.names
        self.preferred_indices = [
            i for i, name in self.class_names.items()
            if name in self.preferred_classes
        ]
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of object detections
        """
        # Run detection
        results = self.model(frame, verbose=False)
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                confidence = float(boxes.conf[i])
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Get box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class
                class_id = int(boxes.cls[i])
                class_name = self.class_names[class_id]
                
                # Calculate center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                detections.append({
                    'type': 'object',
                    'class': class_name,
                    'class_id': class_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': confidence,
                    'is_preferred': class_id in self.preferred_indices
                })
        
        return detections
    
    def get_saliency_score(self, detection: Dict, frame_shape: tuple) -> float:
        """
        Calculate saliency score for an object
        
        Higher score = more important to track
        """
        h, w = frame_shape[:2]
        
        # Base score from confidence
        score = detection['confidence']
        
        # Bonus for preferred classes
        if detection['is_preferred']:
            score *= 1.5
        
        # Size factor (larger objects more important)
        bbox = detection['bbox']
        obj_width = bbox[2] - bbox[0]
        obj_height = bbox[3] - bbox[1]
        size_ratio = (obj_width * obj_height) / (w * h)
        score *= (1 + size_ratio * 2)
        
        # Center bias (objects near center more important)
        center_x, center_y = detection['center']
        dist_from_center = np.sqrt(
            ((center_x - w/2) / (w/2))**2 + 
            ((center_y - h/2) / (h/2))**2
        )
        score *= (2 - dist_from_center)
        
        # Special boost for certain classes
        priority_classes = {'person': 3.0, 'dog': 2.0, 'cat': 2.0}
        if detection['class'] in priority_classes:
            score *= priority_classes[detection['class']]
        
        return score
    
    def select_best_object(
        self,
        detections: List[Dict],
        frame_shape: tuple,
        previous_target: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Select the best object to track
        
        Args:
            detections: List of detected objects
            frame_shape: Shape of the frame
            previous_target: Previously tracked object for continuity
            
        Returns:
            Best object to track or None
        """
        if not detections:
            return None
        
        # Calculate saliency scores
        for det in detections:
            det['saliency'] = self.get_saliency_score(det, frame_shape)
        
        # If we have a previous target, check for continuity
        if previous_target and 'class' in previous_target:
            # Look for same class near previous position
            prev_center = previous_target['center']
            
            for det in detections:
                if det['class'] == previous_target['class']:
                    # Calculate distance
                    dist = np.sqrt(
                        (det['center'][0] - prev_center[0])**2 +
                        (det['center'][1] - prev_center[1])**2
                    )
                    
                    # Boost score for continuity
                    det['saliency'] *= (1 + 100 / (dist + 100))
        
        # Select highest scoring object
        best = max(detections, key=lambda x: x['saliency'])
        
        return best
