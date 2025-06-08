"""
Face detection using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Dict, Tuple


class FaceDetector:
    """Detect faces and facial landmarks using MediaPipe"""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize face detector
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of face detections with bounding boxes and landmarks
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        detections = []
        
        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            
            for face_landmarks in results.multi_face_landmarks:
                # Get key points
                landmarks = []
                xs = []
                ys = []
                
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
                    xs.append(x)
                    ys.append(y)
                
                # Calculate bounding box
                x_min = max(0, min(xs))
                y_min = max(0, min(ys))
                x_max = min(w, max(xs))
                y_max = min(h, max(ys))
                
                # Get center point (between eyes)
                left_eye = landmarks[33]  # Left eye inner corner
                right_eye = landmarks[133]  # Right eye inner corner
                center_x = (left_eye[0] + right_eye[0]) // 2
                center_y = (left_eye[1] + right_eye[1]) // 2
                
                # Get nose tip for depth estimation
                nose_tip = landmarks[1]
                
                detections.append({
                    'type': 'face',
                    'bbox': (x_min, y_min, x_max, y_max),
                    'center': (center_x, center_y),
                    'nose_tip': nose_tip,
                    'landmarks': landmarks,
                    'confidence': 1.0  # MediaPipe doesn't provide confidence
                })
        
        return detections
    
    def get_face_size(self, detection: Dict) -> float:
        """Calculate relative face size for depth estimation"""
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return np.sqrt(width * height)
    
    def close(self):
        """Release resources"""
        self.face_mesh.close()
