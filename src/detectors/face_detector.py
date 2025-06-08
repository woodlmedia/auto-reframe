import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Dict, Tuple


class FaceDetector:
    """Detect faces and facial landmarks using MediaPipe with enhanced accuracy"""
    
    def __init__(self, min_detection_confidence: float = 0.7):
        """
        Initialize face detector with higher accuracy settings
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Use both face detection and face mesh for better accuracy
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Use full-range model
            min_detection_confidence=min_detection_confidence
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame with confidence scores
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of face detections with bounding boxes, landmarks, and confidence
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        detections = []
        
        # First, use face detection for confidence scores
        detection_results = self.face_detection.process(rgb_frame)
        
        # Then use face mesh for precise landmarks
        mesh_results = self.face_mesh.process(rgb_frame)
        
        if detection_results.detections and mesh_results.multi_face_landmarks:
            # Match detections with mesh results
            for i, (detection, face_landmarks) in enumerate(
                zip(detection_results.detections, mesh_results.multi_face_landmarks)
            ):
                # Get confidence from face detection
                confidence = detection.score[0] if detection.score else 0.5
                
                # Get bounding box from detection
                bbox = detection.location_data.relative_bounding_box
                x_min = max(0, int(bbox.xmin * w))
                y_min = max(0, int(bbox.ymin * h))
                x_max = min(w, int((bbox.xmin + bbox.width) * w))
                y_max = min(h, int((bbox.ymin + bbox.height) * h))
                
                # Get precise landmarks from mesh
                landmarks = []
                xs = []
                ys = []
                
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
                    xs.append(x)
                    ys.append(y)
                
                # Get key facial points for better tracking
                # Using specific landmark indices from MediaPipe
                left_eye_inner = landmarks[133]  # Left eye inner corner
                right_eye_inner = landmarks[362]  # Right eye inner corner
                nose_tip = landmarks[1]  # Nose tip
                
                # Calculate stable center point (between eyes)
                center_x = (left_eye_inner[0] + right_eye_inner[0]) // 2
                center_y = (left_eye_inner[1] + right_eye_inner[1]) // 2
                
                # Calculate face quality score based on visibility
                # Check if key landmarks are visible
                key_landmarks = [1, 133, 362, 61, 291]  # Nose, eyes, mouth corners
                visibility_scores = []
                for idx in key_landmarks:
                    if idx < len(face_landmarks.landmark):
                        visibility_scores.append(face_landmarks.landmark[idx].visibility)
                
                quality_score = np.mean(visibility_scores) if visibility_scores else 0.5
                
                # Combined confidence
                final_confidence = confidence * quality_score
                
                detections.append({
                    'type': 'face',
                    'bbox': (x_min, y_min, x_max, y_max),
                    'center': (center_x, center_y),
                    'nose_tip': nose_tip,
                    'landmarks': landmarks,
                    'confidence': final_confidence,
                    'raw_confidence': confidence,
                    'quality_score': quality_score
                })
        
        elif mesh_results.multi_face_landmarks:
            # Fallback to mesh-only detection
            for face_landmarks in mesh_results.multi_face_landmarks:
                landmarks = []
                xs = []
                ys = []
                
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
                    xs.append(x)
                    ys.append(y)
                
                if xs and ys:
                    x_min = max(0, min(xs))
                    y_min = max(0, min(ys))
                    x_max = min(w, max(xs))
                    y_max = min(h, max(ys))
                    
                    # Key points
                    left_eye = landmarks[133] if len(landmarks) > 133 else landmarks[0]
                    right_eye = landmarks[362] if len(landmarks) > 362 else landmarks[0]
                    center_x = (left_eye[0] + right_eye[0]) // 2
                    center_y = (left_eye[1] + right_eye[1]) // 2
                    
                    detections.append({
                        'type': 'face',
                        'bbox': (x_min, y_min, x_max, y_max),
                        'center': (center_x, center_y),
                        'nose_tip': landmarks[1] if len(landmarks) > 1 else (center_x, center_y),
                        'landmarks': landmarks,
                        'confidence': 0.7,  # Default confidence for mesh-only
                        'raw_confidence': 0.7,
                        'quality_score': 1.0
                    })
        
        return detections
    
    def get_face_size(self, detection: Dict) -> float:
        """Calculate relative face size for depth estimation"""
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return np.sqrt(width * height)
    
    def get_stability_score(self, detection: Dict) -> float:
        """Calculate stability score for tracking priority"""
        # Higher confidence and quality = more stable
        return detection['confidence'] * detection.get('quality_score', 1.0)
    
    def close(self):
        """Release resources"""
        self.face_detection.close()
        self.face_mesh.close()
