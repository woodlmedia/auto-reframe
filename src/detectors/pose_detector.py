"""
Pose detection using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Dict


class PoseDetector:
    """Detect human poses using MediaPipe"""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize pose detector
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect poses in frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of pose detections
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        detections = []
        
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            
            # Get landmarks
            landmarks = []
            xs = []
            ys = []
            
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y, landmark.visibility))
                
                # Only include visible landmarks for bbox
                if landmark.visibility > 0.5:
                    xs.append(x)
                    ys.append(y)
            
            if xs and ys:
                # Calculate bounding box
                x_min = max(0, min(xs))
                y_min = max(0, min(ys))
                x_max = min(w, max(xs))
                y_max = min(h, max(ys))
                
                # Get body center (midpoint between shoulders and hips)
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                
                if all(p[2] > 0.5 for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
                    center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) // 4
                    center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) // 4
                else:
                    # Fallback to bbox center
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2
                
                detections.append({
                    'type': 'pose',
                    'bbox': (x_min, y_min, x_max, y_max),
                    'center': (center_x, center_y),
                    'landmarks': landmarks,
                    'confidence': np.mean([l[2] for l in landmarks])
                })
        
        return detections
    
    def get_pose_size(self, detection: Dict) -> float:
        """Calculate relative pose size"""
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return np.sqrt(width * height)
    
    def close(self):
        """Release resources"""
        self.pose.close()
