import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
from tqdm import tqdm

from ..detectors import FaceDetector, PoseDetector, ObjectDetector
from ..trackers import KalmanTracker, SmoothTracker
from .crop_calculator import CropCalculator
from ..utils.video_utils import get_video_info, create_video_writer
from ..utils.visualization import draw_tracking_overlay

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process videos with intelligent reframing"""
    
    def __init__(self, config: Dict):
        """
        Initialize video processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize detectors
        self.face_detector = FaceDetector(
            config['detection']['face_confidence']
        )
        self.pose_detector = PoseDetector(
            config['detection']['pose_confidence']
        )
        self.object_detector = ObjectDetector(
            confidence_threshold=config['detection']['object_confidence'],
            preferred_classes=config['objects']['preferred_classes']
        )
        
        # Initialize tracker
        if config['tracking'].get('use_kalman', True):
            self.tracker = KalmanTracker(
                process_noise=config['tracking']['kalman_process_noise'],
                measurement_noise=config['tracking']['kalman_measurement_noise']
            )
        else:
            self.tracker = SmoothTracker(
                smoothing_factor=config['tracking']['smoothing_factor']
            )
        
        # Initialize crop calculator
        self.crop_calculator = CropCalculator(
            output_aspect_ratio=config['reframing']['default_aspect_ratio'],
            padding_ratio=config['reframing']['padding_ratio'],
            min_zoom=config['reframing']['min_zoom'],
            max_zoom=config['reframing']['max_zoom'],
            lead_room=config['reframing']['lead_room']
        )
        
        # State tracking
        self.current_target = None
        self.current_crop = None
        self.lost_frames = 0
        self.max_lost_frames = config['tracking']['max_lost_frames']
        
    def process(
        self,
        input_path: str,
        output_path: str,
        preview: bool = False,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """
        Process video file
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            preview: Show preview window
            start_time: Start time in seconds
            end_time: End time in seconds
        """
        # Open video
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame range
        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else total_frames
        
        # Calculate output dimensions
        aspect_ratio = self.config['reframing']['default_aspect_ratio']
        output_height = 1080  # Default HD height
        output_width = int(output_height * aspect_ratio[0] / aspect_ratio[1])
        
        # Create video writer
        writer = create_video_writer(
            output_path,
            fps,
            (output_width, output_height),
            codec=self.config['video']['output_codec']
        )
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames
        frame_count = end_frame - start_frame
        with tqdm(total=frame_count, desc="Processing") as pbar:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame > end_frame:
                    break
                
                # Process frame
                output_frame = self.process_frame(
                    frame,
                    (output_width, output_height)
                )
                
                # Write frame
                writer.write(output_frame)
                
                # Show preview
                if preview:
                    # Create preview with overlay
                    preview_frame = self.create_preview(frame, output_frame)
                    cv2.imshow('Auto Reframe Preview', preview_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                pbar.update(1)
        
        # Cleanup
        cap.release()
        writer.release()
        if preview:
            cv2.destroyAllWindows()
        
        # Close detectors
        self.face_detector.close()
        self.pose_detector.close()
        
    def process_frame(
        self,
        frame: np.ndarray,
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Process single frame
        
        Args:
            frame: Input frame
            output_size: Output size (width, height)
            
        Returns:
            Processed frame
        """
        h, w = frame.shape[:2]
        
        # Detect targets
        target = self.detect_target(frame)
        
        if target:
            # Update tracker
            tracked_center = self.tracker.update(target['center'])
            
            # Calculate target size
            if target['type'] in ['face', 'pose']:
                bbox = target['bbox']
                target_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            else:
                # For objects, use bbox area
                bbox = target['bbox']
                target_size = np.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            
            # Update current target
            self.current_target = target
            self.lost_frames = 0
        else:
            # No target found
            self.lost_frames += 1
            
            if self.lost_frames < self.max_lost_frames and self.current_target:
                # Use predicted position
                tracked_center = self.tracker.predict_next()
                target_size = None
            else:
                # Lost tracking - reset to center
                tracked_center = (w // 2, h // 2)
                target_size = None
                self.current_target = None
                self.tracker.reset()
        
        # Calculate crop
        target_crop = self.crop_calculator.calculate_crop(
            tracked_center,
            target_size,
            (h, w),
            self.current_crop
        )
        
        # Smooth transition
        if self.current_crop:
            smoothed_crop = self.crop_calculator.smooth_transition(
                self.current_crop,
                target_crop,
                self.config['tracking']['smoothing_factor']
            )
        else:
            smoothed_crop = target_crop
        
        self.current_crop = smoothed_crop
        
        # Apply crop and resize
        output_frame = self.crop_calculator.apply_crop(
            frame,
            smoothed_crop,
            output_size
        )
        
        return output_frame
    
    def detect_target(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect best target in frame
        
        Args:
            frame: Input frame
            
        Returns:
            Best target detection or None
        """
        all_detections = []
        
        # Detect faces
        faces = self.face_detector.detect(frame)
        all_detections.extend(faces)
        
        # Detect poses if no faces
        if not faces:
            poses = self.pose_detector.detect(frame)
            all_detections.extend(poses)
        
        # Detect objects if no people
        if not all_detections:
            objects = self.object_detector.detect(frame)
            if objects:
                # Select best object
                best_object = self.object_detector.select_best_object(
                    objects,
                    frame.shape,
                    self.current_target
                )
                if best_object:
                    all_detections.append(best_object)
        
        # Select best detection
        if all_detections:
            # Prioritize by type: face > pose > object
            type_priority = {'face': 3, 'pose': 2, 'object': 1}
            
            # Sort by priority and confidence
            all_detections.sort(
                key=lambda x: (
                    type_priority.get(x['type'], 0),
                    x.get('confidence', 0)
                ),
                reverse=True
            )
            
            return all_detections[0]
        
        return None
    
    def create_preview(
        self,
        original: np.ndarray,
        output: np.ndarray
    ) -> np.ndarray:
        """Create side-by-side preview"""
        h, w = original.shape[:2]
        
        # Scale for preview
        scale = self.config['video']['preview_scale']
        preview_h = int(h * scale)
        preview_w = int(w * scale)
        
        # Resize frames
        original_small = cv2.resize(original, (preview_w, preview_h))
        output_small = cv2.resize(output, (preview_w, preview_h))
        
        # Draw overlay on original
        if self.current_crop:
            draw_tracking_overlay(
                original_small,
                self.current_crop,
                self.current_target,
                scale
            )
        
        # Combine side by side
        preview = np.hstack([original_small, output_small])
        
        # Add labels
        cv2.putText(
            preview, "Original", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            preview, "Reframed", (preview_w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        
        return preview
