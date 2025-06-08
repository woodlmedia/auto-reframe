import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
from tqdm import tqdm

from ..detectors import FaceDetector, PoseDetector, ObjectDetector
from ..trackers import KeyframeTracker
from .crop_calculator import CropCalculator
from ..utils.video_utils import get_video_info, create_video_writer
from ..utils.visualization import draw_tracking_overlay
from ..utils.scene_detector import AdaptiveSceneDetector

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process videos with intelligent reframing using keyframe tracking"""
    
    def __init__(self, config: Dict):
        """
        Initialize video processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize detectors with higher confidence
        self.face_detector = FaceDetector(
            min_detection_confidence=0.7  # Higher for better accuracy
        )
        self.pose_detector = PoseDetector(
            config['detection']['pose_confidence']
        )
        self.object_detector = ObjectDetector(
            confidence_threshold=config['detection']['object_confidence'],
            preferred_classes=config['objects']['preferred_classes']
        )
        
        # Initialize keyframe tracker
        self.tracker = KeyframeTracker(
            confidence_threshold=0.25,  # Only track if confidence > 25%
            interpolation_method='cubic',
            min_keyframe_distance=5
        )
        
        # Initialize scene detector
        self.scene_detector = AdaptiveSceneDetector(
            initial_threshold=30.0,
            min_scene_length=10,
            adaptation_rate=0.1
        )
        
        # Initialize crop calculator (X-axis only)
        self.crop_calculator = CropCalculator(
            output_aspect_ratio=config['reframing']['default_aspect_ratio'],
            lead_room=config['reframing']['lead_room']
        )
        
        # State tracking
        self.current_target = None
        self.current_crop = None
        self.lost_frames = 0
        self.max_lost_frames = config['tracking']['max_lost_frames']
        self.frame_count = 0
        
    def process(
        self,
        input_path: str,
        output_path: str,
        preview: bool = False,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """
        Process video file with keyframe tracking
        
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
        
        # Calculate output dimensions based on aspect ratio
        aspect_ratio = self.config['reframing']['default_aspect_ratio']
        frame_aspect = width / height
        output_aspect = aspect_ratio[0] / aspect_ratio[1]
        
        # Determine output dimensions (crop size, not resize)
        if frame_aspect > output_aspect:
            # Frame is wider - crop horizontally
            output_height = height
            output_width = int(height * output_aspect)
        else:
            # Frame matches or is narrower - use full frame
            output_width = width
            output_height = height
        
        # Create video writer with crop dimensions
        writer = create_video_writer(
            output_path,
            fps,
            (output_width, output_height),
            codec=self.config['video']['output_codec']
        )
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # First pass: detect all targets and scene changes
        logger.info("First pass: Analyzing video...")
        self._analyze_video(cap, start_frame, end_frame)
        
        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.frame_count = start_frame
        
        # Second pass: process with interpolated positions
        logger.info("Second pass: Processing video...")
        frame_count = end_frame - start_frame
        
        with tqdm(total=frame_count, desc="Processing") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if current_frame >= end_frame:
                    break
                
                # Process frame with interpolated tracking - NO RESIZE
                output_frame = self.process_frame_interpolated(
                    frame,
                    current_frame,
                    None  # No output size - keep original dimensions
                )
                
                # Write frame
                writer.write(output_frame)
                
                # Show preview
                if preview:
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
        
        logger.info(f"Processed {len(self.tracker.keyframes)} keyframes")
        logger.info(f"Detected {len(self.scene_detector.scene_changes)} scene changes")
    
    def _analyze_video(self, cap, start_frame: int, end_frame: int):
        """First pass: analyze video for keyframes and scene changes"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = end_frame - start_frame
        with tqdm(total=frame_count, desc="Analyzing") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if current_frame >= end_frame:
                    break
                
                # Detect scene change
                is_scene_change = self.scene_detector.detect_scene_change(
                    frame, current_frame
                )
                
                # Detect target
                target = self.detect_target(frame)
                
                if target:
                    # Update tracker with confidence
                    confidence = target.get('confidence', 0.5)
                    
                    # Force keyframe on scene change
                    if is_scene_change:
                        self.tracker.add_scene_boundary(
                            current_frame,
                            target['center'][0]
                        )
                    else:
                        # Normal update with confidence check
                        self.tracker.update(
                            target['center'][0],
                            confidence,
                            frame_num=current_frame,
                            force_keyframe=False
                        )
                    
                    self.current_target = target
                    self.lost_frames = 0
                else:
                    self.lost_frames += 1
                    
                    # If we just lost tracking at a scene change, add boundary
                    if is_scene_change and self.current_target:
                        # Use last known position
                        self.tracker.add_scene_boundary(
                            current_frame,
                            self.current_target['center'][0]
                        )
                
                pbar.update(1)
    
    def process_frame_interpolated(
        self,
        frame: np.ndarray,
        frame_num: int,
        output_size: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Process single frame using interpolated positions
        
        Args:
            frame: Input frame
            frame_num: Frame number
            output_size: IGNORED - we never resize
            
        Returns:
            Processed frame (cropped only, no resize)
        """
        h, w = frame.shape[:2]
        
        # Get interpolated X position from tracker
        tracked_x = self.tracker._get_interpolated_position(frame_num)
        
        # Calculate crop (X-axis only)
        target_crop = self.crop_calculator.calculate_crop(
            tracked_x,
            (h, w),
            self.current_crop
        )
        
        # No additional smoothing needed - already interpolated
        self.current_crop = target_crop
        
        # Apply crop WITHOUT resize
        output_frame = self.crop_calculator.apply_crop(
            frame,
            target_crop,
            None  # No resize
        )
        
        return output_frame
    
    def process_frame(
        self,
        frame: np.ndarray,
        output_size: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Process single frame (legacy method for compatibility)
        
        Args:
            frame: Input frame
            output_size: IGNORED - we never resize
            
        Returns:
            Processed frame
        """
        h, w = frame.shape[:2]
        
        # Detect targets
        target = self.detect_target(frame)
        
        if target:
            # Update tracker with confidence
            confidence = target.get('confidence', 0.5)
            tracked_x = self.tracker.update(
                target['center'][0],
                confidence,
                frame_num=self.frame_count
            )
            
            self.current_target = target
            self.lost_frames = 0
        else:
            # No target found
            self.lost_frames += 1
            
            if self.lost_frames < self.max_lost_frames and self.current_target:
                # Use last interpolated position
                tracked_x = self.tracker._get_interpolated_position(self.frame_count)
            else:
                # Lost tracking - use center
                tracked_x = w // 2
                self.current_target = None
        
        # Calculate crop (X-axis only)
        target_crop = self.crop_calculator.calculate_crop(
            tracked_x,
            (h, w),
            self.current_crop
        )
        
        self.current_crop = target_crop
        self.frame_count += 1
        
        # Apply crop WITHOUT resize
        output_frame = self.crop_calculator.apply_crop(
            frame,
            target_crop,
            None  # No resize
        )
        
        return output_frame
    
    def detect_target(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect best target in frame with confidence
        
        Args:
            frame: Input frame
            
        Returns:
            Best target detection or None
        """
        all_detections = []
        
        # Detect faces (highest priority)
        faces = self.face_detector.detect(frame)
        all_detections.extend(faces)
        
        # Detect poses if no high-confidence faces
        if not any(f['confidence'] > 0.5 for f in faces):
            poses = self.pose_detector.detect(frame)
            all_detections.extend(poses)
        
        # Detect objects if no people with good confidence
        if not any(d['confidence'] > 0.3 for d in all_detections):
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
        
        # Select best detection based on confidence and type
        if all_detections:
            # Prioritize by type and confidence
            type_priority = {'face': 3, 'pose': 2, 'object': 1}
            
            # Sort by weighted score
            all_detections.sort(
                key=lambda x: (
                    type_priority.get(x['type'], 0) * x.get('confidence', 0)
                ),
                reverse=True
            )
            
            # Return best if confidence is sufficient
            best = all_detections[0]
            if best.get('confidence', 0) > 0.2:  # Lower threshold for detection
                return best
        
        return None
    
    def create_preview(
        self,
        original: np.ndarray,
        output: np.ndarray
    ) -> np.ndarray:
        """Create side-by-side preview with keyframe visualization"""
        h, w = original.shape[:2]
        
        # Scale for preview
        scale = self.config['video']['preview_scale']
        preview_h = int(h * scale)
        preview_w = int(w * scale)
        
        # Resize frames
        original_small = cv2.resize(original, (preview_w, preview_h))
        
        # Output might have different dimensions, so calculate its scale
        out_h, out_w = output.shape[:2]
        output_scale = min(preview_h / out_h, preview_w / out_w)
        output_preview_w = int(out_w * output_scale)
        output_preview_h = int(out_h * output_scale)
        output_small = cv2.resize(output, (output_preview_w, output_preview_h))
        
        # Pad output to match original preview size
        output_padded = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
        y_offset = (preview_h - output_preview_h) // 2
        x_offset = (preview_w - output_preview_w) // 2
        output_padded[y_offset:y_offset+output_preview_h, x_offset:x_offset+output_preview_w] = output_small
        
        # Draw overlay on original
        if self.current_crop:
            draw_tracking_overlay(
                original_small,
                self.current_crop,
                self.current_target,
                scale
            )
        
        # Draw keyframe indicators
        keyframe_positions = self.tracker.get_keyframe_positions()
        if keyframe_positions:
            # Find nearby keyframes
            current_frame = self.frame_count
            nearby_keyframes = [
                kf for kf in keyframe_positions
                if abs(kf[0] - current_frame) < 30
            ]
            
            # Draw keyframe markers
            for kf_frame, kf_x in nearby_keyframes:
                x = int(kf_x * scale)
                color = (0, 255, 255) if kf_frame == current_frame else (0, 128, 128)
                cv2.line(original_small, (x, 0), (x, preview_h), color, 1)
        
        # Combine side by side
        preview = np.hstack([original_small, output_padded])
        
        # Add labels
        cv2.putText(
            preview, "Original + Tracking", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            preview, "Reframed Output", (preview_w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # Add confidence info if target exists
        if self.current_target and 'confidence' in self.current_target:
            conf_text = f"Confidence: {self.current_target['confidence']:.2f}"
            cv2.putText(
                preview, conf_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        
        return preview
