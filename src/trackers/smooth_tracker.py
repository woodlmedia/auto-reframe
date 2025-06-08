import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import deque
from scipy.interpolate import interp1d


class KeyframeTracker:
    """Keyframe-based tracker with confidence thresholds"""
    
    def __init__(
        self,
        confidence_threshold: float = 0.25,
        interpolation_method: str = 'cubic',
        min_keyframe_distance: int = 5
    ):
        """
        Initialize keyframe tracker
        
        Args:
            confidence_threshold: Minimum confidence to add keyframe
            interpolation_method: Method for interpolating between keyframes
            min_keyframe_distance: Minimum frames between keyframes
        """
        self.confidence_threshold = confidence_threshold
        self.interpolation_method = interpolation_method
        self.min_keyframe_distance = min_keyframe_distance
        
        # Keyframe storage
        self.keyframes = []  # List of (frame_num, x_position, confidence)
        self.current_frame = 0
        self.last_keyframe_frame = -min_keyframe_distance
        
        # Current interpolated position
        self.current_x = None
        self.frame_positions = {}  # Cache for interpolated positions
        
    def update(
        self,
        x_position: float,
        confidence: float,
        frame_num: Optional[int] = None,
        force_keyframe: bool = False
    ) -> float:
        """
        Update tracker with new measurement
        
        Args:
            x_position: X position of target
            confidence: Detection confidence (0-1)
            frame_num: Current frame number
            force_keyframe: Force adding a keyframe (for scene changes)
            
        Returns:
            Interpolated X position
        """
        if frame_num is None:
            frame_num = self.current_frame
        
        self.current_frame = frame_num
        
        # Initialize if first frame
        if not self.keyframes:
            self.keyframes.append((frame_num, x_position, confidence))
            self.current_x = x_position
            self.last_keyframe_frame = frame_num
            return x_position
        
        # Check if we should add a keyframe
        should_add_keyframe = (
            force_keyframe or
            (confidence >= self.confidence_threshold and
             frame_num - self.last_keyframe_frame >= self.min_keyframe_distance)
        )
        
        if should_add_keyframe:
            # Add new keyframe
            self.keyframes.append((frame_num, x_position, confidence))
            self.last_keyframe_frame = frame_num
            
            # Clear interpolation cache after this frame
            self.frame_positions = {
                k: v for k, v in self.frame_positions.items()
                if k <= frame_num
            }
        
        # Get interpolated position
        self.current_x = self._get_interpolated_position(frame_num)
        return self.current_x
    
    def _get_interpolated_position(self, frame_num: int) -> float:
        """Get interpolated position for given frame"""
        # Check cache
        if frame_num in self.frame_positions:
            return self.frame_positions[frame_num]
        
        # Need at least 2 keyframes to interpolate
        if len(self.keyframes) < 2:
            return self.keyframes[0][1]
        
        # Extract keyframe data
        frames = [kf[0] for kf in self.keyframes]
        positions = [kf[1] for kf in self.keyframes]
        
        # Handle edge cases
        if frame_num <= frames[0]:
            position = positions[0]
        elif frame_num >= frames[-1]:
            position = positions[-1]
        else:
            # Interpolate
            if self.interpolation_method == 'cubic' and len(frames) >= 4:
                # Use cubic interpolation
                interp_func = interp1d(
                    frames, positions,
                    kind='cubic',
                    fill_value='extrapolate'
                )
            else:
                # Fall back to linear
                interp_func = interp1d(
                    frames, positions,
                    kind='linear',
                    fill_value='extrapolate'
                )
            
            position = float(interp_func(frame_num))
        
        # Cache result
        self.frame_positions[frame_num] = position
        return position
    
    def add_scene_boundary(self, frame_num: int, x_position: float):
        """Add keyframes for scene change"""
        # Add keyframe at end of previous scene
        if frame_num > 0:
            prev_x = self._get_interpolated_position(frame_num - 1)
            self.keyframes.append((frame_num - 1, prev_x, 1.0))
        
        # Add keyframe at start of new scene
        self.keyframes.append((frame_num, x_position, 1.0))
        self.last_keyframe_frame = frame_num
        
        # Sort keyframes by frame number
        self.keyframes.sort(key=lambda x: x[0])
        
        # Clear cache
        self.frame_positions.clear()
    
    def get_keyframe_positions(self) -> List[Tuple[int, float]]:
        """Get all keyframe positions for visualization"""
        return [(kf[0], kf[1]) for kf in self.keyframes]
    
    def reset(self):
        """Reset tracker state"""
        self.keyframes.clear()
        self.current_frame = 0
        self.last_keyframe_frame = -self.min_keyframe_distance
        self.current_x = None
        self.frame_positions.clear()


class SmoothTracker(KeyframeTracker):
    """Compatibility wrapper for old SmoothTracker interface"""
    
    def __init__(self, smoothing_factor: float = 0.1, history_size: int = 30):
        # Convert smoothing factor to confidence threshold
        # Higher smoothing = lower confidence threshold
        confidence_threshold = 0.25 * (1.0 - smoothing_factor)
        super().__init__(confidence_threshold=confidence_threshold)
        
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        # Extract X position and use default confidence
        x_pos = self.update(measurement[0], confidence=0.5)
        # Return same Y position (no vertical movement)
        return (x_pos, measurement[1])
    
    def predict_next(self) -> Tuple[float, float]:
        if self.current_x is None:
            return (0, 0)
        return (self.current_x, 0)
    
    def reset(self, position: Optional[Tuple[float, float]] = None):
        super().reset()
        if position:
            self.update(position[0], confidence=1.0, frame_num=0)
