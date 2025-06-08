import cv2
import numpy as np
from typing import Optional, Tuple, List
from collections import deque


class SceneDetector:
    """Detect scene changes/cuts in video"""
    
    def __init__(
        self,
        threshold: float = 30.0,
        min_scene_length: int = 10,
        use_histogram: bool = True
    ):
        """
        Initialize scene detector
        
        Args:
            threshold: Threshold for scene change detection
            min_scene_length: Minimum frames between scene changes
            use_histogram: Use histogram comparison (more robust)
        """
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.use_histogram = use_histogram
        
        # State
        self.prev_frame = None
        self.prev_hist = None
        self.frames_since_cut = 0
        self.scene_changes = []
        
    def detect_scene_change(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> bool:
        """
        Detect if current frame is a scene change
        
        Args:
            frame: Current frame
            frame_num: Frame number
            
        Returns:
            True if scene change detected
        """
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First frame
        if self.prev_frame is None:
            self.prev_frame = gray
            if self.use_histogram:
                self.prev_hist = self._calculate_histogram(gray)
            return False
        
        # Check for scene change
        is_scene_change = False
        
        if self.use_histogram:
            # Histogram comparison (more robust to motion)
            curr_hist = self._calculate_histogram(gray)
            
            # Compare histograms
            correlation = cv2.compareHist(
                self.prev_hist, curr_hist,
                cv2.HISTCMP_CORREL
            )
            
            # Lower correlation = bigger change
            diff_score = (1.0 - correlation) * 100
            
            if diff_score > self.threshold and self.frames_since_cut >= self.min_scene_length:
                is_scene_change = True
            
            self.prev_hist = curr_hist
        else:
            # Simple frame difference
            diff = cv2.absdiff(self.prev_frame, gray)
            diff_score = np.mean(diff)
            
            if diff_score > self.threshold and self.frames_since_cut >= self.min_scene_length:
                is_scene_change = True
        
        # Update state
        self.prev_frame = gray
        
        if is_scene_change:
            self.scene_changes.append(frame_num)
            self.frames_since_cut = 0
        else:
            self.frames_since_cut += 1
        
        return is_scene_change
    
    def _calculate_histogram(self, gray_frame: np.ndarray) -> np.ndarray:
        """Calculate normalized histogram for frame"""
        # Calculate histogram
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        
        # Normalize
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def get_scene_boundaries(self) -> List[int]:
        """Get all detected scene boundaries"""
        return self.scene_changes.copy()
    
    def reset(self):
        """Reset detector state"""
        self.prev_frame = None
        self.prev_hist = None
        self.frames_since_cut = 0
        self.scene_changes.clear()


class AdaptiveSceneDetector(SceneDetector):
    """Scene detector with adaptive threshold"""
    
    def __init__(
        self,
        initial_threshold: float = 30.0,
        min_scene_length: int = 10,
        adaptation_rate: float = 0.1
    ):
        super().__init__(initial_threshold, min_scene_length, use_histogram=True)
        
        self.initial_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.diff_history = deque(maxlen=100)
        
    def detect_scene_change(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> bool:
        """Detect scene change with adaptive threshold"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_hist = self._calculate_histogram(gray)
            return False
        
        # Calculate difference
        curr_hist = self._calculate_histogram(gray)
        correlation = cv2.compareHist(
            self.prev_hist, curr_hist,
            cv2.HISTCMP_CORREL
        )
        diff_score = (1.0 - correlation) * 100
        
        # Add to history
        self.diff_history.append(diff_score)
        
        # Adapt threshold based on recent history
        if len(self.diff_history) > 10:
            mean_diff = np.mean(self.diff_history)
            std_diff = np.std(self.diff_history)
            
            # Adaptive threshold: mean + k * std
            adaptive_threshold = mean_diff + 2.5 * std_diff
            
            # Blend with initial threshold
            self.threshold = (
                self.adaptation_rate * adaptive_threshold +
                (1 - self.adaptation_rate) * self.initial_threshold
            )
        
        # Check for scene change
        is_scene_change = (
            diff_score > self.threshold and
            self.frames_since_cut >= self.min_scene_length
        )
        
        # Update state
        self.prev_frame = gray
        self.prev_hist = curr_hist
        
        if is_scene_change:
            self.scene_changes.append(frame_num)
            self.frames_since_cut = 0
        else:
            self.frames_since_cut += 1
        
        return is_scene_change

### src/utils/video_utils.py
```python
"""
Video utility functions
"""

import cv2
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def get_video_info(video_path: str) -> Dict:
    """
    Get video information
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': 0.0
    }
    
    if info['fps'] > 0:
        info['duration'] = info['frame_count'] / info['fps']
    
    cap.release()
    
    return info


def create_video_writer(
    output_path: str,
    fps: float,
    frame_size: Tuple[int, int],
    codec: str = 'mp4v'
) -> cv2.VideoWriter:
    """
    Create video writer
    
    Args:
        output_path: Output file path
        fps: Frames per second
        frame_size: (width, height)
        codec: Video codec
        
    Returns:
        VideoWriter object
    """
    # Get codec
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    # Create writer
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        frame_size
    )
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")
    
    return writer
