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
