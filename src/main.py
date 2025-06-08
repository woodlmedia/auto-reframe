#!/usr/bin/env python3
"""
Main entry point for the Auto Reframe application
"""

import argparse
import sys
from pathlib import Path
import yaml
import logging
from typing import Tuple, Optional

from .reframer.video_processor import VideoProcessor
from .utils.video_utils import get_video_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoReframer:
    """Main class for automatic video reframing"""
    
    def __init__(
        self,
        output_aspect_ratio: Tuple[int, int] = (9, 16),
        smoothing_factor: float = 0.1,
        detection_confidence: float = 0.5,
        config_path: Optional[str] = None
    ):
        """
        Initialize the Auto Reframer
        
        Args:
            output_aspect_ratio: Target aspect ratio (width, height)
            smoothing_factor: Smoothing for camera movements (0-1)
            detection_confidence: Minimum confidence for detections
            config_path: Path to custom config file
        """
        # Load config
        self.config = self._load_config(config_path)
        
        # Override with parameters
        self.config['reframing']['default_aspect_ratio'] = list(output_aspect_ratio)
        self.config['tracking']['smoothing_factor'] = smoothing_factor
        self.config['detection']['face_confidence'] = detection_confidence
        
        # Initialize processor
        self.processor = VideoProcessor(self.config)
        
    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """Load configuration from file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        preview: bool = False,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """
        Process a video file
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            preview: Show preview window during processing
            start_time: Start time in seconds
            end_time: End time in seconds
        """
        # Validate input
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Get video info
        info = get_video_info(str(input_path))
        logger.info(f"Input video: {info['width']}x{info['height']}, "
                   f"{info['fps']} fps, {info['duration']:.1f}s")
        
        # Create output directory
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process video
        logger.info("Processing video...")
        self.processor.process(
            str(input_path),
            str(output_path),
            preview=preview,
            start_time=start_time,
            end_time=end_time
        )
        
        logger.info(f"Output saved to: {output_path}")


def parse_aspect_ratio(value: str) -> Tuple[int, int]:
    """Parse aspect ratio from string (e.g., '9:16')"""
    try:
        w, h = map(int, value.split(':'))
        return (w, h)
    except:
        raise argparse.ArgumentTypeError(
            f"Invalid aspect ratio: {value}. Use format 'W:H' (e.g., '9:16')"
        )


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Automatically reframe videos with intelligent tracking"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video file"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output video file"
    )
    
    parser.add_argument(
        "--aspect", "-a",
        type=parse_aspect_ratio,
        default="9:16",
        help="Output aspect ratio (default: 9:16)"
    )
    
    parser.add_argument(
        "--smoothing", "-s",
        type=float,
        default=0.1,
        help="Camera smoothing factor 0-1 (default: 0.1)"
    )
    
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Detection confidence 0-1 (default: 0.5)"
    )
    
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Show preview window"
    )
    
    parser.add_argument(
        "--start",
        type=float,
        help="Start time in seconds"
    )
    
    parser.add_argument(
        "--end",
        type=float,
        help="End time in seconds"
    )
    
    parser.add_argument(
        "--config",
        help="Path to custom config file"
    )
    
    args = parser.parse_args()
    
    try:
        # Create reframer
        reframer = AutoReframer(
            output_aspect_ratio=args.aspect,
            smoothing_factor=args.smoothing,
            detection_confidence=args.confidence,
            config_path=args.config
        )
        
        # Process video
        reframer.process_video(
            args.input,
            args.output,
            preview=args.preview,
            start_time=args.start,
            end_time=args.end
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
