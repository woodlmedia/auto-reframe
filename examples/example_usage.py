from src.main import AutoReframer
import logging

# Set up logging to see keyframe info
logging.basicConfig(level=logging.INFO)


def main():
    # Example 1: Basic usage with X-axis only tracking
    print("Example 1: Basic vertical video conversion (X-axis only)")
    reframer = AutoReframer(
        output_aspect_ratio=(9, 16),  # Vertical video
        smoothing_factor=0.1
    )
    
    # Enable keyframe tracking
    reframer.config['tracking']['use_keyframes'] = True
    reframer.config['tracking']['confidence_threshold'] = 0.25
    
    reframer.process_video(
        "samples/horizontal_video.mp4",
        "output/vertical_reframed.mp4",
        preview=True
    )
    
    # Example 2: Square video with scene detection
    print("\nExample 2: Square video with scene detection")
    reframer = AutoReframer(
        output_aspect_ratio=(1, 1),
        detection_confidence=0.7  # Higher confidence for stable tracking
    )
    
    # Enable scene detection
    reframer.config['scene_detection']['enabled'] = True
    reframer.config['scene_detection']['use_adaptive'] = True
    
    reframer.process_video(
        "samples/action_video.mp4",
        "output/square_reframed.mp4"
    )
    
    # Example 3: Process video with custom keyframe settings
    print("\nExample 3: Custom keyframe tracking")
    reframer = AutoReframer()
    
    # Configure keyframe tracking
    reframer.config['tracking']['use_keyframes'] = True
    reframer.config['tracking']['confidence_threshold'] = 0.3  # Higher threshold
    reframer.config['tracking']['min_keyframe_distance'] = 10  # More spacing
    reframer.config['tracking']['interpolation_method'] = 'cubic'  # Smooth curves
    
    # X-axis only (no zoom or vertical movement)
    reframer.config['reframing']['x_axis_only'] = True
    
    reframer.process_video(
        "samples/interview_video.mp4",
        "output/interview_reframed.mp4",
        preview=True  # See keyframe indicators in preview
    )
    
    # Example 4: Fast action with minimal keyframes
    print("\nExample 4: Fast action video")
    reframer = AutoReframer(
        output_aspect_ratio=(9, 16),
        detection_confidence=0.8  # Very high confidence only
    )
    
    # Only track very confident detections
    reframer.config['tracking']['confidence_threshold'] = 0.5
    reframer.config['tracking']['min_keyframe_distance'] = 15
    
    # Enable lead room for action
    reframer.config['reframing']['lead_room'] = 0.2
    
    reframer.process_video(
        "samples/sports_video.mp4",
        "output/sports_reframed.mp4"
    )


if __name__ == "__main__":
    main()
