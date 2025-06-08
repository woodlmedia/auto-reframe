from src.main import AutoReframer


def main():
    # Example 1: Basic usage
    print("Example 1: Basic vertical video conversion")
    reframer = AutoReframer(
        output_aspect_ratio=(9, 16),  # Vertical video
        smoothing_factor=0.1
    )
    
    reframer.process_video(
        "samples/horizontal_video.mp4",
        "output/vertical_reframed.mp4",
        preview=True
    )
    
    # Example 2: Square video with custom settings
    print("\nExample 2: Square video with tight framing")
    reframer = AutoReframer(
        output_aspect_ratio=(1, 1),
        smoothing_factor=0.05,  # Smoother camera
        detection_confidence=0.7  # Higher confidence threshold
    )
    
    # Load custom config
    reframer.config['reframing']['padding_ratio'] = 0.05  # Tighter framing
    reframer.config['reframing']['max_zoom'] = 3.0  # Allow more zoom
    
    reframer.process_video(
        "samples/action_video.mp4",
        "output/square_reframed.mp4"
    )
    
    # Example 3: Process specific segment
    print("\nExample 3: Process video segment")
    reframer = AutoReframer()
    
    reframer.process_video(
        "samples/long_video.mp4",
        "output/segment_reframed.mp4",
        start_time=30.0,  # Start at 30 seconds
        end_time=90.0     # End at 90 seconds
    )


if __name__ == "__main__":
    main()
