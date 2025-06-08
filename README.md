# Auto Reframe - X-Axis Only Video Stabilization

Automatically reframe videos by tracking subjects and adjusting ONLY the horizontal position. **NO zooming, stretching, or resizing** - just intelligent X-axis shifting to keep subjects centered.

## Key Features

- 🎯 **X-Axis Only Movement**: Only shifts horizontally, no vertical movement
- 🚫 **No Zoom or Resize**: Maintains original video scale - no stretching or warping
- 📊 **Keyframe-Based Tracking**: Only adds keyframes when confidence > 25%
- 🎬 **Scene Detection**: Automatic keyframes at scene changes
- 🔍 **Multi-Level Detection**: Face → Body → Objects fallback
- 📐 **Aspect Ratio Cropping**: Crops to desired aspect ratio without resizing

## How It Works

1. **Analyzes the video** to detect high-confidence targets
2. **Creates keyframes** only when detection confidence exceeds 25%
3. **Interpolates smoothly** between keyframes using cubic splines
4. **Crops to aspect ratio** without any resizing
5. **Outputs at crop dimensions** - no stretching or scaling

## Important: Output Behavior

- **Input**: 1920x1080 (16:9)
- **Target Aspect**: 9:16 (vertical)
- **Output**: 608x1080 (cropped width, same height)
- **NO RESIZING** - output dimensions = crop dimensions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/auto-reframe.git
cd auto-reframe

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
Quick Start
pythonfrom src.main import AutoReframer

# Initialize the reframer
reframer = AutoReframer(
    output_aspect_ratio=(9, 16),  # Crop to vertical
    smoothing_factor=0.1,
    detection_confidence=0.7
)

# Process video - output will be cropped, NOT resized
reframer.process_video(
    input_path="input_video.mp4",
    output_path="output_reframed.mp4"
)
Command Line Usage
bash# Basic usage - crop to 9:16, NO resize
python src/main.py --input video.mp4 --output reframed.mp4 --aspect 9:16

# With higher confidence threshold
python src/main.py --input video.mp4 --output reframed.mp4 --confidence 0.8

# With preview to see tracking
python src/main.py --input video.mp4 --output reframed.mp4 --preview
Configuration
Key settings in config/default_config.yaml:
yamltracking:
  confidence_threshold: 0.25  # Min confidence for keyframes
  interpolation_method: "cubic"  # Smooth curves

reframing:
  x_axis_only: true  # Only horizontal movement
  min_zoom: 1.0  # No zoom
  max_zoom: 1.0  # No zoom

scene_detection:
  enabled: true  # Auto keyframes at cuts
How Keyframe Tracking Works

Detection: Face/body/object detection runs on each frame
Confidence Check: Only positions with >25% confidence create keyframes
Scene Changes: Automatic keyframes at scene boundaries
Interpolation: Cubic spline interpolation between keyframes
X-Axis Only: Only horizontal position changes, no vertical movement

Examples
Convert horizontal to vertical (no resize):
bashpython src/main.py --input horizontal.mp4 --output vertical.mp4 --aspect 9:16
Square crop for Instagram (no resize):
bashpython src/main.py --input video.mp4 --output square.mp4 --aspect 1:1
High confidence tracking only:
bashpython src/main.py --input shaky.mp4 --output stable.mp4 --confidence 0.8
Output Dimensions
The output video dimensions depend on your input and aspect ratio:
InputTarget AspectOutput SizeNote1920x10809:16608x1080Crops width1920x10801:11080x1080Crops width1080x192016:91080x608Crops height1920x108016:91920x1080No crop needed
License
MIT License

# Initialize the reframer
reframer = AutoReframer(
    output_aspect_ratio=(9, 16),  # Vertical video
    smoothing_factor=0.1,
    detection_confidence=0.5
)

# Process a video
reframer.process_video(
    input_path="input_video.mp4",
    output_path="output_reframed.mp4"
)
Command Line Usage
bash# Basic usage
python src/main.py --input video.mp4 --output reframed.mp4 --aspect 9:16

# With custom parameters
python src/main.py --input video.mp4 --output reframed.mp4 \
    --aspect 1:1 --smoothing 0.05 --padding 0.1
Configuration
Edit config/default_config.yaml to change default parameters:

Detection thresholds
Smoothing parameters
Output quality settings
Object saliency preferences

License
MIT License
