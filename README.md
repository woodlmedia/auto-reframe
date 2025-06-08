# Auto Reframe - Intelligent Video Stabilization & Reframing

Automatically reframe and stabilize videos by tracking people or salient objects. Perfect for converting horizontal videos to vertical format while keeping subjects in frame.

## Features

- 🎯 Automatic face and body tracking using MediaPipe
- 🔍 Intelligent object detection when no people are present
- 📐 Smooth camera movements with Kalman filtering
- 🎬 Support for multiple output aspect ratios (9:16, 1:1, 4:5, etc.)
- ⚡ Real-time preview mode
- 🎨 Customizable tracking parameters

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
