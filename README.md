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
