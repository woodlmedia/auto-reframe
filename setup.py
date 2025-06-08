```python
from setuptools import setup, find_packages

setup(
    name="auto-reframe",
    version="1.0.0",
    author="Your Name",
    description="Intelligent video reframing and stabilization",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "ultralytics>=8.0.0",
        "filterpy>=1.4.5",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
