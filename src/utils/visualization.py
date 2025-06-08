"""
Visualization utilities for debugging and preview
"""

import cv2
import numpy as np
from typing import Dict, Optional


def draw_tracking_overlay(
    frame: np.ndarray,
    crop: Dict,
    target: Optional[Dict],
    scale: float = 1.0
):
    """
    Draw tracking overlay on frame
    
    Args:
        frame: Frame to draw on
        crop: Current crop window
        target: Current target detection
        scale: Scale factor for coordinates
    """
    # Draw crop rectangle
    x = int(crop['x'] * scale)
    y = int(crop['y'] * scale)
    w = int(crop['width'] * scale)
    h = int(crop['height'] * scale)
    
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )
    
    # Draw target if available
    if target:
        if 'bbox' in target:
            # Draw bounding box
            bbox = target['bbox']
            x1 = int(bbox[0] * scale)
            y1 = int(bbox[1] * scale)
            x2 = int(bbox[2] * scale)
            y2 = int(bbox[3] * scale)
            
            color = (255, 0, 0) if target['type'] == 'face' else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw center point
        cx = int(target['center'][0] * scale)
        cy = int(target['center'][1] * scale)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        
        # Add label
        label = f"{target['type']}"
        if 'class' in target:
            label = f"{target['class']}"
        if 'confidence' in target:
            label += f" ({target['confidence']:.2f})"
        
        cv2.putText(
            frame,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
