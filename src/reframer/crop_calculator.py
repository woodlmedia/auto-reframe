import numpy as np
from typing import Tuple, Optional, Dict


class CropCalculator:
    """Calculate crop windows - only shifts X position, no resize or zoom"""
    
    def __init__(
        self,
        output_aspect_ratio: Tuple[int, int],
        padding_ratio: float = 0.0,  # Not used
        min_zoom: float = 1.0,  # Not used
        max_zoom: float = 1.0,  # Not used
        lead_room: float = 0.0
    ):
        """
        Initialize crop calculator
        
        Args:
            output_aspect_ratio: Target aspect ratio (width, height)
            lead_room: Horizontal offset for subject (0-0.5)
        """
        self.output_aspect = output_aspect_ratio[0] / output_aspect_ratio[1]
        self.lead_room = lead_room
        
    def calculate_crop(
        self,
        target_x: float,
        frame_shape: Tuple[int, int],
        previous_crop: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate crop window - ONLY X position shift
        
        Args:
            target_x: X position of target
            frame_shape: (height, width) of frame
            previous_crop: Previous crop for smooth transitions
            
        Returns:
            Crop dictionary with x, y, width, height
        """
        h, w = frame_shape
        frame_aspect = w / h
        
        # Calculate crop width based on output aspect ratio
        # But we'll use the full frame height to avoid any vertical cropping
        if frame_aspect > self.output_aspect:
            # Frame is wider than output - we need to crop horizontally
            crop_height = h
            crop_width = int(crop_height * self.output_aspect)
        else:
            # Frame aspect matches or is narrower - use full frame
            crop_width = w
            crop_height = h
        
        # Always start at top of frame (no vertical movement)
        y = 0
        
        # Calculate X position to center on target
        center_x = target_x
        
        # Apply lead room if specified
        if self.lead_room != 0 and previous_crop:
            prev_center = previous_crop['center_x']
            if center_x > prev_center:
                # Moving right - add lead room to right
                center_x += crop_width * self.lead_room * 0.3
            elif center_x < prev_center:
                # Moving left - add lead room to left
                center_x -= crop_width * self.lead_room * 0.3
        
        # Calculate crop X position
        x = int(center_x - crop_width / 2)
        
        # Constrain to frame bounds
        x = max(0, min(x, w - crop_width))
        
        return {
            'x': x,
            'y': y,
            'width': crop_width,
            'height': crop_height,
            'center_x': x + crop_width / 2,
            'center_y': h / 2,
            # Store original dimensions for no-resize mode
            'original_width': w,
            'original_height': h
        }
    
    def smooth_transition(
        self,
        current_crop: Dict,
        target_crop: Dict,
        smoothing: float = 0.1
    ) -> Dict:
        """
        Smooth transition between crops (X position only)
        
        Args:
            current_crop: Current crop window
            target_crop: Target crop window
            smoothing: Smoothing factor (0-1)
            
        Returns:
            Smoothed crop window
        """
        # Only smooth X position
        smoothed = target_crop.copy()
        
        smoothed['x'] = int(
            smoothing * target_crop['x'] + 
            (1 - smoothing) * current_crop['x']
        )
        
        # Update center
        smoothed['center_x'] = smoothed['x'] + smoothed['width'] / 2
        
        return smoothed
    
    def apply_crop(
        self,
        frame: np.ndarray,
        crop: Dict,
        output_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Apply crop to frame - NO RESIZING
        
        Args:
            frame: Input frame
            crop: Crop window dictionary
            output_size: IGNORED - we never resize
            
        Returns:
            Cropped frame at original resolution
        """
        import cv2
        
        # Extract crop region
        x, y = crop['x'], crop['y']
        w, h = crop['width'], crop['height']
        
        # Ensure crop bounds are within frame
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - w))
        y = max(0, min(y, frame_h - h))
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)
        
        # Just crop, NO RESIZE
        cropped = frame[y:y+h, x:x+w]
        
        return cropped
