import numpy as np
from typing import Tuple, Optional, Dict


class CropCalculator:
    """Calculate crop windows for video reframing - X-axis only"""
    
    def __init__(
        self,
        output_aspect_ratio: Tuple[int, int],
        padding_ratio: float = 0.1,
        min_zoom: float = 1.0,  # Always 1.0 - no zoom
        max_zoom: float = 1.0,  # Always 1.0 - no zoom
        lead_room: float = 0.0
    ):
        """
        Initialize crop calculator
        
        Args:
            output_aspect_ratio: Target aspect ratio (width, height)
            padding_ratio: Not used (kept for compatibility)
            min_zoom: Ignored - always 1.0
            max_zoom: Ignored - always 1.0
            lead_room: Horizontal offset for subject (0-0.5)
        """
        self.output_aspect = output_aspect_ratio[0] / output_aspect_ratio[1]
        self.lead_room = lead_room
        
        # Force no zoom
        self.min_zoom = 1.0
        self.max_zoom = 1.0
        
    def calculate_crop(
        self,
        target_x: float,
        frame_shape: Tuple[int, int],
        previous_crop: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate crop window (X-axis adjustment only)
        
        Args:
            target_x: X position of target
            frame_shape: (height, width) of frame
            previous_crop: Previous crop for smooth transitions
            
        Returns:
            Crop dictionary with x, y, width, height
        """
        h, w = frame_shape
        frame_aspect = w / h
        
        # Calculate fixed crop dimensions (no zoom)
        if frame_aspect > self.output_aspect:
            # Frame is wider than output - fit height
            crop_height = h
            crop_width = int(crop_height * self.output_aspect)
        else:
            # Frame is taller than output - fit width
            crop_width = w
            crop_height = int(crop_width / self.output_aspect)
        
        # Fixed Y position (centered vertically)
        y = (h - crop_height) // 2
        
        # Calculate X position based on target
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
        
        # Constrain to frame bounds (X-axis only)
        x = max(0, min(x, w - crop_width))
        
        return {
            'x': x,
            'y': y,
            'width': crop_width,
            'height': crop_height,
            'center_x': x + crop_width / 2,
            'center_y': h / 2  # Always centered vertically
        }
    
    def smooth_transition(
        self,
        current_crop: Dict,
        target_crop: Dict,
        smoothing: float = 0.1
    ) -> Dict:
        """
        Smooth transition between crops (X-axis only)
        
        Args:
            current_crop: Current crop window
            target_crop: Target crop window
            smoothing: Smoothing factor (0-1)
            
        Returns:
            Smoothed crop window
        """
        # Only smooth X position, keep everything else fixed
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
        Apply crop to frame
        
        Args:
            frame: Input frame
            crop: Crop window dictionary
            output_size: Output size (width, height)
            
        Returns:
            Cropped and resized frame
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
        
        cropped = frame[y:y+h, x:x+w]
        
        # Resize if output size specified
        if output_size and (output_size[0] != w or output_size[1] != h):
            cropped = cv2.resize(
                cropped,
                output_size,
                interpolation=cv2.INTER_LINEAR
            )
        
        return cropped
