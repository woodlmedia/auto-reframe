import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class CropCalculator:
    """Calculate crop windows for video reframing"""
    
    def __init__(
        self,
        output_aspect_ratio: Tuple[int, int],
        padding_ratio: float = 0.1,
        min_zoom: float = 0.5,
        max_zoom: float = 2.0,
        lead_room: float = 0.0
    ):
        """
        Initialize crop calculator
        
        Args:
            output_aspect_ratio: Target aspect ratio (width, height)
            padding_ratio: Extra padding around subject (0-1)
            min_zoom: Minimum zoom level
            max_zoom: Maximum zoom level
            lead_room: Horizontal offset for subject (0-0.5)
        """
        self.output_aspect = output_aspect_ratio[0] / output_aspect_ratio[1]
        self.padding_ratio = padding_ratio
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.lead_room = lead_room
        
    def calculate_crop(
        self,
        target_center: Tuple[float, float],
        target_size: Optional[float],
        frame_shape: Tuple[int, int],
        previous_crop: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate crop window
        
        Args:
            target_center: (x, y) center of target
            target_size: Size of target (for zoom calculation)
            frame_shape: (height, width) of frame
            previous_crop: Previous crop for smooth transitions
            
        Returns:
            Crop dictionary with x, y, width, height
        """
        h, w = frame_shape
        frame_aspect = w / h
        
        # Calculate base crop dimensions
        if frame_aspect > self.output_aspect:
            # Frame is wider than output - fit height
            crop_height = h
            crop_width = int(crop_height * self.output_aspect)
        else:
            # Frame is taller than output - fit width
            crop_width = w
            crop_height = int(crop_width / self.output_aspect)
        
        # Apply zoom based on target size
        if target_size is not None:
            # Estimate ideal crop size to frame subject
            ideal_height = target_size * (1 + self.padding_ratio) * 2
            zoom = np.clip(h / ideal_height, self.min_zoom, self.max_zoom)
            
            crop_width = int(crop_width / zoom)
            crop_height = int(crop_height / zoom)
        
        # Apply lead room
        center_x = target_center[0]
        if self.lead_room != 0:
            # Offset center based on movement direction
            if previous_crop:
                prev_center = previous_crop['center_x']
                if center_x > prev_center:
                    # Moving right - add lead room to right
                    center_x += crop_width * self.lead_room * 0.5
                elif center_x < prev_center:
                    # Moving left - add lead room to left
                    center_x -= crop_width * self.lead_room * 0.5
        
        # Calculate crop position
        x = int(center_x - crop_width / 2)
        y = int(target_center[1] - crop_height / 2)
        
        # Constrain to frame bounds
        x = max(0, min(x, w - crop_width))
        y = max(0, min(y, h - crop_height))
        
        # Ensure minimum crop size
        crop_width = max(100, crop_width)
        crop_height = max(100, crop_height)
        
        return {
            'x': x,
            'y': y,
            'width': crop_width,
            'height': crop_height,
            'center_x': center_x,
            'center_y': target_center[1]
        }
    
    def smooth_transition(
        self,
        current_crop: Dict,
        target_crop: Dict,
        smoothing: float = 0.1
    ) -> Dict:
        """
        Smooth transition between crops
        
        Args:
            current_crop: Current crop window
            target_crop: Target crop window
            smoothing: Smoothing factor (0-1)
            
        Returns:
            Smoothed crop window
        """
        # Exponential smoothing for all parameters
        smoothed = {}
        
        for key in ['x', 'y', 'width', 'height']:
            smoothed[key] = int(
                smoothing * target_crop[key] + 
                (1 - smoothing) * current_crop[key]
            )
        
        # Update centers
        smoothed['center_x'] = smoothed['x'] + smoothed['width'] / 2
        smoothed['center_y'] = smoothed['y'] + smoothed['height'] / 2
        
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
        # Extract crop region
        x, y = crop['x'], crop['y']
        w, h = crop['width'], crop['height']
        
        cropped = frame[y:y+h, x:x+w]
        
        # Resize if output size specified
        if output_size:
            cropped = cv2.resize(
                cropped,
                output_size,
                interpolation=cv2.INTER_LINEAR
            )
        
        return cropped
