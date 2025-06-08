import numpy as np
from typing import Tuple, Optional, List
from collections import deque


class SmoothTracker:
    """Exponential moving average tracker with motion prediction"""
    
    def __init__(
        self,
        smoothing_factor: float = 0.1,
        history_size: int = 30
    ):
        """
        Initialize smooth tracker
        
        Args:
            smoothing_factor: Smoothing factor (0-1), lower = smoother
            history_size: Number of positions to keep in history
        """
        self.alpha = smoothing_factor
        self.history = deque(maxlen=history_size)
        self.current_position = None
        self.velocity = (0.0, 0.0)
        
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """
        Update tracker with new measurement
        
        Args:
            measurement: (x, y) position
            
        Returns:
            Smoothed (x, y) position
        """
        if self.current_position is None:
            # First measurement
            self.current_position = measurement
        else:
            # Exponential smoothing
            self.current_position = (
                self.alpha * measurement[0] + (1 - self.alpha) * self.current_position[0],
                self.alpha * measurement[1] + (1 - self.alpha) * self.current_position[1]
            )
            
            # Update velocity estimate
            if len(self.history) > 0:
                prev = self.history[-1]
                self.velocity = (
                    self.current_position[0] - prev[0],
                    self.current_position[1] - prev[1]
                )
        
        self.history.append(self.current_position)
        return self.current_position
    
    def predict_next(self) -> Tuple[float, float]:
        """Predict next position based on velocity"""
        if self.current_position is None:
            return (0, 0)
        
        return (
            self.current_position[0] + self.velocity[0],
            self.current_position[1] + self.velocity[1]
        )
    
    def get_smooth_path(self, n_points: int = 5) -> List[Tuple[float, float]]:
        """Get smoothed path for interpolation"""
        if len(self.history) < 2:
            return list(self.history)
        
        # Use recent history for path
        points = list(self.history)[-n_points:]
        
        # Apply additional smoothing if needed
        if len(points) > 2:
            smoothed = [points[0]]
            for i in range(1, len(points) - 1):
                x = 0.25 * points[i-1][0] + 0.5 * points[i][0] + 0.25 * points[i+1][0]
                y = 0.25 * points[i-1][1] + 0.5 * points[i][1] + 0.25 * points[i+1][1]
                smoothed.append((x, y))
            smoothed.append(points[-1])
            return smoothed
        
        return points
    
    def reset(self, position: Optional[Tuple[float, float]] = None):
        """Reset tracker state"""
        self.history.clear()
        self.current_position = position
        self.velocity = (0.0, 0.0)
        
        if position:
            self.history.append(position)
