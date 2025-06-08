"""
Kalman filter for smooth tracking
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Tuple, Optional


class KalmanTracker:
    """Kalman filter for smooth position tracking"""
    
    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        initial_position: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize Kalman filter
        
        Args:
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
            initial_position: Initial (x, y) position
        """
        # Create 4D Kalman filter (x, y, vx, vy)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0],  # Measure x
            [0, 1, 0, 0]   # Measure y
        ])
        
        # Measurement noise
        self.kf.R = np.eye(2) * measurement_noise
        
        # Process noise
        self.kf.Q = np.eye(4) * process_noise
        
        # Initial uncertainty
        self.kf.P *= 100
        
        # Initialize position if provided
        if initial_position:
            self.kf.x[0] = initial_position[0]
            self.kf.x[1] = initial_position[1]
            
        self.initialized = initial_position is not None
        
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """
        Update filter with new measurement
        
        Args:
            measurement: (x, y) position
            
        Returns:
            Filtered (x, y) position
        """
        if not self.initialized:
            # First measurement - initialize state
            self.kf.x[0] = measurement[0]
            self.kf.x[1] = measurement[1]
            self.initialized = True
        else:
            # Predict and update
            self.kf.predict()
            self.kf.update(np.array(measurement))
        
        return (self.kf.x[0], self.kf.x[1])
    
    def predict_next(self) -> Tuple[float, float]:
        """Predict next position without measurement"""
        # Create a copy to not affect internal state
        x_pred = self.kf.F @ self.kf.x
        return (x_pred[0], x_pred[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate"""
        return (self.kf.x[2], self.kf.x[3])
    
    def reset(self, position: Optional[Tuple[float, float]] = None):
        """Reset filter state"""
        self.kf.x = np.zeros(4)
        self.kf.P = np.eye(4) * 100
        
        if position:
            self.kf.x[0] = position[0]
            self.kf.x[1] = position[1]
            self.initialized = True
        else:
            self.initialized = False
