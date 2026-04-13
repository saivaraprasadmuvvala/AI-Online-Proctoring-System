"""
Violation capturer for saving evidence images.
Captures and labels violation screenshots.
"""

import cv2
import os
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np


class ViolationCapturer:
    """Captures and saves violation evidence images."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize violation capturer.
        
        Args:
            config: Configuration dictionary (optional)
        """
        if config:
            output_path = config.get('global', {}).get('output_path', './reports')
        else:
            output_path = './reports'
        
        self.output_dir = os.path.join(output_path, "violation_captures")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def capture_violation(self, frame: np.ndarray, violation_type: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Save violation screenshot with metadata.
        
        Args:
            frame: Image frame to save
            violation_type: Type of violation
            timestamp: Optional timestamp string
            
        Returns:
            Dictionary with violation capture information
        """
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{violation_type}_{timestamp}.jpg"
        path = os.path.join(self.output_dir, filename)
        
        # Draw violation label on image
        labeled_frame = frame.copy()
        cv2.putText(labeled_frame, f"{violation_type} - {timestamp}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imwrite(path, labeled_frame)
        return {
            'type': violation_type,
            'timestamp': timestamp,
            'image_path': os.path.abspath(path)
        }

