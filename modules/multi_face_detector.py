"""
Multi-face detection module.
Detects when multiple people are present in the frame.
"""

import cv2
import torch
from facenet_pytorch import MTCNN
from typing import Optional, Dict, Any


class MultiFaceDetector:
    """Detects multiple faces in a frame."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-face detector.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            keep_all=True,
            post_process=False,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            device=self.device
        )
        
        # Configuration
        if config:
            multi_face_config = config.get('detection', {}).get('multi_face', {})
            self.threshold = multi_face_config.get('alert_threshold', 5)
        else:
            self.threshold = 5
        
        self.consecutive_frames = 0
        self.alert_logger = None
    
    def set_alert_logger(self, alert_logger):
        """Set alert logger for logging alerts."""
        self.alert_logger = alert_logger
    
    def detect_multiple_faces(self, frame: cv2.typing.MatLike) -> bool:
        """
        Detect if multiple faces are present in the frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            True if multiple faces detected for threshold frames, False otherwise
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = self.detector.detect(rgb_frame)
            
            if boxes is not None and len(boxes) > 1:
                # Count faces with high confidence
                high_conf_faces = sum(p > 0.9 for p in probs) if probs is not None else 0
                
                if high_conf_faces >= 2:
                    self.consecutive_frames += 1
                    if self.consecutive_frames >= self.threshold and self.alert_logger:
                        self.alert_logger.log_alert(
                            "MULTIPLE_FACES",
                            f"Detected {high_conf_faces} faces for {self.consecutive_frames} frames"
                        )
                    return True
            else:
                self.consecutive_frames = 0
                
            return False
        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert(
                    "MULTI_FACE_DETECTION_ERROR",
                    f"Error in multi-face detection: {str(e)}"
                )
            return False

