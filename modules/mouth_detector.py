"""
Mouth movement detection for talking/whispering detection.
Uses MediaPipe face landmarks to detect mouth opening/closing.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Optional
from collections import deque


class MouthDetector:
    """Detects mouth movements indicating talking or whispering."""
    
    # MediaPipe mouth landmark indices
    MOUTH_LANDMARKS = {
        'upper_lip': [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
        'lower_lip': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 321, 375, 307, 320],
        'mouth_corners': [61, 291]  # Left and right corners
    }
    
    def __init__(self, movement_threshold: int = 5, config: Optional[Dict[str, Any]] = None):
        """
        Initialize mouth movement detector.
        
        Args:
            movement_threshold: Number of consecutive frames with movement to trigger alert
            config: Configuration dictionary (optional)
        """
        if config:
            mouth_config = config.get('detection', {}).get('mouth', {})
            self.movement_threshold = mouth_config.get('movement_threshold', movement_threshold)
        else:
            self.movement_threshold = movement_threshold
        self.mouth_open_history = deque(maxlen=10)
        self.consecutive_movements = 0
        self.last_mouth_state = None
        self.mouth_movement_count = 0
        self.alert_logger = None
        
        # MediaPipe face mesh for landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def calculate_mouth_openness(self, landmarks) -> float:
        """
        Calculate how open the mouth is.
        
        Args:
            landmarks: List of (x, y) landmark coordinates
            
        Returns:
            Mouth openness ratio (0 = closed, >0.1 = open)
        """
        if len(landmarks) < 468:
            return 0.0
        
        # Get key mouth points
        upper_lip_center = landmarks[13]  # Upper lip center
        lower_lip_center = landmarks[14]  # Lower lip center
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        
        # Calculate vertical distance (mouth opening)
        vertical_dist = np.sqrt(
            (upper_lip_center[0] - lower_lip_center[0])**2 +
            (upper_lip_center[1] - lower_lip_center[1])**2
        )
        
        # Calculate horizontal distance (mouth width)
        horizontal_dist = np.sqrt(
            (left_corner[0] - right_corner[0])**2 +
            (left_corner[1] - right_corner[1])**2
        )
        
        # Normalize by mouth width
        if horizontal_dist > 0:
            openness_ratio = vertical_dist / horizontal_dist
        else:
            openness_ratio = 0.0
        
        return openness_ratio
    
    def detect_talking(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect if person is talking based on mouth movements.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Dictionary with talking detection results
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = self.face_mesh.process(rgb_frame)
        
        is_talking = False
        mouth_openness = 0.0
        confidence = 0.0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Convert landmarks to pixel coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            
            # Calculate mouth openness
            mouth_openness = self.calculate_mouth_openness(landmarks)
            self.mouth_open_history.append(mouth_openness)
            
            # Determine if mouth is currently open
            mouth_open = mouth_openness > 0.08  # Threshold for open mouth
            
            # Track consecutive movements
            if mouth_open != self.last_mouth_state:
                self.consecutive_movements += 1
            else:
                self.consecutive_movements = 0
            
            self.last_mouth_state = mouth_open
            
            # Check if talking (mouth opening/closing repeatedly)
            if len(self.mouth_open_history) >= 5:
                # Calculate variance in mouth openness (talking = high variance)
                variance = np.var(list(self.mouth_open_history))
                avg_openness = np.mean(list(self.mouth_open_history))
                
                # Talking if: high variance OR mouth open for several frames
                is_talking = variance > 0.001 or (mouth_open and self.consecutive_movements >= self.movement_threshold)
                confidence = min(1.0, variance * 1000)  # Normalize confidence
        
        return {
            'is_talking': is_talking,
            'mouth_openness': mouth_openness,
            'confidence': confidence,
            'consecutive_movements': self.consecutive_movements
        }
    
    def set_alert_logger(self, alert_logger):
        """Set alert logger for logging alerts."""
        self.alert_logger = alert_logger
    
    def monitor_mouth(self, frame: np.ndarray) -> bool:
        """
        Monitor mouth for movement (simpler interface matching exam version).
        
        Args:
            frame: BGR image frame
            
        Returns:
            True if mouth movement detected, False otherwise
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False
            
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get mouth landmarks (using more points for better accuracy)
        mouth_points = [
            13,  # Upper inner lip
            14,  # Lower inner lip
            78,  # Right corner
            306,  # Left corner
            312,  # Upper outer lip
            317,  # Lower outer lip
        ]
        
        # Calculate mouth openness
        upper_lip = face_landmarks.landmark[13].y
        lower_lip = face_landmarks.landmark[14].y
        mouth_open = lower_lip - upper_lip
        
        # Calculate mouth width
        right_corner = face_landmarks.landmark[78].x
        left_corner = face_landmarks.landmark[306].x
        mouth_width = abs(right_corner - left_corner)
        
        if mouth_open > 0.03 or mouth_width > 0.2:  # Thresholds for mouth movement
            self.mouth_movement_count += 1
            
            if self.mouth_movement_count > self.movement_threshold and self.alert_logger:
                self.alert_logger.log_alert(
                    "MOUTH_MOVEMENT", 
                    "Excessive mouth movement detected (possible talking)"
                )
                self.mouth_movement_count = 0
            return True
        else:
            self.mouth_movement_count = max(0, self.mouth_movement_count - 1)
            return False
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

