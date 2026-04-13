"""
Advanced eye tracking module for gaze detection.
Based on exam folder implementation with MediaPipe face mesh.
"""

import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any


class EyeTracker:
    """Advanced eye tracking with gaze direction and blink detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize eye tracker.
        
        Args:
            config: Configuration dictionary (optional, will use defaults if not provided)
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Configuration
        if config:
            eye_config = config.get('detection', {}).get('eyes', {})
            self.gaze_threshold = eye_config.get('gaze_threshold', 2)
            self.blink_threshold = eye_config.get('blink_threshold', 0.3)
            self.gaze_sensitivity = eye_config.get('gaze_sensitivity', 15)
            self.consecutive_frames = eye_config.get('consecutive_frames', 3)
        else:
            self.gaze_threshold = 2
            self.blink_threshold = 0.3
            self.gaze_sensitivity = 15
            self.consecutive_frames = 3
        
        # State tracking
        self.last_gaze_change = datetime.now()
        self.gaze_direction = "center"
        self.eye_ratio = 0.3
        self.gaze_changes = 0
        self.alert_logger = None
        
        # Landmark indices for left and right eyes
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        
        # For EAR (Eye Aspect Ratio) calculation
        self.EYE_ASPECT_RATIO_THRESH = self.blink_threshold
        self.EYE_ASPECT_RATIO_CONSEC_FRAMES = self.consecutive_frames
    
    def set_alert_logger(self, alert_logger):
        """Set alert logger for logging alerts."""
        self.alert_logger = alert_logger
    
    def _calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        Args:
            eye_points: Array of eye landmark coordinates
            
        Returns:
            Eye aspect ratio value
        """
        # Compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def track_eyes(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Track eyes and determine gaze direction with detailed information.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Dictionary with eye tracking results
        """
        try:
            # Convert frame to RGB and process
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return {
                    'gaze_direction': self.gaze_direction,
                    'eye_ratio': self.eye_ratio,
                    'eyes_closed': False,
                    'is_looking_away': False
                }
            
            face_landmarks = results.multi_face_landmarks[0]
            frame_h, frame_w = frame.shape[:2]
            
            # Get eye landmarks in pixel coordinates
            left_eye_coords = np.array([(face_landmarks.landmark[i].x * frame_w, 
                                       face_landmarks.landmark[i].y * frame_h) 
                                      for i in self.LEFT_EYE_INDICES])
            
            right_eye_coords = np.array([(face_landmarks.landmark[i].x * frame_w, 
                                        face_landmarks.landmark[i].y * frame_h) 
                                       for i in self.RIGHT_EYE_INDICES])
            
            # Calculate Eye Aspect Ratio (EAR) for both eyes
            left_ear = self._calculate_ear(left_eye_coords)
            right_ear = self._calculate_ear(right_eye_coords)
            self.eye_ratio = (left_ear + right_ear) / 2.0
            
            # Calculate gaze direction based on eye position
            left_eye_center = np.mean(left_eye_coords, axis=0)
            right_eye_center = np.mean(right_eye_coords, axis=0)
            
            # Calculate horizontal difference between eye centers and nose
            nose_tip = np.array([face_landmarks.landmark[4].x * frame_w,
                                face_landmarks.landmark[4].y * frame_h])
            
            left_diff = left_eye_center[0] - nose_tip[0]
            right_diff = right_eye_center[0] - nose_tip[0]
            horiz_diff = (left_diff + right_diff) / 2.0
            
            # Calculate vertical gaze (up/down)
            eye_center_y = (left_eye_center[1] + right_eye_center[1]) / 2
            nose_y = nose_tip[1]
            vert_diff = eye_center_y - nose_y
            
            # Determine gaze direction (both horizontal and vertical)
            new_gaze = "center"
            if abs(horiz_diff) > abs(vert_diff):
                # Horizontal movement dominates
                if horiz_diff < -self.gaze_sensitivity:
                    new_gaze = "left"
                elif horiz_diff > self.gaze_sensitivity:
                    new_gaze = "right"
            else:
                # Vertical movement dominates
                if vert_diff < -self.gaze_sensitivity:
                    new_gaze = "up"
                elif vert_diff > self.gaze_sensitivity:
                    new_gaze = "down"
            
            # Check if eyes are closed (blink detection)
            eyes_closed = self.eye_ratio < self.EYE_ASPECT_RATIO_THRESH
            
            # Update gaze changes
            current_time = datetime.now()
            if new_gaze != self.gaze_direction:
                self.gaze_changes += 1
                self.gaze_direction = new_gaze
                self.last_gaze_change = current_time
                
            # Check for excessive eye movement
            if (self.gaze_changes > 3 and 
                (current_time - self.last_gaze_change).total_seconds() < self.gaze_threshold and
                self.alert_logger):
                self.alert_logger.log_alert(
                    "EYE_MOVEMENT",
                    "Excessive eye movement detected"
                )
                self.gaze_changes = 0
            
            # Check if eyes are closed for too long
            if eyes_closed and self.alert_logger:
                self.alert_logger.log_alert(
                    "EYES_CLOSED",
                    f"Eyes closed (EAR: {self.eye_ratio:.3f})"
                )
            
            # Determine if looking away from screen
            is_looking_away = new_gaze not in ["center", "down"]  # Down is usually looking at screen
            
            return {
                'gaze_direction': self.gaze_direction,
                'eye_ratio': self.eye_ratio,
                'eyes_closed': eyes_closed,
                'is_looking_away': is_looking_away,
                'gaze_changes': self.gaze_changes
            }
            
        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert(
                    "EYE_TRACKING_ERROR",
                    f"Error in eye tracking: {str(e)}"
                )
            return {
                'gaze_direction': self.gaze_direction,
                'eye_ratio': self.eye_ratio,
                'eyes_closed': False,
                'is_looking_away': False
            }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

