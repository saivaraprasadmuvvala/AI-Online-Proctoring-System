"""
Face pose and head movement detection.
Detects when face is turning away or out of focus using MediaPipe landmarks.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Optional, Tuple
from collections import deque


class FacePoseDetector:
    """Detects face pose, head turning, and focus issues."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize face pose detector.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key landmarks for pose estimation
        # Face outline points
        self.FACE_OUTLINE = [10, 151, 9, 175, 18, 200, 199, 175]
        # Left eye outer corner
        self.LEFT_EYE_OUTER = 33
        # Right eye outer corner  
        self.RIGHT_EYE_OUTER = 263
        # Nose tip
        self.NOSE_TIP = 4
        # Chin
        self.CHIN = 18
        # Forehead center
        self.FOREHEAD = 10
        
        # State tracking
        self.pose_history = deque(maxlen=10)
        self.last_pose = None
        self.alert_logger = None
        
        # Thresholds
        if config:
            pose_config = config.get('detection', {}).get('face_pose', {})
            self.turn_threshold = pose_config.get('turn_threshold', 30)  # degrees
            self.blur_threshold = pose_config.get('blur_threshold', 0.3)  # variance threshold
        else:
            self.turn_threshold = 30
            self.blur_threshold = 0.3
    
    def set_alert_logger(self, alert_logger):
        """Set alert logger."""
        self.alert_logger = alert_logger
    
    def _calculate_head_pose(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Calculate head pose (pitch, yaw, roll) from landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_shape: (height, width) of frame
            
        Returns:
            Dictionary with pose angles and status
        """
        h, w = frame_shape[:2]
        
        # Get key points in pixel coordinates
        left_eye = np.array([
            landmarks.landmark[self.LEFT_EYE_OUTER].x * w,
            landmarks.landmark[self.LEFT_EYE_OUTER].y * h
        ])
        right_eye = np.array([
            landmarks.landmark[self.RIGHT_EYE_OUTER].x * w,
            landmarks.landmark[self.RIGHT_EYE_OUTER].y * h
        ])
        nose_tip = np.array([
            landmarks.landmark[self.NOSE_TIP].x * w,
            landmarks.landmark[self.NOSE_TIP].y * h
        ])
        chin = np.array([
            landmarks.landmark[self.CHIN].x * w,
            landmarks.landmark[self.CHIN].y * h
        ])
        forehead = np.array([
            landmarks.landmark[self.FOREHEAD].x * w,
            landmarks.landmark[self.FOREHEAD].y * h
        ])
        
        # Calculate eye center
        eye_center = (left_eye + right_eye) / 2
        
        # Calculate yaw (left/right turn)
        eye_vector = right_eye - left_eye
        eye_length = np.linalg.norm(eye_vector)
        if eye_length > 0:
            # Normalize
            eye_vector_norm = eye_vector / eye_length
            # Calculate angle from horizontal
            yaw_angle = np.degrees(np.arcsin(eye_vector_norm[1]))
        else:
            yaw_angle = 0
        
        # Calculate pitch (up/down tilt)
        face_center = (eye_center + chin) / 2
        face_vector = chin - eye_center
        face_length = np.linalg.norm(face_vector)
        if face_length > 0:
            face_vector_norm = face_vector / face_length
            pitch_angle = np.degrees(np.arcsin(face_vector_norm[1]))
        else:
            pitch_angle = 0
        
        # Calculate roll (head tilt)
        roll_angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
        
        # Determine if face is turned away
        is_turned = abs(yaw_angle) > self.turn_threshold or abs(pitch_angle) > self.turn_threshold
        
        # Determine direction
        direction = "center"
        if abs(yaw_angle) > abs(pitch_angle):
            if yaw_angle > self.turn_threshold:
                direction = "turned_right"
            elif yaw_angle < -self.turn_threshold:
                direction = "turned_left"
        else:
            if pitch_angle > self.turn_threshold:
                direction = "looking_up"
            elif pitch_angle < -self.turn_threshold:
                direction = "looking_down"
        
        return {
            'yaw': yaw_angle,
            'pitch': pitch_angle,
            'roll': roll_angle,
            'is_turned': is_turned,
            'direction': direction,
            'angles': (yaw_angle, pitch_angle, roll_angle)
        }
    
    def _detect_blur(self, frame: np.ndarray, face_region: Tuple[int, int, int, int]) -> float:
        """
        Detect if face region is blurry/out of focus.
        
        Args:
            frame: Image frame
            face_region: (x, y, width, height) of face region
            
        Returns:
            Blur score (lower = more blurry)
        """
        x, y, w, h = face_region
        h_frame, w_frame = frame.shape[:2]
        
        # Ensure valid region
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        if w < 50 or h < 50:
            return 0.0
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (measure of focus)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return laplacian_var
    
    def detect_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect face pose and movement.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Dictionary with pose detection results
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        result = {
            'face_detected': False,
            'is_turned': False,
            'direction': 'center',
            'is_blurry': False,
            'blur_score': 0.0,
            'angles': (0, 0, 0)
        }
        
        if not results.multi_face_landmarks:
            return result
        
        face_landmarks = results.multi_face_landmarks[0]
        result['face_detected'] = True
        
        # Calculate head pose
        pose = self._calculate_head_pose(face_landmarks, frame.shape[:2])
        result.update(pose)
        
        # Get face bounding box for blur detection
        h, w = frame.shape[:2]
        xs = [lm.x * w for lm in face_landmarks.landmark]
        ys = [lm.y * h for lm in face_landmarks.landmark]
        x = int(min(xs))
        y = int(min(ys))
        face_w = int(max(xs) - min(xs))
        face_h = int(max(ys) - min(ys))
        
        # Detect blur
        blur_score = self._detect_blur(frame, (x, y, face_w, face_h))
        result['blur_score'] = blur_score
        result['is_blurry'] = blur_score < (self.blur_threshold * 1000)  # Normalize threshold
        
        # Store in history
        self.pose_history.append(pose)
        self.last_pose = pose
        
        # Alert if turned away
        if pose['is_turned'] and self.alert_logger:
            self.alert_logger.log_alert(
                "FACE_TURNED",
                f"Face turned {pose['direction']} (yaw: {pose['yaw']:.1f}°, pitch: {pose['pitch']:.1f}°)"
            )
        
        # Alert if blurry
        if result['is_blurry'] and self.alert_logger:
            self.alert_logger.log_alert(
                "FACE_BLURRY",
                f"Face out of focus (blur score: {blur_score:.2f})"
            )
        
        return result
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

