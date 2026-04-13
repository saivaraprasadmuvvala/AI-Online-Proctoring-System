"""
Gaze estimation module using MediaPipe face landmarks.
Detects head direction and off-screen gaze.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple, List


class GazeEstimator:
    """Gaze estimation using MediaPipe face landmarks."""
    
    # MediaPipe face mesh landmark indices
    # Left eye landmarks
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    # Right eye landmarks
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    # Nose tip
    NOSE_TIP = 1
    # Face center (forehead)
    FOREHEAD = 10
    
    def __init__(self):
        """Initialize gaze estimator."""
        pass
    
    def get_eye_centers(self, landmarks: List[Tuple[int, int]], 
                       frame_shape: Tuple[int, int]) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Calculate left and right eye centers from landmarks.
        
        Args:
            landmarks: List of (x, y) landmark coordinates
            frame_shape: (height, width) of frame
            
        Returns:
            (left_eye_center, right_eye_center) tuples
        """
        if len(landmarks) < 468:  # MediaPipe face mesh has 468 landmarks
            return None, None
        
        # Get left eye landmarks
        left_eye_points = [landmarks[i] for i in self.LEFT_EYE_INDICES if i < len(landmarks)]
        right_eye_points = [landmarks[i] for i in self.RIGHT_EYE_INDICES if i < len(landmarks)]
        
        if not left_eye_points or not right_eye_points:
            return None, None
        
        # Calculate centroids
        left_eye_center = (
            int(np.mean([p[0] for p in left_eye_points])),
            int(np.mean([p[1] for p in left_eye_points]))
        )
        right_eye_center = (
            int(np.mean([p[0] for p in right_eye_points])),
            int(np.mean([p[1] for p in right_eye_points]))
        )
        
        return left_eye_center, right_eye_center
    
    def estimate_head_direction(self, landmarks: List[Tuple[int, int]], 
                                frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Estimate head direction based on face landmarks.
        
        Args:
            landmarks: List of (x, y) landmark coordinates
            frame_shape: (height, width) of frame
            
        Returns:
            Dictionary with:
            - direction: str ('left', 'right', 'up', 'down', 'center')
            - angle: float (approximate angle in degrees)
            - is_offscreen: bool
        """
        if len(landmarks) < 468:
            return {
                'direction': 'unknown',
                'angle': 0.0,
                'is_offscreen': False
            }
        
        h, w = frame_shape[:2]
        frame_center_x = w / 2
        frame_center_y = h / 2
        
        # Get key landmarks
        nose_tip = landmarks[self.NOSE_TIP] if self.NOSE_TIP < len(landmarks) else None
        forehead = landmarks[self.FOREHEAD] if self.FOREHEAD < len(landmarks) else None
        left_eye, right_eye = self.get_eye_centers(landmarks, frame_shape)
        
        if not nose_tip or not left_eye or not right_eye:
            return {
                'direction': 'unknown',
                'angle': 0.0,
                'is_offscreen': False
            }
        
        # Calculate eye center (midpoint between eyes)
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        
        # Calculate offsets from frame center
        offset_x = eye_center_x - frame_center_x
        offset_y = eye_center_y - frame_center_y
        
        # Normalize offsets (as percentage of frame size)
        norm_offset_x = offset_x / frame_center_x
        norm_offset_y = offset_y / frame_center_y
        
        # Determine direction
        threshold = 0.15  # 15% threshold for direction change
        
        if abs(norm_offset_x) > abs(norm_offset_y):
            if norm_offset_x > threshold:
                direction = 'right'
            elif norm_offset_x < -threshold:
                direction = 'left'
            else:
                direction = 'center'
        else:
            if norm_offset_y > threshold:
                direction = 'down'
            elif norm_offset_y < -threshold:
                direction = 'up'
            else:
                direction = 'center'
        
        # Calculate approximate angle
        angle = np.arctan2(offset_y, offset_x) * 180 / np.pi
        
        # Determine if off-screen (looking away significantly)
        # Consider off-screen if looking left/right/up/down beyond threshold
        is_offscreen = direction in ['left', 'right', 'up', 'down']
        
        return {
            'direction': direction,
            'angle': float(angle),
            'is_offscreen': is_offscreen,
            'offset_x': float(norm_offset_x),
            'offset_y': float(norm_offset_y)
        }
    
    def estimate_gaze(self, landmarks: List[Tuple[int, int]], 
                     frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Main gaze estimation function.
        
        Args:
            landmarks: List of (x, y) landmark coordinates
            frame_shape: (height, width) of frame
            
        Returns:
            Dictionary with gaze information
        """
        head_info = self.estimate_head_direction(landmarks, frame_shape)
        left_eye, right_eye = self.get_eye_centers(landmarks, frame_shape)
        
        return {
            **head_info,
            'left_eye_center': left_eye,
            'right_eye_center': right_eye
        }
    
    def is_looking_away(self, gaze_info: Dict[str, Any], 
                       duration_seconds: float, 
                       threshold_seconds: float = 4.0) -> bool:
        """
        Determine if user has been looking away for too long.
        
        Args:
            gaze_info: Gaze estimation result
            duration_seconds: How long the current gaze state has persisted
            threshold_seconds: Threshold in seconds (default 4.0)
            
        Returns:
            True if looking away for more than threshold
        """
        if gaze_info.get('is_offscreen', False):
            return duration_seconds > threshold_seconds
        return False
