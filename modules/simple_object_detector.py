"""
Lightweight object detection using MediaPipe and simple computer vision.
Detects common prohibited objects like phones, books, papers.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Optional
from collections import deque


class SimpleObjectDetector:
    """Lightweight object detector using color/shape detection and MediaPipe."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize simple object detector.
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Use MediaPipe for hand detection (can indicate phone holding)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Configuration
        if config:
            obj_config = config.get('detection', {}).get('objects', {})
            self.min_confidence = obj_config.get('min_confidence', 0.5)
        else:
            self.min_confidence = 0.5
        
        self.detection_history = deque(maxlen=10)
        self.alert_logger = None
    
    def set_alert_logger(self, alert_logger):
        """Set alert logger."""
        self.alert_logger = alert_logger
    
    def _detect_rectangular_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect rectangular objects (books, papers, phones) using contour detection.
        
        Args:
            frame: BGR image frame
            
        Returns:
            List of detected objects
        """
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = frame.shape[:2]
        min_area = (w * h) * 0.01  # At least 1% of frame
        max_area = (w * h) * 0.3   # At most 30% of frame
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 corners)
            if len(approx) >= 4:
                # Get bounding box
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Calculate aspect ratio
                aspect_ratio = w_box / h_box if h_box > 0 else 0
                
                # Check if it looks like a phone (long and thin) or book/paper (rectangular)
                if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio
                    # Calculate rectangularity (how close to rectangle)
                    rect_area = w_box * h_box
                    extent = area / rect_area if rect_area > 0 else 0
                    
                    if extent > 0.7:  # At least 70% rectangular
                        # Determine object type based on size and aspect ratio
                        obj_type = "unknown"
                        if aspect_ratio > 1.5:
                            obj_type = "phone"  # Long and thin
                        elif 0.8 < aspect_ratio < 1.2:
                            obj_type = "book"  # Square-ish
                        else:
                            obj_type = "paper"  # Rectangular
                        
                        detections.append({
                            'bbox': (x, y, w_box, h_box),
                            'class_name': obj_type,
                            'confidence': extent,  # Use extent as confidence
                            'area': area
                        })
        
        return detections
    
    def _detect_hands_near_face(self, frame: np.ndarray, face_bbox: Optional[tuple]) -> bool:
        """
        Detect if hands are near face (might indicate phone or object).
        
        Args:
            frame: BGR image frame
            face_bbox: Face bounding box (x, y, w, h) or None
            
        Returns:
            True if hands detected near face
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks or not face_bbox:
            return False
        
        fx, fy, fw, fh = face_bbox
        face_center_x = fx + fw / 2
        face_center_y = fy + fh / 2
        
        h, w = frame.shape[:2]
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand center (wrist)
            wrist = hand_landmarks.landmark[0]
            hand_x = wrist.x * w
            hand_y = wrist.y * h
            
            # Calculate distance from face center
            distance = np.sqrt((hand_x - face_center_x)**2 + (hand_y - face_center_y)**2)
            
            # If hand is close to face (within 1.5x face width)
            if distance < fw * 1.5:
                return True
        
        return False
    
    def detect_objects(self, frame: np.ndarray, face_bbox: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Detect prohibited objects in frame.
        
        Args:
            frame: BGR image frame
            face_bbox: Optional face bounding box for context
            
        Returns:
            List of detected objects
        """
        detections = []
        
        # Method 1: Detect rectangular objects
        rectangular_objs = self._detect_rectangular_objects(frame)
        detections.extend(rectangular_objs)
        
        # Method 2: Check for hands near face (might indicate phone)
        if face_bbox:
            hands_near_face = self._detect_hands_near_face(frame, face_bbox)
            if hands_near_face:
                # Add a generic "object" detection
                detections.append({
                    'bbox': face_bbox,
                    'class_name': 'suspicious_object',
                    'confidence': 0.6,
                    'area': face_bbox[2] * face_bbox[3]
                })
        
        # Filter by confidence
        filtered = [d for d in detections if d['confidence'] >= self.min_confidence]
        
        # Store in history
        if filtered:
            self.detection_history.append(filtered)
        
        # Alert if objects detected
        if filtered and self.alert_logger:
            for obj in filtered:
                self.alert_logger.log_alert(
                    "OBJECT_DETECTED",
                    f"Detected {obj['class_name']} (confidence: {obj['confidence']:.2f})"
                )
        
        return filtered
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detected objects on frame.
        
        Args:
            frame: BGR image frame
            detections: List of detected objects
            
        Returns:
            Frame with drawn detections
        """
        for obj in detections:
            x, y, w, h = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x, y - text_height - 10),
                        (x + text_width, y), (0, 0, 255), -1)
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'hands'):
            self.hands.close()

