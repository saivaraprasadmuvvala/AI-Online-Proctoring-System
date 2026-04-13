"""
Rule-based anomaly detection engine.
Tracks state over time and generates anomaly events.
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime


class AnomalyEngine:
    """Rule-based anomaly detection engine."""
    
    def __init__(self):
        """Initialize anomaly engine with state tracking."""
        self.state = {
            'face_missing_start': None,  # Timestamp when face first went missing
            'gaze_offscreen_start': None,  # Timestamp when gaze first went offscreen
            'last_face_detected': None,  # Timestamp of last face detection
            'last_identity_match': None,  # Timestamp of last identity match
        }
        
        # Thresholds (in seconds) - Made more sensitive
        self.face_missing_threshold = 2.0  # Reduced from 3.0 to 2.0 seconds
        self.gaze_offscreen_threshold = 3.0  # Reduced from 4.0 to 3.0 seconds
    
    def reset_state(self):
        """Reset all state tracking."""
        self.state = {
            'face_missing_start': None,
            'gaze_offscreen_start': None,
            'last_face_detected': None,
            'last_identity_match': None,
        }
    
    def process_frame(self, 
                     detections: List[Dict[str, Any]],
                     recognition_result: Optional[Dict[str, Any]],
                     gaze_result: Optional[Dict[str, Any]],
                     expected_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a frame and detect anomalies.
        
        Args:
            detections: List of face detections from detector
            recognition_result: Recognition result from recognizer
            gaze_result: Gaze estimation result
            expected_name: Expected candidate name for identity verification
            
        Returns:
            List of detected anomaly events (empty if none)
        """
        current_time = time.time()
        anomalies = []
        
        # Update last face detected timestamp
        if detections:
            self.state['last_face_detected'] = current_time
            if self.state['face_missing_start'] is not None:
                # Face was missing but now detected again
                self.state['face_missing_start'] = None
        
        # Check for face missing
        if not detections:
            if self.state['face_missing_start'] is None:
                self.state['face_missing_start'] = current_time
            else:
                duration = current_time - self.state['face_missing_start']
                if duration >= self.face_missing_threshold:
                    # Only trigger once per continuous period
                    if self.state.get('face_missing_triggered') is None:
                        anomalies.append({
                            'event_type': 'face_missing',
                            'details': {
                                'duration_seconds': duration,
                                'threshold': self.face_missing_threshold
                            }
                        })
                        self.state['face_missing_triggered'] = True
        else:
            # Face detected, reset trigger flag
            self.state['face_missing_triggered'] = None
        
        # Check for multiple faces
        if len(detections) > 1:
            anomalies.append({
                'event_type': 'multiple_faces',
                'details': {
                    'face_count': len(detections),
                    'scores': [d.get('score', 0.0) for d in detections]
                }
            })
        
        # Check for gaze offscreen
        if gaze_result is not None:
            is_offscreen = gaze_result.get('is_offscreen', False)
            
            if is_offscreen:
                if self.state['gaze_offscreen_start'] is None:
                    self.state['gaze_offscreen_start'] = current_time
                else:
                    duration = current_time - self.state['gaze_offscreen_start']
                    if duration >= self.gaze_offscreen_threshold:
                        # Only trigger once per continuous period
                        if self.state.get('gaze_offscreen_triggered') is None:
                            anomalies.append({
                                'event_type': 'gaze_offscreen',
                                'details': {
                                    'duration_seconds': duration,
                                    'threshold': self.gaze_offscreen_threshold,
                                    'direction': gaze_result.get('direction', 'unknown'),
                                    'angle': gaze_result.get('angle', 0.0)
                                }
                            })
                            self.state['gaze_offscreen_triggered'] = True
            else:
                # Gaze is on-screen, reset
                self.state['gaze_offscreen_start'] = None
                self.state['gaze_offscreen_triggered'] = None
        
        return anomalies
    
    def process_browser_event(self, event_type: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process browser events (tab switch, window blur, etc.).
        
        Args:
            event_type: Type of browser event ('tab_switch', 'window_blur', 'window_focus')
            details: Optional additional details
            
        Returns:
            Anomaly event dictionary
        """
        if details is None:
            details = {}
        
        # Tab switch and window blur are always anomalies
        if event_type in ['tab_switch', 'window_blur']:
            return {
                'event_type': event_type,
                'details': details
            }
        
        # Window focus is informational, not an anomaly
        if event_type == 'window_focus':
            return {
                'event_type': event_type,
                'details': details
            }
        
        return None
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary for debugging."""
        current_time = time.time()
        summary = {}
        
        if self.state['face_missing_start'] is not None:
            summary['face_missing_duration'] = current_time - self.state['face_missing_start']
        else:
            summary['face_missing_duration'] = 0.0
        
        if self.state['gaze_offscreen_start'] is not None:
            summary['gaze_offscreen_duration'] = current_time - self.state['gaze_offscreen_start']
        else:
            summary['gaze_offscreen_duration'] = 0.0
        
        summary['last_face_detected'] = self.state['last_face_detected']
        summary['last_identity_match'] = self.state['last_identity_match']
        
        return summary
