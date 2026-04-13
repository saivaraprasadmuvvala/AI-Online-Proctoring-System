"""
Text-to-speech alert system for real-time verbal warnings.
"""

import time
from typing import Dict, Optional

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    pyttsx3 = None


class AlertSpeaker:
    """Text-to-speech alert system."""
    
    def __init__(self, enabled: bool = True, volume: float = 0.8, cooldown: int = 10):
        """
        Initialize alert speaker.
        
        Args:
            enabled: Enable/disable voice alerts
            volume: Volume level (0.0 to 1.0)
            cooldown: Minimum seconds between same alert
        """
        self.enabled = enabled
        self.volume = volume
        self.cooldown = cooldown
        self.last_alert_time = {}
        
        if TTS_AVAILABLE and enabled:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('volume', volume)
                # Set voice properties
                voices = self.engine.getProperty('voices')
                if voices:
                    self.engine.setProperty('voice', voices[0].id)  # Use first available voice
                self.available = True
                print("✓ Alert speaker initialized")
            except Exception as e:
                print(f"✗ Alert speaker failed: {e}")
                self.available = False
                self.engine = None
        else:
            self.available = False
            self.engine = None
    
    def speak_alert(self, alert_type: str, message: str = None):
        """
        Speak an alert message.
        
        Args:
            alert_type: Type of alert (face_missing, object_detected, etc.)
            message: Custom message (optional)
        """
        if not self.enabled or not self.available or self.engine is None:
            return
        
        # Check cooldown
        current_time = time.time()
        if alert_type in self.last_alert_time:
            if current_time - self.last_alert_time[alert_type] < self.cooldown:
                return  # Still in cooldown
        
        # Generate message if not provided
        if message is None:
            messages = {
                'face_missing': "Warning! Face not detected. Please return to camera view.",
                'multiple_faces': "Warning! Multiple faces detected.",
                'object_detected': "Warning! Prohibited object detected.",
                'talking_detected': "Warning! Talking detected. Please remain silent.",
                'voice_detected': "Warning! Voice activity detected.",
                'gaze_offscreen': "Warning! Please look at the screen.",
                'identity_mismatch': "Warning! Identity verification failed."
            }
            message = messages.get(alert_type, f"Alert: {alert_type}")
        
        try:
            self.engine.say(message)
            self.engine.runAndWait()
            self.last_alert_time[alert_type] = current_time
        except Exception as e:
            print(f"Failed to speak alert: {e}")

