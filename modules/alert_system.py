"""
Alert system with text-to-speech capabilities.
Provides voice alerts for violations and suspicious activities.
"""

import os
import tempfile
import threading
import time
from typing import Optional, Dict, Any

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
    PYGAME_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    PYGAME_AVAILABLE = False
    gTTS = None
    pygame = None
    print("⚠️ gTTS or pygame not available - voice alerts disabled. Install with: pip install gTTS pygame")


class AlertSystem:
    """Text-to-speech alert system for proctoring violations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize alert system.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        alert_config = self.config.get('logging', {}).get('alert_system', {})
        self.alert_cooldown = alert_config.get('cooldown', 10)
        self.voice_alerts = alert_config.get('voice_alerts', True)
        self.alert_volume = alert_config.get('alert_volume', 0.8)
        
        self.last_alert_time = {}
        self.available = GTTS_AVAILABLE and PYGAME_AVAILABLE
        
        if self.available:
            try:
                pygame.mixer.init()
            except Exception as e:
                print(f"⚠️ Failed to initialize pygame mixer: {e}")
                self.available = False
        
        # Alert messages database
        self.alerts = {
            "FACE_DISAPPEARED": "Please look at the screen",
            "FACE_REAPPEARED": "Thank you for looking at the screen",
            "MULTIPLE_FACES": "We detected multiple people",
            "OBJECT_DETECTED": "Unauthorized object detected",
            "GAZE_AWAY": "Please focus on your screen",
            "EYES_CLOSED": "Please keep your eyes open",
            "FACE_TURNED": "Please face the camera",
            "FACE_BLURRY": "Please adjust your position for better focus",
            "MOUTH_MOVING": "Please maintain silence during exam",
            "MOUTH_MOVEMENT": "Please maintain silence during exam",
            "SPEECH_VIOLATION": "Speaking during exam is not allowed",
            "VOICE_DETECTED": "We detected voice, Please maintain silence during the exam",
            "TALKING_DETECTED": "Talking detected, Please maintain silence",
            "EYE_MOVEMENT": "Please focus on your screen",
        }
    
    def _can_alert(self, alert_type: str) -> bool:
        """
        Check if enough time has passed since last alert.
        
        Args:
            alert_type: Type of alert
            
        Returns:
            True if alert can be triggered, False otherwise
        """
        current_time = time.time()
        last_time = self.last_alert_time.get(alert_type, 0)
        return (current_time - last_time) >= self.alert_cooldown
    
    def speak_alert(self, alert_type: str):
        """
        Convert text to speech and play it.
        
        Args:
            alert_type: Type of alert to speak
        """
        if not self.available or not self.voice_alerts:
            return
        
        if not self._can_alert(alert_type):
            return
        
        self.last_alert_time[alert_type] = time.time()
        
        if alert_type not in self.alerts:
            return
        
        def _play_audio():
            try:
                # Generate speech
                tts = gTTS(text=self.alerts[alert_type], lang='en')
                
                # Save temporary audio file
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
                    temp_path = fp.name
                    tts.save(temp_path)
                
                # Play audio
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.set_volume(self.alert_volume)
                pygame.mixer.music.play()
                
                # Wait until playback finishes
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Cleanup
                os.unlink(temp_path)
            except Exception as e:
                print(f"Audio alert failed: {str(e)}")
        
        # Run in separate thread to avoid blocking
        threading.Thread(target=_play_audio, daemon=True).start()
    
    def add_alert(self, alert_type: str, message: str):
        """
        Add or update an alert message.
        
        Args:
            alert_type: Type of alert
            message: Alert message text
        """
        self.alerts[alert_type] = message

