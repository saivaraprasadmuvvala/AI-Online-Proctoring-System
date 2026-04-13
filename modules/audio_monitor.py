"""
Audio monitoring module to detect voice/whispering.
Uses PyAudio for audio capture and analysis.
"""

import numpy as np
from typing import Dict, Any, Optional
from collections import deque
import time

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except (ImportError, OSError):
    PYAUDIO_AVAILABLE = False
    pyaudio = None
    print("⚠️ PyAudio not available - audio monitoring disabled. Install with: brew install portaudio && pip install pyaudio")


class AudioMonitor:
    """Monitors audio for voice/whispering detection."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 energy_threshold: float = 0.001,
                 zcr_threshold: float = 0.35,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize audio monitor.
        
        Args:
            sample_rate: Audio sample rate
            chunk_size: Audio chunk size
            energy_threshold: Energy threshold for voice detection
            zcr_threshold: Zero crossing rate threshold
            config: Configuration dictionary (optional)
        """
        if config:
            audio_config = config.get('detection', {}).get('audio_monitoring', {})
            self.sample_rate = audio_config.get('sample_rate', sample_rate)
            self.energy_threshold = audio_config.get('energy_threshold', energy_threshold)
            self.zcr_threshold = audio_config.get('zcr_threshold', zcr_threshold)
            self.whisper_enabled = audio_config.get('whisper_enabled', False)
            self.whisper_model_name = audio_config.get('whisper_model', 'tiny.en')
        else:
            self.sample_rate = sample_rate
            self.energy_threshold = energy_threshold
            self.zcr_threshold = zcr_threshold
            self.whisper_enabled = False
            self.whisper_model_name = 'tiny.en'
        
        self.chunk_size = chunk_size
        
        self.audio_stream = None
        self.audio_buffer = deque(maxlen=10)
        self.is_monitoring = False
        self.alert_system = None
        self.alert_logger = None
        self.whisper_model = None
        
        # Load whisper model if enabled
        if self.whisper_enabled:
            try:
                import whisper
                self.whisper_model = whisper.load_model(self.whisper_model_name)
            except ImportError:
                print("⚠️ Whisper not available - install with: pip install openai-whisper")
                self.whisper_enabled = False
        
        if PYAUDIO_AVAILABLE:
            try:
                self.audio = pyaudio.PyAudio()
                self.available = True
                print("✓ Audio monitor initialized")
            except Exception as e:
                print(f"✗ Audio monitor failed: {e}")
                self.available = False
                self.audio = None
        else:
            self.available = False
            self.audio = None
    
    def start_monitoring(self):
        """Start audio monitoring."""
        if not self.available or self.audio is None:
            return False
        
        try:
            self.audio_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self.is_monitoring = True
            return True
        except Exception as e:
            print(f"Failed to start audio monitoring: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop audio monitoring."""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        self.is_monitoring = False
    
    def calculate_energy(self, audio_data: np.ndarray) -> float:
        """Calculate audio energy."""
        return np.mean(audio_data ** 2)
    
    def calculate_zcr(self, audio_data: np.ndarray) -> float:
        """Calculate zero crossing rate (indicates voice)."""
        signs = np.sign(audio_data)
        zcr = np.mean(np.abs(np.diff(signs))) / 2.0
        return zcr
    
    def detect_voice(self) -> Dict[str, Any]:
        """
        Detect voice/whispering in audio stream.
        
        Returns:
            Dictionary with voice detection results
        """
        if not self.is_monitoring or self.audio_stream is None:
            return {
                'voice_detected': False,
                'energy': 0.0,
                'zcr': 0.0,
                'confidence': 0.0
            }
        
        try:
            # Read audio chunk
            audio_bytes = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate features
            energy = self.calculate_energy(audio_data)
            zcr = self.calculate_zcr(audio_data)
            
            # Determine if voice is detected
            voice_detected = energy > self.energy_threshold and zcr > self.zcr_threshold
            
            # Calculate confidence
            energy_confidence = min(1.0, energy / (self.energy_threshold * 10))
            zcr_confidence = min(1.0, zcr / (self.zcr_threshold * 2))
            confidence = (energy_confidence + zcr_confidence) / 2.0
            
            self.audio_buffer.append({
                'energy': energy,
                'zcr': zcr,
                'voice_detected': voice_detected,
                'timestamp': time.time()
            })
            
            return {
                'voice_detected': voice_detected,
                'energy': energy,
                'zcr': zcr,
                'confidence': confidence
            }
        
        except Exception as e:
            return {
                'voice_detected': False,
                'energy': 0.0,
                'zcr': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def set_alert_system(self, alert_system):
        """Set alert system for voice alerts."""
        self.alert_system = alert_system
    
    def set_alert_logger(self, alert_logger):
        """Set alert logger for logging alerts."""
        self.alert_logger = alert_logger
    
    def _is_voice(self, audio_data: np.ndarray) -> bool:
        """
        Ultra-fast voice detection using energy and ZCR.
        
        Args:
            audio_data: Audio data array
            
        Returns:
            True if voice detected, False otherwise
        """
        audio_norm = audio_data / 32768.0
        
        # 1. Energy detection
        energy = np.mean(audio_norm**2)
        if energy < self.energy_threshold:
            return False
            
        # 2. Zero-crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio_norm))))
        if zcr > self.zcr_threshold:
            return False
            
        return True
    
    def _handle_voice_detection(self):
        """Process detected voice and trigger alerts."""
        if self.alert_system:
            self.alert_system.speak_alert("VOICE_DETECTED")
            
        if self.alert_logger:
            self.alert_logger.log_alert("VOICE_DETECTED", "Voice activity detected")
            
        if self.whisper_enabled and self.whisper_model:
            self._process_with_whisper()
    
    def _process_with_whisper(self):
        """Optional Whisper processing for speech recognition."""
        try:
            import whisper
            audio = np.concatenate(list(self.audio_buffer))
            result = self.whisper_model.transcribe(
                audio.astype(np.float32) / 32768.0,
                fp16=False,
                language='en'
            )
            
            text = result['text'].strip().lower()
            if any(word in text for word in ['help', 'answer', 'whisper']):
                if self.alert_system:
                    self.alert_system.speak_alert("SPEECH_VIOLATION")
                    
        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert("WHISPER_ERROR", str(e))
    
    def __del__(self):
        """Cleanup resources."""
        self.stop_monitoring()
        if self.audio and hasattr(self.audio, 'terminate'):
            self.audio.terminate()

