"""
Screen recording module to capture examinee's screen activity.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
import time
import os
from datetime import datetime
import threading

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    mss = None


class ScreenRecorder:
    """Records screen activity during exam."""
    
    def __init__(self, monitor_index: int = 0, fps: int = 5, config: Optional[Dict[str, Any]] = None):
        """
        Initialize screen recorder.
        
        Args:
            monitor_index: Monitor index (0 = primary)
            fps: Frames per second for recording
            config: Configuration dictionary (optional)
        """
        if config:
            screen_config = config.get('screen', {})
            self.monitor_index = screen_config.get('monitor_index', monitor_index)
            self.fps = screen_config.get('fps', fps)
            self.recording_enabled = screen_config.get('recording', True)
        else:
            self.monitor_index = monitor_index
            self.fps = fps
            self.recording_enabled = True
        self.frame_interval = 1.0 / self.fps
        self.last_capture_time = 0
        self.recording = False
        self.video_writer = None
        self.output_path = None
        self.frame_count = 0
        self.start_time = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.monitor = None
        
        if MSS_AVAILABLE:
            try:
                self.sct = mss.mss()
                self.available = True
                print("✓ Screen recorder initialized")
            except Exception as e:
                print(f"✗ Screen recorder failed: {e}")
                self.available = False
                self.sct = None
        else:
            self.available = False
            self.sct = None
    
    def _initialize_sct(self):
        """Initialize MSS in the thread where it will be used."""
        if self.sct:
            monitors = self.sct.monitors
            if len(monitors) > self.monitor_index + 1:  # +1 because monitor 0 is all screens
                self.monitor = monitors[self.monitor_index + 1]
            else:
                self.monitor = monitors[1]  # Default to first monitor
    
    def start_recording(self, output_path: Optional[str] = None):
        """
        Start screen recording.
        
        Args:
            output_path: Optional path to save recording (auto-generated if None)
        """
        if not self.available or not self.recording_enabled:
            return False
        
        try:
            if not output_path:
                # Auto-generate output path
                recording_path = self.config.get('video', {}).get('recording_path', './recordings') if hasattr(self, 'config') else './recordings'
                os.makedirs(recording_path, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(recording_path, f"screen_{timestamp}.mp4")
            
            self._initialize_sct()
            if not self.monitor:
                return False
            
            width = self.monitor['width']
            height = self.monitor['height']
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.fps, (width, height)
            )
            self.output_path = output_path
            self.recording = True
            self.frame_count = 0
            self.start_time = datetime.now()
            
            # Start capture thread
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Failed to start screen recording: {e}")
            return False
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while not self.stop_event.is_set():
            with self.lock:
                if self.sct and self.monitor:
                    screenshot = self.sct.grab(self.monitor)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    if self.video_writer:
                        self.video_writer.write(frame)
                        self.frame_count += 1
            
            # Control capture rate
            time.sleep(1.0 / self.fps)
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single screen frame.
        
        Returns:
            BGR image frame or None
        """
        if not self.available or not self.recording:
            return None
        
        current_time = time.time()
        if current_time - self.last_capture_time < self.frame_interval:
            return None
        
        try:
            monitor = self.sct.monitors[self.monitor_index]
            screenshot = self.sct.grab(monitor)
            
            # Convert to numpy array
            img = np.array(screenshot)
            # Convert BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Write to video if recording
            if self.video_writer:
                self.video_writer.write(img)
            
            self.last_capture_time = current_time
            return img
        
        except Exception as e:
            return None
    
    def stop_recording(self) -> Optional[Dict[str, Any]]:
        """
        Stop recording and return recording info.
        
        Returns:
            Dictionary with recording information or None
        """
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.recording = False
        
        if self.output_path and self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            return {
                'filename': self.output_path,
                'frame_count': self.frame_count,
                'duration': duration
            }
        return None
    
    def __del__(self):
        """Cleanup resources."""
        self.stop_recording()

