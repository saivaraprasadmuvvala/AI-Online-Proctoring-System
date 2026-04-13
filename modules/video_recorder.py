"""
Video recorder for webcam recording.
Records video sessions with configurable settings.
"""

import cv2
import os
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np


class VideoRecorder:
    """Records video from webcam."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize video recorder.
        
        Args:
            config: Configuration dictionary (optional)
        """
        if config:
            video_config = config.get('video', {})
            self.recording_path = video_config.get('recording_path', './recordings')
            self.resolution = tuple(video_config.get('resolution', [1280, 720]))
            self.fps = video_config.get('fps', 30)
        else:
            self.recording_path = './recordings'
            self.resolution = (1280, 720)
            self.fps = 30
        
        self.writer = None
        self.filename = None
        self.frame_count = 0
        self.start_time = datetime.now()
    
    def start_recording(self):
        """Start video recording."""
        if not os.path.exists(self.recording_path):
            os.makedirs(self.recording_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.recording_path, f"webcam_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.filename,
            fourcc,
            self.fps,
            self.resolution
        )
        self.frame_count = 0
        self.start_time = datetime.now()
    
    def record_frame(self, frame: np.ndarray):
        """
        Record a frame.
        
        Args:
            frame: Frame to record
        """
        if self.writer:
            # Resize frame if needed
            if frame.shape[:2][::-1] != self.resolution:
                frame = cv2.resize(frame, self.resolution)
            self.writer.write(frame)
            self.frame_count += 1
    
    def stop_recording(self) -> Optional[Dict[str, Any]]:
        """
        Stop recording and return recording info.
        
        Returns:
            Dictionary with recording information or None
        """
        if self.writer:
            self.writer.release()
            self.writer = None
            duration = (datetime.now() - self.start_time).total_seconds()
            return {
                'filename': self.filename,
                'frame_count': self.frame_count,
                'duration': duration,
                'fps': self.frame_count / duration if duration > 0 else 0
            }
        return None

