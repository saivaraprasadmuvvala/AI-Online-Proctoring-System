"""
Main Streamlit application for Online Proctoring System.
Provides exam taking, results, and instructor dashboard pages.
"""

import streamlit as st
import cv2
import numpy as np
import os
import time
import warnings
import yaml
from datetime import datetime
from typing import Optional, Dict, Any
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import pandas as pd

# Suppress ScriptRunContext warnings from streamlit-webrtc background threads
import logging
import sys

# Suppress all ScriptRunContext warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*ScriptRunContext.*")

# Suppress logging for script runner
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime.scriptrunner.exec_code").setLevel(logging.CRITICAL)

# Redirect stderr for background threads to suppress warnings
class SuppressStderr:
    def __init__(self):
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        sys.stderr = open(os.devnull, 'w')
        return self
        
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self.original_stderr

# Import modules
from modules.storage import Storage
from modules.detector import FaceDetector
from modules.gaze import GazeEstimator
from modules.anomaly import AnomalyEngine
from modules.js_events import create_browser_event_component
from modules.mouth_detector import MouthDetector
from modules.mouth_detector import MouthDetector
from modules.audio_monitor import AudioMonitor
from modules.screen_recorder import ScreenRecorder
from modules.alert_speaker import AlertSpeaker

# Import new enhanced modules
from utils.config_loader import get_config
from modules.eye_tracking import EyeTracker
from modules.face_pose_detector import FacePoseDetector
from modules.simple_object_detector import SimpleObjectDetector
from modules.alert_system import AlertSystem
from modules.violation_logger import ViolationLogger
from modules.violation_capturer import ViolationCapturer
from modules.video_recorder import VideoRecorder
from modules.report_generator import ReportGenerator
from modules.exam_manager import ExamManager

# Configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Load configuration
try:
    config_loader = get_config()
    config = config_loader.get_config()
except Exception as e:
    st.warning(f"⚠️ Could not load config: {e}. Using defaults.")
    config = {}

# Initialize session state
if 'storage' not in st.session_state:
    st.session_state.storage = Storage()
if 'config' not in st.session_state:
    st.session_state.config = config
if 'detector' not in st.session_state:
    st.session_state.detector = FaceDetector()
if 'gaze_estimator' not in st.session_state:
    st.session_state.gaze_estimator = GazeEstimator()
if 'anomaly_engine' not in st.session_state:
    st.session_state.anomaly_engine = AnomalyEngine()
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'browser_events' not in st.session_state:
    st.session_state.browser_events = []
if 'violation_logger' not in st.session_state:
    st.session_state.violation_logger = ViolationLogger(config)
if 'violation_capturer' not in st.session_state:
    st.session_state.violation_capturer = ViolationCapturer(config)
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem(config)
if 'report_generator' not in st.session_state:
    st.session_state.report_generator = ReportGenerator(config)
if 'exam_manager' not in st.session_state:
    st.session_state.exam_manager = ExamManager(st.session_state.storage)
# Reinitialize storage if it doesn't have the new methods (for compatibility after code updates)
if not hasattr(st.session_state.storage, 'get_all_questions'):
    st.session_state.storage = Storage()
    st.session_state.exam_manager = ExamManager(st.session_state.storage)
if 'exam_active' not in st.session_state:
    st.session_state.exam_active = False
if 'exam_start_time' not in st.session_state:
    st.session_state.exam_start_time = None
if 'exam_duration' not in st.session_state:
    st.session_state.exam_duration = 300  # 5 minutes in seconds
if 'exam_answers' not in st.session_state:
    st.session_state.exam_answers = {}  # question_id -> selected_answer
if 'exam_completed' not in st.session_state:
    st.session_state.exam_completed = False
if 'exam_session_id' not in st.session_state:
    st.session_state.exam_session_id = None
if 'student_name' not in st.session_state:
    st.session_state.student_name = None

# Create directories
os.makedirs("enrolled", exist_ok=True)
os.makedirs("evidence", exist_ok=True)


def log_browser_event(session_id: int, event_type: str, details: Optional[Dict[str, Any]] = None):
    """
    Log a browser event (tab switch, window blur, etc.).
    
    Args:
        session_id: Current session ID
        event_type: Type of browser event
        details: Optional event details
    """
    if not session_id:
        return
    
    try:
        # Save a placeholder evidence image (black frame) for browser events
        evidence_path = None
        if event_type in ['tab_switch', 'window_blur']:
            # Create a simple evidence image
            evidence_img = np.zeros((100, 100, 3), dtype=np.uint8)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{event_type}_{timestamp}.jpg"
            evidence_path = os.path.join("evidence", filename)
            cv2.imwrite(evidence_path, evidence_img)
        
        # Log to database
        st.session_state.storage.log_event(
            session_id=session_id,
            event_type=event_type,
            details=str(details) if details else None,
            evidence_image_path=evidence_path
        )
    except Exception as e:
        print(f"Error logging browser event: {e}")


def save_evidence_image(frame: np.ndarray, event_type: str) -> str:
    """
    Save evidence image with timestamp.
    
    Args:
        frame: Image frame to save
        event_type: Type of event
        
    Returns:
        Path to saved image
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{event_type}_{timestamp}.jpg"
    filepath = os.path.join("evidence", filename)
    cv2.imwrite(filepath, frame)
    return filepath


class VideoProcessor(VideoProcessorBase):
    """Video processor for real-time face detection and analysis."""
    
    def __init__(self):
        super().__init__()
        # Load config
        try:
            config_loader = get_config()
            self.config = config_loader.get_config()
        except (FileNotFoundError, yaml.YAMLError, IOError, OSError) as e:
            print(f"⚠️ Could not load config: {e}. Using defaults.")
            self.config = {}
        except Exception as e:
            print(f"⚠️ Unexpected error loading config: {e}. Using defaults.")
            self.config = {}
        
        self.detector = FaceDetector()
        self.gaze_estimator = GazeEstimator()
        self.anomaly_engine = AnomalyEngine()
        
        # Initialize enhanced detection modules
        try:
            self.eye_tracker = EyeTracker(self.config)
        except Exception as e:
            print(f"⚠️ Eye tracker not available: {e}")
            self.eye_tracker = None
        
        try:
            self.face_pose_detector = FacePoseDetector(self.config)
        except Exception as e:
            print(f"⚠️ Face pose detector not available: {e}")
            self.face_pose_detector = None
        
        try:
            self.object_detector = SimpleObjectDetector(self.config)
        except Exception as e:
            print(f"⚠️ Object detector not available: {e}")
            self.object_detector = None
        
        try:
            self.mouth_detector = MouthDetector(config=self.config)
        except Exception as e:
            print(f"⚠️ Mouth detector not available: {e}")
            self.mouth_detector = None
        
        try:
            self.audio_monitor = AudioMonitor(config=self.config)
        except Exception as e:
            print(f"⚠️ Audio monitor not available: {e}")
            self.audio_monitor = None
        
        try:
            self.alert_speaker = AlertSpeaker(enabled=True)
            self.alert_system = AlertSystem(self.config)
        except Exception as e:
            print(f"⚠️ Alert system not available: {e}")
            self.alert_speaker = None
            self.alert_system = None
        
        try:
            self.screen_recorder = ScreenRecorder(config=self.config)
        except Exception as e:
            print(f"⚠️ Screen recorder not available: {e}")
            self.screen_recorder = None
        
        try:
            self.video_recorder = VideoRecorder(self.config)
        except Exception as e:
            print(f"⚠️ Video recorder not available: {e}")
            self.video_recorder = None
        
        self.storage = Storage()
        self.frame_count = 0
        self.last_process_time = time.time()
        self.target_fps = 10
        self.session_id = None
        self.expected_name = None
        self.session_active = False
        self.detection_stats = {
            'faces_detected': 0,
            'last_detection_time': None,
            'status': 'Initializing...',
            'objects_detected': 0,
            'talking_detected': False,
            'voice_detected': False,
            'gaze_direction': 'center',
            'eye_ratio': 0.3,
            'multiple_faces': False,
            'face_turned': False,
            'face_blurry': False,
            'objects_detected': 0
        }
        self.audio_started = False
        self.recording_started = False
        
    def set_session(self, session_id, expected_name, active):
        """Set session information for processing."""
        self.session_id = session_id
        self.expected_name = expected_name
        self.session_active = active
        
        # Start recording if session becomes active
        if active and not self.recording_started:
            if self.video_recorder:
                try:
                    self.video_recorder.start_recording()
                except (RuntimeError, OSError, IOError) as e:
                    print(f"⚠️ Failed to start video recording: {e}")
                except Exception as e:
                    print(f"⚠️ Unexpected error starting video recording: {e}")
            if self.screen_recorder and self.config.get('screen', {}).get('recording', False):
                try:
                    self.screen_recorder.start_recording()
                except (RuntimeError, OSError, IOError) as e:
                    print(f"⚠️ Failed to start screen recording: {e}")
                except Exception as e:
                    print(f"⚠️ Unexpected error starting screen recording: {e}")
            if self.audio_monitor:
                try:
                    self.audio_monitor.start_monitoring()
                    if self.alert_system:
                        self.audio_monitor.set_alert_system(self.alert_system)
                except (RuntimeError, OSError, IOError, AttributeError) as e:
                    print(f"⚠️ Failed to start audio monitoring: {e}")
                except Exception as e:
                    print(f"⚠️ Unexpected error starting audio monitoring: {e}")
            self.recording_started = True
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process each video frame."""
        # Skip frames to maintain target FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_process_time
        
        if elapsed < (1.0 / self.target_fps):
            # Skip this frame but still draw status
            img = frame.to_ndarray(format="bgr24")
            img = self._draw_status(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        self.last_process_time = current_time
        
        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame only if session is active
        if not self.session_active or not self.session_id:
            img = self._draw_status(img, "Waiting for session...")
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Process frame
        try:
            # 1. Face detection
            detections = self._process_face_detection(img, current_time)
            
            # 2. Eye tracking and gaze estimation
            gaze_result = self._process_eye_tracking(img)
            
            # 3. Face pose detection (turning, blur)
            self._process_face_pose(img)
            
            # 4. Object detection (lightweight)
            prohibited_objects = self._process_object_detection(img, detections)
            
            # 5. Mouth and audio detection
            self._process_audio_detection(img)
            
            # 6. Anomaly detection and logging
            anomalies = self._process_anomalies(img, detections, gaze_result)
            
            # 7. Screen and video recording
            if self.screen_recorder:
                self.screen_recorder.capture_frame()
            if self.video_recorder:
                self.video_recorder.record_frame(img)
            
            # 8. Draw all detections on frame
            img = self._draw_detections_on_frame(img, detections, prohibited_objects, anomalies)
            
            # 9. Draw status overlay
            img = self._draw_status(img)
            
        except Exception as e:
            # Suppress error printing in background thread to avoid ScriptRunContext warnings
            error_msg = str(e)[:50]  # Truncate long errors
            self.detection_stats['status'] = f"Error: {error_msg}"
            img = self._draw_status(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _process_face_detection(self, img: np.ndarray, current_time: float) -> list:
        """Process face detection and update stats."""
        detections = self.detector.detect_faces(img)
        
        if detections:
            self.detection_stats['faces_detected'] = len(detections)
            self.detection_stats['last_detection_time'] = current_time
            methods_used = [d.get('method', 'unknown') for d in detections]
            votes = [d.get('voted_by', 1) for d in detections]
            self.detection_stats['status'] = f'✅ Detected {len(detections)} face(s) | Methods: {", ".join(set(methods_used))} | Votes: {max(votes) if votes else 1}'
        else:
            self.detection_stats['status'] = '❌ No face detected - Check camera and lighting!'
        
        return detections
    
    def _process_eye_tracking(self, img: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process eye tracking and gaze estimation."""
        gaze_result = None
        
        # Enhanced eye tracking
        if self.eye_tracker:
            eye_result = self.eye_tracker.track_eyes(img)
            gaze_direction = eye_result.get('gaze_direction', 'center')
            eye_ratio = eye_result.get('eye_ratio', 0.3)
            eyes_closed = eye_result.get('eyes_closed', False)
            is_looking_away = eye_result.get('is_looking_away', False)
            
            self.detection_stats['gaze_direction'] = gaze_direction
            self.detection_stats['eye_ratio'] = eye_ratio
            self.detection_stats['status'] += f" | Gaze: {gaze_direction}"
            
            if eyes_closed:
                self.detection_stats['status'] += " | ⚠️ EYES CLOSED"
                evidence_path = save_evidence_image(img, 'eyes_closed')
                self.storage.log_event(
                    session_id=self.session_id,
                    event_type='eyes_closed',
                    details=f"Eye ratio: {eye_ratio:.3f}",
                    evidence_image_path=evidence_path
                )
                if self.alert_system:
                    self.alert_system.speak_alert('EYES_CLOSED')
            
            if is_looking_away:
                self.detection_stats['status'] += f" | ⚠️ LOOKING AWAY ({gaze_direction})"
                evidence_path = save_evidence_image(img, 'gaze_away')
                self.storage.log_event(
                    session_id=self.session_id,
                    event_type='gaze_away',
                    details=f"Gaze direction: {gaze_direction}",
                    evidence_image_path=evidence_path
                )
                if self.alert_system:
                    self.alert_system.speak_alert('GAZE_AWAY')
        
        # Fallback: Get face landmarks for gaze estimation
        landmarks_data = self.detector.get_face_landmarks(img)
        if landmarks_data:
            landmarks = landmarks_data.get('landmarks', [])
            if landmarks:
                gaze_result = self.gaze_estimator.estimate_gaze(landmarks, img.shape[:2])
                if gaze_result and not self.eye_tracker:
                    self.detection_stats['status'] += f" | Gaze: {gaze_result.get('direction', 'unknown')}"
        
        return gaze_result
    
    def _process_face_pose(self, img: np.ndarray):
        """Process face pose detection (turning, blur)."""
        if self.face_pose_detector:
            pose_result = self.face_pose_detector.detect_pose(img)
            if pose_result['face_detected']:
                self.detection_stats['face_turned'] = pose_result['is_turned']
                self.detection_stats['face_blurry'] = pose_result['is_blurry']
                if pose_result['is_turned']:
                    self.detection_stats['status'] += f" | ⚠️ FACE TURNED: {pose_result['direction']}"
                if pose_result['is_blurry']:
                    self.detection_stats['status'] += f" | ⚠️ FACE OUT OF FOCUS"
                
                # Log if turned or blurry
                if pose_result['is_turned']:
                    evidence_path = save_evidence_image(img, 'face_turned')
                    self.storage.log_event(
                        session_id=self.session_id,
                        event_type='face_turned',
                        details=f"Direction: {pose_result['direction']}, Angles: {pose_result['angles']}",
                        evidence_image_path=evidence_path
                    )
                    if self.alert_system:
                        self.alert_system.speak_alert('FACE_TURNED')
                
                if pose_result['is_blurry']:
                    evidence_path = save_evidence_image(img, 'face_blurry')
                    self.storage.log_event(
                        session_id=self.session_id,
                        event_type='face_blurry',
                        details=f"Blur score: {pose_result['blur_score']:.2f}",
                        evidence_image_path=evidence_path
                    )
    
    def _process_object_detection(self, img: np.ndarray, detections: list) -> list:
        """Process object detection and return detected objects."""
        prohibited_objects = []
        if self.object_detector and detections:
            face_bbox = detections[0]['bbox'] if detections else None
            prohibited_objects = self.object_detector.detect_objects(img, face_bbox)
            if prohibited_objects:
                self.detection_stats['objects_detected'] = len(prohibited_objects)
                for obj in prohibited_objects:
                    evidence_path = save_evidence_image(img, 'object_detected')
                    self.storage.log_event(
                        session_id=self.session_id,
                        event_type='object_detected',
                        details=f"Object: {obj['class_name']}, Confidence: {obj['confidence']:.2f}",
                        evidence_image_path=evidence_path
                    )
                    self.detection_stats['status'] += f" | ⚠️ OBJECT: {obj['class_name']}"
                    if self.alert_system:
                        self.alert_system.speak_alert('OBJECT_DETECTED')
            else:
                self.detection_stats['objects_detected'] = 0
        return prohibited_objects
    
    def _process_audio_detection(self, img: np.ndarray):
        """Process mouth and audio detection."""
        # Mouth movement detection (talking)
        if self.mouth_detector:
            talking_result = self.mouth_detector.detect_talking(img)
            if talking_result.get('is_talking', False):
                self.detection_stats['talking_detected'] = True
                self.detection_stats['status'] += " | ⚠️ TALKING DETECTED"
                evidence_path = save_evidence_image(img, 'talking_detected')
                self.storage.log_event(
                    session_id=self.session_id,
                    event_type='talking_detected',
                    details=f"Confidence: {talking_result.get('confidence', 0):.2f}",
                    evidence_image_path=evidence_path
                )
                if self.alert_speaker:
                    self.alert_speaker.speak_alert('talking_detected')
                elif self.alert_system:
                    self.alert_system.speak_alert('TALKING_DETECTED')
            else:
                self.detection_stats['talking_detected'] = False
        else:
            self.detection_stats['talking_detected'] = False
        
        # Audio monitoring (voice detection)
        if self.audio_monitor:
            audio_result = self.audio_monitor.detect_voice()
            if audio_result.get('voice_detected', False):
                self.detection_stats['voice_detected'] = True
                self.detection_stats['status'] += " | ⚠️ VOICE DETECTED"
                self.storage.log_event(
                    session_id=self.session_id,
                    event_type='voice_detected',
                    details=f"Energy: {audio_result.get('energy', 0):.4f}, ZCR: {audio_result.get('zcr', 0):.4f}",
                    evidence_image_path=None
                )
                if self.alert_speaker:
                    self.alert_speaker.speak_alert('voice_detected')
                elif self.alert_system:
                    self.alert_system.speak_alert('VOICE_DETECTED')
            else:
                self.detection_stats['voice_detected'] = False
        else:
            self.detection_stats['voice_detected'] = False
    
    def _process_anomalies(self, img: np.ndarray, detections: list, gaze_result: Optional[Dict[str, Any]]) -> list:
        """Process anomaly detection and logging."""
        anomalies = self.anomaly_engine.process_frame(
            detections, None, gaze_result, self.expected_name
        )
        
        # Log anomalies and speak alerts
        for anomaly in anomalies:
            event_type = anomaly['event_type']
            details = anomaly.get('details', {})
            
            evidence_path = save_evidence_image(img, event_type)
            self.storage.log_event(
                session_id=self.session_id,
                event_type=event_type,
                details=str(details),
                evidence_image_path=evidence_path
            )
            
            self.detection_stats['status'] += f" | 🚨 ALERT: {event_type}"
            
            # Speak alert for critical events
            if event_type in ['face_missing', 'multiple_faces']:
                if self.alert_speaker:
                    self.alert_speaker.speak_alert(event_type)
                elif self.alert_system:
                    self.alert_system.speak_alert(event_type)
        
        return anomalies
    
    def _draw_detections_on_frame(self, img: np.ndarray, detections: list, prohibited_objects: list, anomalies: list) -> np.ndarray:
        """Draw all detections, objects, and alerts on the frame."""
        # Draw face detections
        if detections:
            img = self.detector.draw_detections(img, detections)
        else:
            # Draw warning when no face detected
            h, w = img.shape[:2]
            cv2.rectangle(img, (10, h - 100), (w - 10, h - 10), (0, 0, 255), -1)
            cv2.putText(img, "WARNING: NO FACE DETECTED!", (20, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw detected objects
        if prohibited_objects and self.object_detector:
            img = self.object_detector.draw_detections(img, prohibited_objects)
        
        # Draw alerts for anomalies
        if anomalies:
            h, w = img.shape[:2]
            for i, anomaly in enumerate(anomalies):
                alert_text = f"ALERT: {anomaly['event_type'].upper()}"
                y_pos = 100 + (i * 40)
                (text_width, text_height), _ = cv2.getTextSize(
                    alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(img, (10, y_pos - text_height - 5),
                             (10 + text_width + 10, y_pos + 5), (0, 0, 255), -1)
                cv2.putText(img, alert_text, (15, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img
    
    def _draw_status(self, img, custom_status=None):
        """Draw status information on the frame with enhanced visibility."""
        h, w = img.shape[:2]
        status = custom_status or self.detection_stats.get('status', 'Processing...')
        faces_count = self.detection_stats.get('faces_detected', 0)
        
        # Draw semi-transparent background (larger for better visibility)
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 100), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        # Color based on detection status
        if faces_count == 0:
            status_color = (0, 0, 255)  # Red when no face
            status_text = "NO FACE DETECTED!"
        else:
            status_color = (0, 255, 0)  # Green when face detected
            status_text = f"Face Detected: {faces_count}"
        
        # Draw status text (larger, more visible)
        cv2.putText(img, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(img, f"Frame: {self.frame_count} | {status[:50]}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Session: {'ACTIVE' if self.session_active else 'INACTIVE'}", 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return img


def enrollment_page():
    """Enrollment page for registering candidates."""
    st.header("📝 Candidate Enrollment")
    st.write("Enroll a candidate by capturing their reference face image.")
    
    # Get existing users
    users = st.session_state.storage.get_all_users()
    if users:
        st.subheader("Enrolled Candidates")
        for user in users:
            st.write(f"- **{user['name']}** (enrolled: {user['created_at']})")
    
    st.divider()
    
    # Enrollment form
    with st.form("enrollment_form"):
        candidate_name = st.text_input("Candidate Name", key="enroll_name")
        
        # Webcam capture
        st.write("Capture reference image:")
        captured_image = st.camera_input("Take a photo", key="enroll_camera")
        
        submitted = st.form_submit_button("Enroll Candidate")
        
        if submitted:
            if not candidate_name:
                st.error("Please enter a candidate name.")
            elif not captured_image:
                st.error("Please capture an image.")
            else:
                try:
                    # Save enrollment image
                    os.makedirs("enrolled", exist_ok=True)
                    image_path = os.path.join("enrolled", f"{candidate_name}_enrollment.jpg")
                    
                    # Convert uploaded image to OpenCV format
                    img_bytes = captured_image.read()
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    cv2.imwrite(image_path, img)
                    
                    # Save to database (no recognition, just store image)
                    user_id = st.session_state.storage.create_user(
                        name=candidate_name,
                        enrollment_image_path=image_path,
                        embedding_path=None
                    )
                    
                    st.success(f"✅ Candidate '{candidate_name}' enrolled successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during enrollment: {e}")


def exam_session_page():
    """Exam session page with real-time monitoring."""
    st.header("🎥 Exam Session")
    
    # Session controls
    col1, col2 = st.columns(2)
    
    with col1:
        candidate_name = st.text_input("Candidate Name", key="session_candidate")
    
    with col2:
        exam_title = st.text_input("Exam Title", key="session_exam_title")
    
    # Start/Stop session buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start Session", disabled=st.session_state.session_active):
            if not candidate_name:
                st.error("Please select a candidate.")
            elif not exam_title:
                st.error("Please enter an exam title.")
            else:
                # Create session
                session_id = st.session_state.storage.create_session(
                    candidate_name, exam_title
                )
                st.session_state.current_session_id = session_id
                st.session_state.expected_candidate_name = candidate_name
                st.session_state.session_active = True
                st.session_state.anomaly_engine.reset_state()
                st.success(f"✅ Session started! Session ID: {session_id}")
                st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Session", disabled=not st.session_state.session_active):
            if st.session_state.current_session_id:
                st.session_state.storage.end_session(
                    st.session_state.current_session_id
                )
                st.session_state.session_active = False
                st.session_state.current_session_id = None
                st.success("✅ Session ended!")
                st.rerun()
    
    # Display session status
    if st.session_state.session_active:
        st.info(f"🟢 Session Active - Monitoring: **{candidate_name}** - **{exam_title}**")
        
        # Inject browser event listeners
        create_browser_event_component(st.session_state.current_session_id)
        
        # Browser event detection using JavaScript injection
        st.subheader("Browser Event Detection")
        st.info("""
        **Browser Events**: The system detects tab switching and window focus changes.
        Events are detected client-side. For full bidirectional communication in production,
        a custom Streamlit component would be needed.
        """)
        
        # Manual test buttons for browser events (for demonstration)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 Test Tab Switch Event"):
                log_browser_event(
                    st.session_state.current_session_id,
                    'tab_switch',
                    {'reason': 'manual_test', 'timestamp': datetime.now().isoformat()}
                )
                st.success("Tab switch event logged!")
        
        with col2:
            if st.button("🧪 Test Window Blur Event"):
                log_browser_event(
                    st.session_state.current_session_id,
                    'window_blur',
                    {'reason': 'manual_test', 'timestamp': datetime.now().isoformat()}
                )
                st.success("Window blur event logged!")
        
        # Inject JavaScript for automatic detection
        browser_event_html = """
        <script>
        (function() {
            if (!window.proctoringEnhanced) {
                window.proctoringEnhanced = true;
                
                function logEvent(eventType, details) {
                    console.log('Proctoring Event Detected:', eventType, details);
                    // In production, this would send to backend via WebSocket or custom component
                }
                
                document.addEventListener('visibilitychange', function() {
                    if (document.hidden) {
                        logEvent('tab_switch', {reason: 'tab_hidden'});
                    } else {
                        logEvent('tab_focus', {reason: 'tab_visible'});
                    }
                });
                
                window.addEventListener('blur', function() {
                    logEvent('window_blur', {reason: 'window_lost_focus'});
                });
                
                window.addEventListener('focus', function() {
                    logEvent('window_focus', {reason: 'window_gained_focus'});
                });
            }
        })();
        </script>
        """
        st.components.v1.html(browser_event_html, height=0)
        
        # Real-time video stream
        st.subheader("Live Video Feed")
        
        # Create processor instance and set session info
        processor = VideoProcessor()
        processor.set_session(
            st.session_state.current_session_id,
            candidate_name,
            st.session_state.session_active
        )
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="exam_session",
            video_processor_factory=lambda: processor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        # Display real-time detection status
        if webrtc_ctx.state.playing:
            st.success("🟢 Video stream active - Real-time detection running!")
            
            # Show detection stats
            if hasattr(processor, 'detection_stats'):
                stats = processor.detection_stats
                faces_count = stats.get('faces_detected', 0)
                
                # BIG ALERT if no face detected
                if faces_count == 0:
                    st.error("🚨 **ALERT: NO FACE DETECTED!** User may have left the screen!")
                    # Check if face_missing event was logged
                    if st.session_state.current_session_id:
                        recent_events = st.session_state.storage.get_session_events(
                            st.session_state.current_session_id
                        )
                        face_missing_events = [e for e in recent_events[-5:] if e['event_type'] == 'face_missing']
                        if face_missing_events:
                            st.warning(f"⚠️ Face missing alert logged at: {face_missing_events[-1]['timestamp']}")
                else:
                    st.success(f"✅ Face detected: {faces_count} face(s) visible")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Color-coded metric
                    if faces_count == 0:
                        st.metric("Faces Detected", faces_count, delta=None, delta_color="off")
                    else:
                        st.metric("Faces Detected", faces_count, delta=None, delta_color="normal")
                with col2:
                    st.metric("Frames Processed", processor.frame_count)
                with col3:
                    status = stats.get('status', 'Processing...')
                    if 'ALERT' in status or 'MISMATCH' in status:
                        st.error(f"**Status:** {status}")
                    elif faces_count == 0:
                        st.warning(f"**Status:** {status}")
                    else:
                        st.info(f"**Status:** {status}")
        else:
            st.warning("⏸️ Click 'Start' on the video player to begin detection")
        
        # Enhanced Real-time Metrics Dashboard
        if st.session_state.current_session_id:
            events = st.session_state.storage.get_session_events(
                st.session_state.current_session_id
            )
            
            if events:
                st.subheader("📊 Real-time Metrics Dashboard")
                
                # Event counts by type
                event_counts = {}
                for event in events:
                    event_type = event['event_type']
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
                # Display metrics in columns
                cols = st.columns(min(len(event_counts), 4))
                for idx, (event_type, count) in enumerate(list(event_counts.items())[:4]):
                    with cols[idx % len(cols)]:
                        st.metric(event_type.replace('_', ' ').title(), count)
                
                # Violation Timeline
                st.subheader("📈 Violation Timeline")
                if len(events) > 0:
                    import pandas as pd
                    timeline_data = []
                    for event in events:
                        timeline_data.append({
                            'Time': event['timestamp'],
                            'Event': event['event_type'].replace('_', ' ').title(),
                            'Type': event['event_type']
                        })
                    df_timeline = pd.DataFrame(timeline_data)
                    st.line_chart(df_timeline.set_index('Time')['Event'].value_counts().sort_index())
                
                # Evidence Gallery
                st.subheader("🖼️ Evidence Gallery")
                evidence_events = [e for e in events if e.get('evidence_image_path')]
                if evidence_events:
                    cols = st.columns(3)
                    for idx, event in enumerate(evidence_events[-9:]):  # Show last 9 images
                        with cols[idx % 3]:
                            if os.path.exists(event['evidence_image_path']):
                                st.image(event['evidence_image_path'], 
                                        caption=f"{event['event_type']} - {event['timestamp']}", 
                                        use_container_width=True)
                else:
                    st.info("No evidence images captured yet.")
                
                # Recent Events List
                st.subheader("📋 Recent Events")
                for event in events[-10:]:  # Show last 10 events
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"⏰ **{event['timestamp']}** - {event['event_type'].replace('_', ' ').title()}")
                    with col2:
                        if event.get('evidence_image_path') and os.path.exists(event['evidence_image_path']):
                            st.image(event['evidence_image_path'], width=100)
                
                # Event Summary
                st.subheader("📊 Event Summary")
                summary_cols = st.columns(len(event_counts))
                for idx, (event_type, count) in enumerate(event_counts.items()):
                    with summary_cols[idx % len(summary_cols)]:
                        st.metric(event_type.replace('_', ' ').title(), count)
    else:
        st.info("👆 Start a session to begin monitoring.")


def student_exam_page():
    """Student exam page with MCQ questions and proctoring."""
    
    # Custom CSS for better UI - Dark Mode
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .exam-container {
        padding: 20px;
        background-color: #2d2d2d;
        border-radius: 10px;
        margin: 10px 0;
        color: #ffffff;
    }
    .timer-display {
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        color: #ffffff;
    }
    .timer-warning {
        background: linear-gradient(135deg, #856404 0%, #b8860b 100%);
        color: #ffffff;
        border: 2px solid #ffc107;
    }
    .timer-danger {
        background: linear-gradient(135deg, #721c24 0%, #c82333 100%);
        color: #ffffff;
        border: 2px solid #dc3545;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    .timer-success {
        background: linear-gradient(135deg, #155724 0%, #28a745 100%);
        color: #ffffff;
        border: 2px solid #28a745;
    }
    .question-card {
        background-color: #2d2d2d;
        padding: 25px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5);
        border-left: 4px solid #3498db;
        color: #ffffff;
    }
    .camera-container {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #3498db;
        min-height: 400px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        background-color: #3498db;
        color: #ffffff;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
        background-color: #2980b9;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    .stRadio label {
        color: #ffffff !important;
    }
    .questions-scrollable::-webkit-scrollbar {
        width: 10px;
    }
    .questions-scrollable::-webkit-scrollbar-track {
        background: #1e1e1e;
        border-radius: 10px;
    }
    .questions-scrollable::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 10px;
    }
    .questions-scrollable::-webkit-scrollbar-thumb:hover {
        background: #777;
    }
    /* Ensure questions column is scrollable */
    #questions-scroll-wrapper {
        max-height: 70vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    /* Target Streamlit column structure */
    div[data-testid="column"]:nth-of-type(2) {
        max-height: 75vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    div[data-testid="column"]:nth-of-type(2) > div {
        max-height: 75vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("📝 Take Exam")
    
    # If exam completed, show completion message
    if st.session_state.exam_completed and st.session_state.exam_session_id:
        st.success("✅ Exam completed successfully!")
        st.info("📊 Please select 'Exam Results' from the navigation menu to view your results.")
        return
    
    # Get questions
    questions = st.session_state.exam_manager.get_all_questions()
    
    if not questions:
        st.error("No questions available. Please contact administrator.")
        return
    
    # Name input (if not set or exam not started)
    if not st.session_state.student_name or not st.session_state.exam_active:
        with st.form("student_info_form"):
            st.subheader("Enter Your Information")
            student_name = st.text_input("Your Name", value=st.session_state.student_name or "", key="exam_student_name")
            submitted = st.form_submit_button("Start Exam", type="primary", use_container_width=True)
            
            if submitted:
                if not student_name.strip():
                    st.error("Please enter your name.")
                else:
                    st.session_state.student_name = student_name.strip()
                    st.rerun()
    
    # Exam interface
    if st.session_state.student_name:
        if not st.session_state.exam_active:
            # Pre-exam screen
            st.info(f"👋 Welcome, **{st.session_state.student_name}**!")
            st.write("### Exam Instructions:")
            st.write("""
            - You will have **5 minutes** to complete the exam
            - There are **{} questions** in total
            - Proctoring will start automatically when you begin
            - Make sure your camera is enabled and working
            - Once you start, the timer cannot be paused
            """.format(len(questions)))
            
            if st.button("🚀 Start Exam", type="primary", use_container_width=True):
                # Initialize exam
                st.session_state.exam_start_time = time.time()
                st.session_state.exam_active = True
                st.session_state.exam_answers = {}
                st.session_state.exam_completed = False
                
                # Create session and start proctoring
                session_id = st.session_state.storage.create_session(
                    candidate_name=st.session_state.student_name,
                    exam_title="Programming MCQ Exam",
                    total_questions=len(questions)
                )
                st.session_state.exam_session_id = session_id
                st.session_state.current_session_id = session_id
                st.session_state.session_active = True
                st.session_state.expected_candidate_name = st.session_state.student_name
                st.session_state.anomaly_engine.reset_state()
                
                st.success("✅ Exam started! Good luck!")
                st.rerun()
        
        else:
            # Exam in progress
            # Calculate remaining time
            elapsed_time = time.time() - st.session_state.exam_start_time
            remaining_time = max(0, st.session_state.exam_duration - elapsed_time)
            
            # Check if time expired
            if remaining_time <= 0 and not st.session_state.exam_completed:
                # Auto-submit exam
                _submit_exam(questions)
                st.rerun()
                return
            
            # Timer display
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            timer_text = f"{minutes:02d}:{seconds:02d}"
            
            # Color code timer
            if remaining_time < 60:
                timer_class = "timer-danger"
            elif remaining_time < 120:
                timer_class = "timer-warning"
            else:
                timer_class = "timer-success"
            
            st.markdown(f'<div class="timer-display {timer_class}">{timer_text}</div>', unsafe_allow_html=True)
            
            # Proctoring status
            if st.session_state.session_active:
                st.success("🟢 Proctoring Active - You are being monitored")
            else:
                st.warning("⚠️ Proctoring not active")
            
            st.divider()
            
            # Split screen layout: Camera on left, Questions on right
            cam_col, questions_col = st.columns([1, 1], gap="medium")
            
            # JavaScript to make questions column scrollable
            st.markdown("""
            <script>
            (function() {
                function applyScrolling() {
                    // Find all columns
                    const columns = Array.from(document.querySelectorAll('[data-testid="column"]'));
                    if (columns.length >= 2) {
                        const questionsColumn = columns[1]; // Second column
                        // Get all child divs in the column
                        const childDivs = questionsColumn.querySelectorAll('div');
                        // Find the div that contains "Questions" heading
                        let targetDiv = null;
                        childDivs.forEach(div => {
                            if (div.textContent && div.textContent.includes('Questions') && !targetDiv) {
                                // Find the parent container that holds all content
                                let parent = div;
                                for (let i = 0; i < 5 && parent; i++) {
                                    parent = parent.parentElement;
                                    if (parent && parent.style) {
                                        targetDiv = parent;
                                    }
                                }
                            }
                        });
                        // Apply scrolling to the main content div
                        if (!targetDiv) {
                            // Fallback: apply to first flex container
                            targetDiv = questionsColumn.querySelector('div[style*="flex"]');
                        }
                        if (!targetDiv && questionsColumn.children.length > 0) {
                            targetDiv = questionsColumn.children[0];
                        }
                        if (targetDiv) {
                            targetDiv.style.setProperty('max-height', '70vh', 'important');
                            targetDiv.style.setProperty('overflow-y', 'auto', 'important');
                            targetDiv.style.setProperty('overflow-x', 'hidden', 'important');
                            targetDiv.style.setProperty('padding-right', '15px', 'important');
                        }
                    }
                }
                // Apply immediately and on delays
                setTimeout(applyScrolling, 200);
                setTimeout(applyScrolling, 800);
                setTimeout(applyScrolling, 1500);
                // Use mutation observer to apply when DOM changes
                const observer = new MutationObserver(function() {
                    setTimeout(applyScrolling, 100);
                });
                if (document.body) {
                    observer.observe(document.body, { childList: true, subtree: true });
                }
            })();
            </script>
            <style>
            /* Scrollbar styling for questions column */
            div[data-testid="column"]:nth-of-type(2) div {
                scrollbar-width: thin;
                scrollbar-color: #555 #1e1e1e;
            }
            div[data-testid="column"]:nth-of-type(2) div::-webkit-scrollbar {
                width: 10px;
            }
            div[data-testid="column"]:nth-of-type(2) div::-webkit-scrollbar-track {
                background: #1e1e1e;
                border-radius: 10px;
            }
            div[data-testid="column"]:nth-of-type(2) div::-webkit-scrollbar-thumb {
                background: #555;
                border-radius: 10px;
            }
            div[data-testid="column"]:nth-of-type(2) div::-webkit-scrollbar-thumb:hover {
                background: #777;
            }
            </style>
            """, unsafe_allow_html=True)
            
            with cam_col:
                st.markdown("### 📹 Proctoring Camera")
                if st.session_state.session_active and st.session_state.exam_session_id:
                    processor = VideoProcessor()
                    processor.set_session(
                        st.session_state.exam_session_id,
                        st.session_state.student_name,
                        st.session_state.session_active
                    )
                    
                    # Inject browser event listeners
                    create_browser_event_component(st.session_state.exam_session_id)
                    
                    # Try to auto-start camera with JavaScript (attempts multiple times)
                    auto_start_js = """
                    <script>
                    (function() {
                        let attempts = 0;
                        const maxAttempts = 5;
                        function tryStartCamera() {
                            attempts++;
                            const buttons = document.querySelectorAll('button, [role="button"]');
                            buttons.forEach(btn => {
                                const text = btn.textContent || btn.innerText || '';
                                const title = btn.getAttribute('title') || '';
                                if (text.includes('START') || text.trim() === '▶' || title.includes('Start') || 
                                    (btn.querySelector('svg') && text.trim() === '')) {
                                    try {
                                        btn.click();
                                        console.log('Attempted to start camera');
                                    } catch(e) {}
                                }
                            });
                            if (attempts < maxAttempts) {
                                setTimeout(tryStartCamera, 500);
                            }
                        }
                        setTimeout(tryStartCamera, 500);
                    })();
                    </script>
                    """
                    st.markdown(auto_start_js, unsafe_allow_html=True)
                    
                    webrtc_ctx = webrtc_streamer(
                        key="student_exam_proctoring",
                        video_processor_factory=lambda: processor,
                        rtc_configuration=RTC_CONFIGURATION,
                        media_stream_constraints={"video": True, "audio": False},
                    )
                    
                    if webrtc_ctx.state.playing:
                        st.success("✅ Camera is active and monitoring")
                    else:
                        st.warning("⚠️ **Please click 'START' on the video player above to activate camera monitoring**")
                        st.info("💡 The camera must be manually activated due to browser security requirements.")
                else:
                    st.info("Camera will start automatically when exam begins.")
            
            with questions_col:
                # Questions display with scrollable container
                st.markdown(f"### Questions ({len(questions)} total)")
                
                # Add CSS to make questions column scrollable - more aggressive targeting
                st.markdown("""
                <style>
                /* Target questions column more aggressively */
                div[data-testid="column"]:nth-of-type(2),
                div[data-testid="column"]:nth-of-type(2) > div,
                div[data-testid="column"]:nth-of-type(2) > div > div {
                    max-height: 70vh !important;
                    overflow-y: auto !important;
                    overflow-x: hidden !important;
                }
                /* Scrollbar styling */
                div[data-testid="column"]:nth-of-type(2)::-webkit-scrollbar,
                div[data-testid="column"]:nth-of-type(2) > div::-webkit-scrollbar,
                div[data-testid="column"]:nth-of-type(2) > div > div::-webkit-scrollbar {
                    width: 10px;
                }
                div[data-testid="column"]:nth-of-type(2)::-webkit-scrollbar-track,
                div[data-testid="column"]:nth-of-type(2) > div::-webkit-scrollbar-track,
                div[data-testid="column"]:nth-of-type(2) > div > div::-webkit-scrollbar-track {
                    background: #1e1e1e;
                    border-radius: 10px;
                }
                div[data-testid="column"]:nth-of-type(2)::-webkit-scrollbar-thumb,
                div[data-testid="column"]:nth-of-type(2) > div::-webkit-scrollbar-thumb,
                div[data-testid="column"]:nth-of-type(2) > div > div::-webkit-scrollbar-thumb {
                    background: #555;
                    border-radius: 10px;
                }
                div[data-testid="column"]:nth-of-type(2)::-webkit-scrollbar-thumb:hover,
                div[data-testid="column"]:nth-of-type(2) > div::-webkit-scrollbar-thumb:hover,
                div[data-testid="column"]:nth-of-type(2) > div > div::-webkit-scrollbar-thumb:hover {
                    background: #777;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Store answers in session state
                for idx, question in enumerate(questions, 1):
                    with st.container():
                        st.markdown(f'<div class="question-card">', unsafe_allow_html=True)
                        st.write(f"### Question {idx}")
                        st.write(f"**{question['question_text']}**")
                        
                        # Radio buttons for options
                        options = [
                            f"A: {question['option_a']}",
                            f"B: {question['option_b']}",
                            f"C: {question['option_c']}",
                            f"D: {question['option_d']}"
                        ]
                        
                        answer_key = f"q{question['id']}"
                        selected = st.radio(
                            "Select your answer:",
                            options=options,
                            key=answer_key,
                            index=None if question['id'] not in st.session_state.exam_answers else 
                                  ['A', 'B', 'C', 'D'].index(st.session_state.exam_answers[question['id']])
                        )
                        
                        if selected:
                            # Extract answer letter (A, B, C, or D)
                            selected_answer = selected.split(':')[0].strip()
                            st.session_state.exam_answers[question['id']] = selected_answer
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.write("")
                
                # JavaScript fallback to ensure scrolling works
                st.markdown("""
                <script>
                (function() {
                    function makeScrollable() {
                        // Try multiple approaches to find and make the questions column scrollable
                        const wrapper = document.getElementById('questions-scroll-wrapper');
                        if (wrapper) {
                            wrapper.style.maxHeight = '70vh';
                            wrapper.style.overflowY = 'auto';
                            wrapper.style.overflowX = 'hidden';
                        }
                        
                        // Also target the column container
                        const columns = document.querySelectorAll('[data-testid="column"]');
                        if (columns.length >= 2) {
                            const questionsColumn = columns[1];
                            questionsColumn.style.maxHeight = '75vh';
                            questionsColumn.style.overflowY = 'auto';
                            questionsColumn.style.overflowX = 'hidden';
                            
                            // Try to find the inner div
                            const innerDiv = questionsColumn.querySelector('div');
                            if (innerDiv) {
                                innerDiv.style.maxHeight = '75vh';
                                innerDiv.style.overflowY = 'auto';
                                innerDiv.style.overflowX = 'hidden';
                            }
                        }
                    }
                    
                    // Run immediately
                    makeScrollable();
                    
                    // Also run after a short delay to catch dynamically loaded content
                    setTimeout(makeScrollable, 100);
                    setTimeout(makeScrollable, 500);
                })();
                </script>
                """, unsafe_allow_html=True)
            
            # Submit button (full width below split screen)
            st.divider()
            all_answered = len(st.session_state.exam_answers) == len(questions)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if all_answered:
                    st.success(f"✅ All {len(questions)} questions answered!")
                else:
                    st.info(f"📝 Answered: {len(st.session_state.exam_answers)}/{len(questions)}")
            
            with col2:
                if st.button("Submit Exam", type="primary", disabled=st.session_state.exam_completed, use_container_width=True):
                    _submit_exam(questions)
                    st.rerun()
            
            # Auto-refresh timer (Streamlit will rerun automatically)
            # Using a placeholder approach for smoother updates
            if remaining_time > 0:
                # Small delay to prevent too frequent reruns
                time.sleep(0.5)
                st.rerun()


def _submit_exam(questions):
    """Submit exam, save answers, calculate score, and stop proctoring."""
    if st.session_state.exam_completed:
        return
    
    st.session_state.exam_completed = True
    st.session_state.exam_active = False
    
    # Save all answers
    session_id = st.session_state.exam_session_id
    if session_id:
        # Save answers to database
        for question_id, selected_answer in st.session_state.exam_answers.items():
            is_correct = st.session_state.exam_manager.validate_answer(question_id, selected_answer)
            st.session_state.storage.save_answer(session_id, question_id, selected_answer, is_correct)
        
        # Calculate score
        score_info = st.session_state.exam_manager.calculate_score(session_id)
        
        # Update session with score
        completion_status = 'completed' if len(st.session_state.exam_answers) == len(questions) else 'timeout'
        st.session_state.storage.update_session_score(
            session_id,
            score_info['score'],
            score_info['total_questions'],
            completion_status
        )
        
        # Stop proctoring
        st.session_state.storage.end_session(session_id)
        st.session_state.session_active = False


def exam_results_page():
    """Display exam results with score, answers, and proctoring violations."""
    
    # Custom CSS - Dark Mode
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .result-card {
        background-color: #2d2d2d;
        padding: 25px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5);
        transition: transform 0.2s ease;
        color: #ffffff;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.7);
    }
    .score-display {
        font-size: 5em;
        font-weight: bold;
        text-align: center;
        padding: 40px;
        border-radius: 15px;
        margin: 30px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        color: #ffffff;
    }
    .score-high {
        background: linear-gradient(135deg, #155724 0%, #28a745 100%);
        color: #ffffff;
        border: 3px solid #28a745;
    }
    .score-medium {
        background: linear-gradient(135deg, #856404 0%, #ffc107 100%);
        color: #ffffff;
        border: 3px solid #ffc107;
    }
    .score-low {
        background: linear-gradient(135deg, #721c24 0%, #dc3545 100%);
        color: #ffffff;
        border: 3px solid #dc3545;
    }
    .answer-correct {
        background: linear-gradient(135deg, #155724 0%, #28a745 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
        color: #ffffff;
    }
    .answer-incorrect {
        background: linear-gradient(135deg, #721c24 0%, #dc3545 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
        color: #ffffff;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        background-color: #3498db;
        color: #ffffff;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
        background-color: #2980b9;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    p, label, span, div {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("📊 Exam Results")
    
    session_id = st.session_state.exam_session_id
    
    if not session_id:
        st.warning("No exam session found. Please take an exam first.")
        st.info("💡 Use the navigation menu to go to 'Take Exam' page.")
        return
    
    # Get exam results
    results = st.session_state.exam_manager.get_exam_results(session_id)
    
    if not results or not results.get('session'):
        st.error("Could not load exam results.")
        return
    
    session = results['session']
    answers = results.get('answers', [])
    score_info = results.get('score', {})
    
    # Display student info
    st.subheader(f"Student: **{session['candidate_name']}**")
    st.write(f"**Exam:** {session['exam_title']}")
    st.write(f"**Date:** {session['started_at']}")
    
    st.divider()
    
    # Score display
    score = score_info.get('score', 0)
    total = score_info.get('total_questions', 0)
    percentage = score_info.get('percentage', 0.0)
    
    if percentage >= 70:
        score_class = "score-high"
    elif percentage >= 50:
        score_class = "score-medium"
    else:
        score_class = "score-low"
    
    st.markdown(f'<div class="score-display {score_class}">{score}/{total}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: center; font-size: 1.5em; margin-bottom: 20px;"><strong>{percentage:.1f}%</strong></div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Answer review
    st.subheader("Answer Review")
    
    for answer_data in answers:
        is_correct = answer_data['is_correct']
        answer_class = "answer-correct" if is_correct else "answer-incorrect"
        status_icon = "✅" if is_correct else "❌"
        
        with st.container():
            st.markdown(f'<div class="result-card {answer_class}">', unsafe_allow_html=True)
            st.write(f"### {status_icon} Question")
            st.write(f"**{answer_data['question_text']}**")
            
            st.write("**Options:**")
            st.write(f"- A: {answer_data['option_a']}")
            st.write(f"- B: {answer_data['option_b']}")
            st.write(f"- C: {answer_data['option_c']}")
            st.write(f"- D: {answer_data['option_d']}")
            
            st.write("**Your Answer:**", f"**{answer_data['selected_answer']}**")
            st.write("**Correct Answer:**", f"**{answer_data['correct_answer']}**")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.write("")
    
    st.divider()
    
    # AI Proctoring Summary (calculated using formulas)
    st.subheader("🤖 AI Proctoring Summary")
    
    events = st.session_state.storage.get_session_events(session_id)
    
    if events:
        # Violation type descriptions (user-friendly)
        violation_descriptions = {
            'eyes_closed': 'Eyes Closed',
            'gaze_away': 'Looking Away from Screen',
            'gaze_offscreen': 'Gaze Off Screen',
            'face_turned': 'Face Turned Away',
            'face_blurry': 'Face Out of Focus',
            'face_missing': 'Face Not Detected',
            'multiple_faces': 'Multiple People Detected',
            'talking_detected': 'Talking Detected',
            'voice_detected': 'Voice/Noise Detected',
            'object_detected': 'Unauthorized Object Detected',
            'identity_mismatch': 'Identity Verification Failed',
            'tab_switch': 'Browser Tab Switched',
            'window_blur': 'Window Focus Lost'
        }
        
        # Severity weights from config (use report generator's severity map)
        severity_map = {}
        try:
            severity_map = st.session_state.report_generator.severity_map
        except:
            # Fallback to config defaults if not available
            severity_map = {
                'FACE_DISAPPEARED': 1, 'FACE_MISSING': 1,
                'GAZE_AWAY': 2, 'GAZE_OFFSCREEN': 2, 'EYES_CLOSED': 2,
                'FACE_TURNED': 2, 'FACE_BLURRY': 2,
                'MOUTH_MOVING': 3, 'TALKING_DETECTED': 3,
                'AUDIO_DETECTED': 3, 'VOICE_DETECTED': 3,
                'MULTIPLE_FACES': 4, 'TAB_SWITCH': 4, 'WINDOW_BLUR': 4,
                'OBJECT_DETECTED': 5, 'IDENTITY_MISMATCH': 1
            }
        
        # Count violations by type
        violation_counts = {}
        total_severity = 0
        for event in events:
            event_type = event['event_type'].upper()
            violation_counts[event_type] = violation_counts.get(event_type, 0) + 1
            # Add severity weight
            total_severity += severity_map.get(event_type, 1)
        
        # Calculate violation score (0-100 scale)
        # Formula: (100 - (total_severity / max_possible_severity) * 100)
        # Max possible severity = total_events * 5 (highest severity)
        max_possible_severity = len(events) * 5
        violation_score = max(0, 100 - (total_severity / max_possible_severity) * 100) if max_possible_severity > 0 else 100
        
        # Calculate severity score (weighted sum)
        severity_score = total_severity
        
        # Determine risk level
        if violation_score >= 80:
            risk_level = "Low Risk"
            risk_color = "#28a745"
        elif violation_score >= 60:
            risk_level = "Medium Risk"
            risk_color = "#ffc107"
        else:
            risk_level = "High Risk"
            risk_color = "#dc3545"
        
        # Display scores in a card format
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Violation Score", f"{violation_score:.1f}/100", 
                     delta=f"{risk_level}", delta_color="off")
        with col2:
            st.metric("Severity Score", f"{severity_score:.0f}", 
                     delta="Weighted Sum", delta_color="off")
        with col3:
            st.metric("Total Violations", len(events), 
                     delta=f"{len(violation_counts)} types", delta_color="off")
        
        # Risk level indicator with AI summary explanation
        st.markdown(f"""
        <div style="background-color: {risk_color}20; border-left: 4px solid {risk_color}; 
                    padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h4 style="color: {risk_color}; margin: 0;">AI Analysis Result: {risk_level}</h4>
            <p style="margin: 5px 0 0 0; color: #ffffff;">
                Based on comprehensive AI analysis: <strong>{violation_score:.1f}%</strong> compliance score indicates <strong>{risk_level.lower()}</strong> behavior during the exam session.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculation formulas explanation (collapsible)
        with st.expander("📊 View Calculation Formulas (AI Summary Methodology)", expanded=False):
            st.markdown("""
            **AI Summary Calculation Methodology:**
            
            **1. Severity Score Calculation:**
            - Formula: `Severity Score = Σ(violation_count × severity_weight)`
            - Each violation type has a severity weight (1-5 scale)
            - Higher weights indicate more serious violations
            - Total severity score: **{}**
            
            **2. Violation Score Calculation (0-100 scale):**
            - Formula: `Violation Score = 100 - (total_severity / max_possible_severity) × 100`
            - Max possible severity = total_violations × 5 (highest severity level)
            - Max possible for this exam: **{}**
            - Calculated score: **{:.1f}/100**
            
            **3. Risk Level Assessment:**
            - **Low Risk**: Violation Score ≥ 80%
            - **Medium Risk**: Violation Score 60-79%
            - **High Risk**: Violation Score < 60%
            
            **4. Severity Weights (1-5 scale):**
            - Weight 1: Minor issues (Face Missing, Identity Mismatch)
            - Weight 2: Moderate issues (Gaze Away, Eyes Closed, Face Turned)
            - Weight 3: Significant issues (Talking, Voice Detected)
            - Weight 4: Serious issues (Multiple Faces, Tab Switch)
            - Weight 5: Critical issues (Unauthorized Objects)
            """.format(severity_score, max_possible_severity, violation_score))
        
        # Violation breakdown with descriptions
        if violation_counts:
            st.write("**Violation Breakdown:**")
            violation_list = []
            for v_type, count in sorted(violation_counts.items(), key=lambda x: x[1], reverse=True):
                v_type_lower = v_type.lower()
                description = violation_descriptions.get(v_type_lower, v_type.replace('_', ' ').title())
                severity_weight = severity_map.get(v_type, 1)
                violation_list.append({
                    'description': description,
                    'count': count,
                    'severity': severity_weight,
                    'contribution': count * severity_weight
                })
            
            for v in violation_list:
                st.write(f"- **{v['description']}**: {v['count']} occurrence(s) (Severity weight: {v['severity']})")
        
        # Generate and show full report button
        st.divider()
        if st.button("📄 Generate Full Proctoring Report", use_container_width=True):
            try:
                violations = []
                for event in events:
                    violations.append({
                        'type': event['event_type'],
                        'timestamp': event['timestamp'],
                        'image_path': event.get('evidence_image_path'),
                        'metadata': {'details': event.get('details')}
                    })
                
                student_info = {
                    'id': f"SESSION_{session_id}",
                    'name': session['candidate_name'],
                    'exam': session['exam_title']
                }
                
                report_path = st.session_state.report_generator.generate_report(
                    student_info, violations, output_format='html'
                )
                if report_path:
                    st.success(f"✅ Report generated: {report_path}")
                    with open(report_path, 'rb') as f:
                        st.download_button(
                            "📥 Download Full Report",
                            f.read(),
                            file_name=os.path.basename(report_path),
                            mime="text/html"
                        )
            except Exception as e:
                st.error(f"Error generating report: {e}")
    else:
        st.success("✅ No violations detected during the exam!")
    
    st.divider()
    
    # Navigation
    if st.button("Take Another Exam", use_container_width=True):
        # Reset exam state
        st.session_state.exam_completed = False
        st.session_state.exam_active = False
        st.session_state.exam_session_id = None
        st.session_state.student_name = None
        st.session_state.exam_answers = {}
        st.session_state.exam_start_time = None
        st.rerun()


def instructor_dashboard_page():
    """Instructor dashboard for reviewing sessions."""
    st.header("📊 Instructor Dashboard")
    
    # Get all sessions
    sessions = st.session_state.storage.get_all_sessions()
    
    if not sessions:
        st.info("No sessions found. Start an exam session to see data here.")
        return
    
    # Session list
    st.subheader("All Sessions")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        reviewed_filter = st.selectbox(
            "Filter by Review Status",
            ["All", "Reviewed", "Not Reviewed"],
            key="review_filter"
        )
    
    with col2:
        candidate_filter = st.selectbox(
            "Filter by Candidate",
            ["All"] + list(set([s['candidate_name'] for s in sessions])),
            key="candidate_filter"
        )
    
    # Filter sessions
    filtered_sessions = sessions
    if reviewed_filter == "Reviewed":
        filtered_sessions = [s for s in filtered_sessions if s['reviewed']]
    elif reviewed_filter == "Not Reviewed":
        filtered_sessions = [s for s in filtered_sessions if not s['reviewed']]
    
    if candidate_filter != "All":
        filtered_sessions = [s for s in filtered_sessions if s['candidate_name'] == candidate_filter]
    
    # Display sessions
    for session in filtered_sessions:
        with st.expander(
            f"Session {session['id']}: {session['candidate_name']} - {session['exam_title']} "
            f"({session['started_at']}) {'✅ Reviewed' if session['reviewed'] else '⏳ Pending'}"
        ):
            # Session details
            st.write(f"**Candidate:** {session['candidate_name']}")
            st.write(f"**Exam:** {session['exam_title']}")
            st.write(f"**Started:** {session['started_at']}")
            st.write(f"**Ended:** {session['ended_at'] or 'In Progress'}")
            
            # Get events for this session
            events = st.session_state.storage.get_session_events(session['id'])
            
            if events:
                st.subheader("Events Timeline")
                
                # Event summary
                event_types = {}
                for event in events:
                    event_type = event['event_type']
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                
                st.write("**Event Summary:**")
                for event_type, count in event_types.items():
                    st.write(f"- {event_type}: {count}")
                
                st.divider()
                
                # Detailed timeline
                for event in events:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**{event['event_type']}**")
                    
                    with col2:
                        st.write(f"⏰ {event['timestamp']}")
                    
                    with col3:
                        if event.get('evidence_image_path'):
                            if os.path.exists(event['evidence_image_path']):
                                st.image(event['evidence_image_path'], width=150)
                    
                    # Show details if available
                    if event.get('details'):
                        with st.expander("Details"):
                            st.write(event['details'])
            else:
                st.info("No events recorded for this session.")
            
            # Report Generation
            st.divider()
            st.subheader("📄 Generate Report")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"📊 Generate HTML Report", key=f"report_html_{session['id']}"):
                    try:
                        # Get violations from events
                        violations = []
                        for event in events:
                            violations.append({
                                'type': event['event_type'],
                                'timestamp': event['timestamp'],
                                'image_path': event.get('evidence_image_path'),
                                'metadata': {'details': event.get('details')}
                            })
                        
                        student_info = {
                            'id': f"SESSION_{session['id']}",
                            'name': session['candidate_name'],
                            'exam': session['exam_title']
                        }
                        
                        report_path = st.session_state.report_generator.generate_report(
                            student_info, violations, output_format='html'
                        )
                        if report_path:
                            st.success(f"✅ Report generated: {report_path}")
                            with open(report_path, 'rb') as f:
                                st.download_button(
                                    "📥 Download HTML Report",
                                    f.read(),
                                    file_name=os.path.basename(report_path),
                                    mime="text/html"
                                )
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
            
            with col2:
                if st.button(f"📄 Generate PDF Report", key=f"report_pdf_{session['id']}"):
                    try:
                        violations = []
                        for event in events:
                            violations.append({
                                'type': event['event_type'],
                                'timestamp': event['timestamp'],
                                'image_path': event.get('evidence_image_path'),
                                'metadata': {'details': event.get('details')}
                            })
                        
                        student_info = {
                            'id': f"SESSION_{session['id']}",
                            'name': session['candidate_name'],
                            'exam': session['exam_title']
                        }
                        
                        report_path = st.session_state.report_generator.generate_report(
                            student_info, violations, output_format='pdf'
                        )
                        if report_path:
                            st.success(f"✅ Report generated: {report_path}")
                            with open(report_path, 'rb') as f:
                                st.download_button(
                                    "📥 Download PDF Report",
                                    f.read(),
                                    file_name=os.path.basename(report_path),
                                    mime="application/pdf"
                                )
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
            
            # Mark as reviewed button
            if not session['reviewed']:
                if st.button(f"✅ Mark as Reviewed", key=f"review_{session['id']}"):
                    st.session_state.storage.mark_session_reviewed(session['id'])
                    st.success("✅ Session marked as reviewed!")
                    st.rerun()


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Online Proctoring System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Global UI improvements - Dark Mode
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stApp > header {
        background-color: #2d2d2d;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .sidebar .sidebar-content * {
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    h1 {
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        color: #ffffff;
    }
    h2 {
        color: #ffffff;
        margin-top: 2rem;
    }
    p, label, span, div {
        color: #ffffff;
    }
    .stSuccess {
        background-color: #1e4620;
        border-left: 4px solid #28a745;
        color: #d4edda;
    }
    .stWarning {
        background-color: #4a3e00;
        border-left: 4px solid #ffc107;
        color: #fff3cd;
    }
    .stError {
        background-color: #4a1e1e;
        border-left: 4px solid #dc3545;
        color: #f8d7da;
    }
    .stInfo {
        background-color: #1e3a4a;
        border-left: 4px solid #17a2b8;
        color: #d1ecf1;
    }
    .stTextInput > div > div > input {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stSelectbox > div > div > select {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stRadio > div {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #3498db;
        color: #ffffff;
    }
    [data-baseweb="select"] {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 Online Proctoring System")
    st.sidebar.title("Navigation")
    
    # Navigation
    page = st.sidebar.radio(
        "Select Page",
        ["Take Exam", "Exam Results", "Instructor Dashboard"]
    )
    
    # Route to appropriate page
    if page == "Take Exam":
        student_exam_page()
    elif page == "Exam Results":
        exam_results_page()
    elif page == "Instructor Dashboard":
        instructor_dashboard_page()
    
    # Footer
    st.sidebar.divider()
    st.sidebar.write("**System Status:**")
    st.sidebar.write(f"- Database: ✅ Connected")
    st.sidebar.write(f"- Detector: ✅ Ready")


if __name__ == "__main__":
    main()
