"""
Robust face detection module using multiple models (MediaPipe, RetinaFace, MTCNN, YOLOv8).
Ensemble approach for maximum reliability.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Optional, Tuple
import time

# Try to import advanced models
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

try:
    import insightface
    from insightface.app import FaceAnalysis
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False
    FaceAnalysis = None

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    MTCNN = None

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    dlib = None


class FaceDetector:
    """Robust face detector using ensemble of multiple models."""
    
    def __init__(self, min_detection_confidence: float = 0.3):
        """
        Initialize multiple face detection models for robustness.
        
        Args:
            min_detection_confidence: Minimum confidence threshold
        """
        self.min_confidence = min_detection_confidence
        self.detection_methods = []
        
        # Method 1: MediaPipe (always available, fast)
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mediapipe_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full-range model
            min_detection_confidence=0.2  # Very low threshold
        )
        self.detection_methods.append('mediapipe')
        print("✓ MediaPipe face detector initialized")
        
        # Method 2: RetinaFace (best accuracy)
        self.retinaface = None
        if RETINAFACE_AVAILABLE:
            try:
                self.retinaface = FaceAnalysis(providers=['CPUExecutionProvider'])
                self.retinaface.prepare(ctx_id=0, det_size=(640, 640))
                self.detection_methods.append('retinaface')
                print("✓ RetinaFace detector initialized")
            except Exception as e:
                print(f"✗ RetinaFace failed: {e}")
        
        # Method 3: MTCNN (good for small faces)
        self.mtcnn = None
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn = MTCNN(min_face_size=40, steps_threshold=[0.6, 0.7, 0.7])
                self.detection_methods.append('mtcnn')
                print("✓ MTCNN detector initialized")
            except Exception as e:
                print(f"✗ MTCNN failed: {e}")
        
        # Method 4: YOLOv8 (if available)
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                # Try to load a face detection model or use person detection
                self.yolo_model = YOLO('yolov8n.pt')
                self.detection_methods.append('yolo')
                print("✓ YOLOv8 detector initialized")
            except Exception as e:
                print(f"✗ YOLOv8 failed: {e}")
        
        # Method 5: dlib HOG (fallback)
        self.dlib_detector = None
        if DLIB_AVAILABLE:
            try:
                self.dlib_detector = dlib.get_frontal_face_detector()
                self.detection_methods.append('dlib')
                print("✓ dlib HOG detector initialized")
            except Exception as e:
                print(f"✗ dlib failed: {e}")
        
        print(f"Active detection methods: {self.detection_methods}")
        
        # For face landmarks (used by gaze module)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
    
    def detect_faces_mediapipe(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MediaPipe."""
        detections = []
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = self.mediapipe_detector.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                score = detection.score[0]
                if score >= self.min_confidence:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    if width > 30 and height > 30:
                        detections.append({
                            'bbox': (x, y, width, height),
                            'score': float(score),
                            'method': 'mediapipe'
                        })
        return detections
    
    def detect_faces_retinaface(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using RetinaFace."""
        detections = []
        if self.retinaface is None:
            return detections
        
        try:
            faces = self.retinaface.get(frame)
            h, w = frame.shape[:2]
            for face in faces:
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                width = x2 - x
                height = y2 - y
                score = face.det_score
                
                if score >= self.min_confidence and width > 30 and height > 30:
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    detections.append({
                        'bbox': (x, y, width, height),
                        'score': float(score),
                        'method': 'retinaface'
                    })
        except (AttributeError, RuntimeError, ValueError) as e:
            # Silently fail for expected errors (model not loaded, wrong format, etc.)
            # These are non-critical since we have fallback detection methods
            pass
        
        return detections
    
    def detect_faces_mtcnn(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MTCNN."""
        detections = []
        if self.mtcnn is None:
            return detections
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mtcnn.detect_faces(rgb_frame)
            
            if results:
                h, w = frame.shape[:2]
                for result in results:
                    if result['confidence'] >= self.min_confidence:
                        x, y, width, height = result['box']
                        x = max(0, x)
                        y = max(0, y)
                        width = min(width, w - x)
                        height = min(height, h - y)
                        
                        if width > 30 and height > 30:
                            detections.append({
                                'bbox': (x, y, width, height),
                                'score': float(result['confidence']),
                                'method': 'mtcnn'
                            })
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            # Silently fail for expected errors (model not loaded, wrong format, etc.)
            # These are non-critical since we have fallback detection methods
            pass
        
        return detections
    
    def detect_faces_dlib(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using dlib HOG."""
        detections = []
        if self.dlib_detector is None:
            return detections
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.dlib_detector(gray, 1)  # Upsample once
            
            h, w = frame.shape[:2]
            for face in faces:
                x = face.left()
                y = face.top()
                width = face.width()
                height = face.height()
                
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width > 30 and height > 30:
                    detections.append({
                        'bbox': (x, y, width, height),
                        'score': 0.8,  # dlib doesn't provide confidence
                        'method': 'dlib'
                    })
        except (AttributeError, RuntimeError, ValueError) as e:
            # Silently fail for expected errors (model not loaded, wrong format, etc.)
            # These are non-critical since we have fallback detection methods
            pass
        
        return detections
    
    def _merge_detections(self, all_detections: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Merge detections from multiple methods using NMS-like approach."""
        if not all_detections:
            return []
        
        # Flatten all detections
        flat_detections = []
        for det_list in all_detections:
            flat_detections.extend(det_list)
        
        if not flat_detections:
            return []
        
        # Group overlapping detections
        merged = []
        used = set()
        
        for i, det1 in enumerate(flat_detections):
            if i in used:
                continue
            
            x1, y1, w1, h1 = det1['bbox']
            group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(flat_detections[i+1:], i+1):
                if j in used:
                    continue
                
                x2, y2, w2, h2 = det2['bbox']
                
                # Calculate IoU (Intersection over Union)
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap
                
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area
                
                if union_area > 0:
                    iou = overlap_area / union_area
                    if iou > 0.3:  # If boxes overlap significantly
                        group.append(det2)
                        used.add(j)
            
            # Average the bboxes in the group, take max confidence
            if group:
                avg_x = int(np.mean([d['bbox'][0] for d in group]))
                avg_y = int(np.mean([d['bbox'][1] for d in group]))
                avg_w = int(np.mean([d['bbox'][2] for d in group]))
                avg_h = int(np.mean([d['bbox'][3] for d in group]))
                max_score = max([d['score'] for d in group])
                methods = ','.join(set([d['method'] for d in group]))
                
                merged.append({
                    'bbox': (avg_x, avg_y, avg_w, avg_h),
                    'score': max_score,
                    'method': methods,
                    'voted_by': len(group)  # How many methods agreed
                })
        
        # Sort by confidence and votes
        merged.sort(key=lambda x: (x['voted_by'], x['score']), reverse=True)
        
        return merged
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using ensemble of multiple models.
        
        Args:
            frame: BGR image frame
            
        Returns:
            List of merged face detections
        """
        all_detections = []
        
        # Try all available methods
        if 'mediapipe' in self.detection_methods:
            mp_dets = self.detect_faces_mediapipe(frame)
            if mp_dets:
                all_detections.append(mp_dets)
        
        if 'retinaface' in self.detection_methods:
            rf_dets = self.detect_faces_retinaface(frame)
            if rf_dets:
                all_detections.append(rf_dets)
        
        if 'mtcnn' in self.detection_methods:
            mtcnn_dets = self.detect_faces_mtcnn(frame)
            if mtcnn_dets:
                all_detections.append(mtcnn_dets)
        
        if 'dlib' in self.detection_methods:
            dlib_dets = self.detect_faces_dlib(frame)
            if dlib_dets:
                all_detections.append(dlib_dets)
        
        # Merge detections from all methods
        merged = self._merge_detections(all_detections)
        
        return merged
    
    def get_face_landmarks(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Get face landmarks using MediaPipe Face Mesh."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            landmarks = []
            bbox_coords = []
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
                bbox_coords.append((x, y))
            
            if bbox_coords:
                xs = [p[0] for p in bbox_coords]
                ys = [p[1] for p in bbox_coords]
                x = max(0, min(xs))
                y = max(0, min(ys))
                width = min(w - x, max(xs) - x)
                height = min(h - y, max(ys) - y)
                
                return {
                    'landmarks': landmarks,
                    'bbox': (x, y, width, height)
                }
        
        return None
    
    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                  padding: int = 20) -> Optional[np.ndarray]:
        """Crop face region from frame."""
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]
        
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_frame, x + w + padding)
        y2 = min(h_frame, y + h + padding)
        
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2]
        return None
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes with method info."""
        for det in detections:
            x, y, w, h = det['bbox']
            score = det['score']
            method = det.get('method', 'unknown')
            votes = det.get('voted_by', 1)
            
            # Color based on votes (more methods agreeing = more reliable)
            if votes >= 2:
                color = (0, 255, 0)  # Green - high confidence
            else:
                color = (0, 255, 255)  # Yellow - single method
            
            # Draw thick rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Draw label with method info
            label = f"Face: {score:.2f} ({method})"
            if votes > 1:
                label += f" [{votes} methods]"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x, y - text_height - 10), 
                        (x + text_width, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'mediapipe_detector'):
            self.mediapipe_detector.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
