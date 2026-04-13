"""
Object detection module using YOLOv8 to detect prohibited objects.
Detects phones, books, papers, etc. during exams.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from ultralytics import YOLO


class ObjectDetector:
    """YOLOv8-based object detector for prohibited items."""
    
    # Prohibited object classes in COCO dataset
    PROHIBITED_CLASSES = {
        'cell phone': 67,
        'book': 73,  # Not in COCO, but we'll use custom
        'laptop': 63,
        'mouse': 64,
        'keyboard': 66,
        'remote': 65,
    }
    
    def __init__(self, model_size: str = 'nano', config: Optional[Dict[str, Any]] = None):
        """
        Initialize YOLOv8 object detector.
        
        Args:
            model_size: Model size ('nano', 'small', 'medium', 'large')
            config: Configuration dictionary (optional)
        """
        try:
            model_name = f'yolov8{model_size[0]}.pt'  # yolov8n.pt, yolov8s.pt, etc.
            self.model = YOLO(model_name)
            self.available = True
            print(f"✓ YOLOv8 object detector initialized ({model_name})")
        except Exception as e:
            print(f"✗ YOLOv8 object detector failed: {e}")
            self.model = None
            self.available = False
        
        # Load config if provided
        if config:
            objects_config = config.get('detection', {}).get('objects', {})
            self.min_confidence = objects_config.get('min_confidence', 0.5)
        else:
            self.min_confidence = 0.5
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect prohibited objects in frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            List of detected prohibited objects
        """
        if not self.available or self.model is None:
            return []
        
        prohibited_objects = []
        
        try:
            # Run YOLOv8 detection
            results = self.model(frame, conf=self.min_confidence, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = result.names[cls_id]
                        
                        # Check if it's a prohibited object
                        if class_name.lower() in ['cell phone', 'phone', 'mobile phone']:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            prohibited_objects.append({
                                'class': 'cell_phone',
                                'class_name': class_name,
                                'confidence': conf,
                                'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                                'threat_level': 'high'
                            })
                        
                        # Detect books/papers (using general object detection)
                        elif class_name.lower() in ['book', 'notebook'] or cls_id == 73:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            prohibited_objects.append({
                                'class': 'book',
                                'class_name': class_name,
                                'confidence': conf,
                                'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                                'threat_level': 'high'
                            })
                        
                        # Detect laptops/tablets
                        elif class_name.lower() in ['laptop', 'tablet']:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            prohibited_objects.append({
                                'class': 'electronic_device',
                                'class_name': class_name,
                                'confidence': conf,
                                'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                                'threat_level': 'high'
                            })
        
        except Exception as e:
            pass  # Silently fail
        
        return prohibited_objects
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw prohibited object detections on frame."""
        for det in detections:
            x, y, w, h = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']
            threat = det['threat_level']
            
            # Red color for prohibited objects
            color = (0, 0, 255)  # Red
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Draw label
            label = f"⚠️ {class_name}: {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(frame, (x, y - text_height - 10),
                        (x + text_width, y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

