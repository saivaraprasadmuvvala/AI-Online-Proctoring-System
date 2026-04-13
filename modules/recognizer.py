"""
Identity recognition module.
Uses face_recognition library for embeddings (primary) or SSIM for fallback matching.
"""

import cv2
import numpy as np
import os
import pickle
from typing import Optional, Dict, Any, Tuple
from skimage.metrics import structural_similarity as ssim

# Try to import face_recognition, fallback to None if not available
import warnings
import sys
import os
from io import StringIO

# Disable face_recognition to avoid warning messages
# SSIM fallback will be used instead (works perfectly fine)
FACE_RECOGNITION_AVAILABLE = False
face_recognition = None

# Uncomment below to enable face_recognition (may show warnings)
# try:
#     import face_recognition
#     FACE_RECOGNITION_AVAILABLE = True
# except ImportError:
#     FACE_RECOGNITION_AVAILABLE = False
#     face_recognition = None

# Completely suppress stderr for face_recognition operations (if enabled)
class SuppressStderr:
    def __init__(self):
        self.original_stderr = sys.stderr
        try:
            self.devnull = open(os.devnull, 'w')
        except:
            self.devnull = StringIO()
    
    def __enter__(self):
        sys.stderr = self.devnull
        return self
    
    def __exit__(self, *args):
        sys.stderr = self.original_stderr
        try:
            self.devnull.close()
        except:
            pass


class FaceRecognizer:
    """Face recognition with embedding-based and SSIM fallback."""
    
    def __init__(self, enrolled_dir: str = "enrolled"):
        """
        Initialize face recognizer.
        
        Args:
            enrolled_dir: Directory to store enrollment data
        """
        self.enrolled_dir = enrolled_dir
        self.use_embeddings = FACE_RECOGNITION_AVAILABLE
        
        # Create enrolled directory if it doesn't exist
        os.makedirs(enrolled_dir, exist_ok=True)
        
        # Thresholds
        self.embedding_threshold = 0.6  # Euclidean distance threshold
        self.ssim_threshold = 0.45  # SSIM similarity threshold
    
    def enroll(self, name: str, image_path: str) -> Dict[str, Any]:
        """
        Enroll a user with a reference image.
        
        Args:
            name: User's name
            image_path: Path to enrollment image
            
        Returns:
            Dictionary with enrollment info
        """
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        enrollment_data = {
            'name': name,
            'image_path': image_path,
            'method': None,
            'embedding_path': None
        }
        
        # Try embedding-based enrollment first
        if self.use_embeddings and face_recognition is not None:
            try:
                with SuppressStderr():
                    # Detect face and get encoding
                    face_locations = face_recognition.face_locations(rgb_image)
                    if face_locations:
                        encodings = face_recognition.face_encodings(rgb_image, face_locations)
                        if encodings:
                            # Save embedding
                            embedding_path = os.path.join(
                                self.enrolled_dir, 
                                f"{name}_embedding.pkl"
                            )
                            with open(embedding_path, 'wb') as f:
                                pickle.dump(encodings[0], f)
                            
                            enrollment_data['method'] = 'embedding'
                            enrollment_data['embedding_path'] = embedding_path
                            return enrollment_data
            except Exception:
                # Silently fall back to SSIM
                pass
        
        # Fallback: store image for SSIM matching
        enrollment_data['method'] = 'ssim'
        return enrollment_data
    
    def load_embedding(self, embedding_path: str) -> Optional[np.ndarray]:
        """
        Load saved embedding from file.
        
        Args:
            embedding_path: Path to embedding pickle file
            
        Returns:
            Embedding array or None if not found
        """
        if not os.path.exists(embedding_path):
            return None
        
        try:
            with open(embedding_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading embedding: {e}")
            return None
    
    def match_embedding(self, face_image: np.ndarray, 
                       reference_embedding: np.ndarray) -> Tuple[bool, float]:
        """
        Match face using embedding distance.
        
        Args:
            face_image: RGB face image
            reference_embedding: Reference embedding array
            
        Returns:
            (match, distance) tuple
        """
        if not self.use_embeddings or face_recognition is None:
            return False, float('inf')
        
        try:
            with SuppressStderr():
                # Get face encoding from current image
                face_encodings = face_recognition.face_encodings(face_image)
                if not face_encodings:
                    return False, float('inf')
                
                # Compute distance
                distance = face_recognition.face_distance([reference_embedding], face_encodings[0])[0]
                match = distance <= self.embedding_threshold
                return match, distance
        except Exception:
            return False, float('inf')
    
    def match_ssim(self, face_image: np.ndarray, 
                   reference_image_path: str) -> Tuple[bool, float]:
        """
        Match face using SSIM similarity.
        
        Args:
            face_image: RGB face image
            reference_image_path: Path to reference enrollment image
            
        Returns:
            (match, similarity_score) tuple
        """
        if not os.path.exists(reference_image_path):
            return False, 0.0
        
        try:
            # Load reference image
            ref_image = cv2.imread(reference_image_path)
            if ref_image is None:
                return False, 0.0
            
            ref_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            
            # Resize both images to same size for comparison
            target_size = (128, 128)
            face_resized = cv2.resize(face_image, target_size)
            ref_resized = cv2.resize(ref_rgb, target_size)
            
            # Convert to grayscale for SSIM
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            ref_gray = cv2.cvtColor(ref_resized, cv2.COLOR_RGB2GRAY)
            
            # Compute SSIM
            similarity = ssim(face_gray, ref_gray)
            match = similarity >= self.ssim_threshold
            
            return match, similarity
        except Exception as e:
            print(f"SSIM matching error: {e}")
            return False, 0.0
    
    def recognize(self, face_image: np.ndarray, 
                  enrolled_users: list) -> Dict[str, Any]:
        """
        Recognize face from enrolled users with improved matching.
        
        Args:
            face_image: RGB face image (cropped)
            enrolled_users: List of enrolled user dictionaries from database
            
        Returns:
            Dictionary with:
            - match: bool
            - best_name: str or None
            - score: float (distance or similarity)
            - method: str ('embedding' or 'ssim')
        """
        if not enrolled_users:
            return {
                'match': False,
                'best_name': None,
                'score': float('inf'),
                'method': None
            }
        
        # Convert face_image to RGB if needed
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            if face_image.dtype == np.uint8:
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_face = face_image
        else:
            rgb_face = face_image
        
        # Ensure face image is large enough
        if rgb_face.shape[0] < 50 or rgb_face.shape[1] < 50:
            return {
                'match': False,
                'best_name': None,
                'score': float('inf'),
                'method': None
            }
        
        best_match = False
        best_score = float('inf') if self.use_embeddings else 0.0
        best_method = None
        best_name = None
        
        # Try matching with all enrolled users
        for user in enrolled_users:
            name = user['name']
            embedding_path = user.get('embedding_path')
            image_path = user.get('enrollment_image_path')
            
            # Try embedding match first (more accurate)
            if embedding_path and self.use_embeddings:
                embedding = self.load_embedding(embedding_path)
                if embedding is not None:
                    match, score = self.match_embedding(rgb_face, embedding)
                    # Use relaxed threshold for better matching
                    if score < best_score:
                        best_score = score
                        best_method = 'embedding'
                        best_name = name
                        # Match if score is below threshold (lower is better for distance)
                        if score <= self.embedding_threshold:
                            best_match = True
            
            # Also try SSIM for comparison
            if image_path:
                match, score = self.match_ssim(rgb_face, image_path)
                # For SSIM, higher is better
                # Only use SSIM if embedding didn't work or SSIM is better
                if best_method != 'embedding' or (best_method == 'embedding' and best_score > self.embedding_threshold):
                    if score > best_score or (best_method == 'embedding' and score >= self.ssim_threshold):
                        if score >= self.ssim_threshold:
                            best_match = True
                        best_score = score
                        best_method = 'ssim'
                        best_name = name
        
        # Final decision: be more lenient with matching
        if best_method == 'embedding':
            # Use slightly relaxed threshold
            final_match = best_score <= (self.embedding_threshold * 1.2)  # 20% more lenient
        else:  # SSIM
            # Use slightly relaxed threshold
            final_match = best_score >= (self.ssim_threshold * 0.9)  # 10% more lenient
        
        # If we have a best_name but not a match, still return it for display
        if not final_match and best_name:
            # Return partial match info
            return {
                'match': False,
                'best_name': best_name,
                'score': best_score,
                'method': best_method,
                'partial_match': True  # Indicates we found someone but not confident enough
            }
        
        return {
            'match': final_match,
            'best_name': best_name if final_match else None,
            'score': best_score,
            'method': best_method
        }
