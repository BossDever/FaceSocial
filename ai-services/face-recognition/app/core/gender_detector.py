import cv2
import numpy as np
from typing import Tuple, Optional
import os

class GenderDetector:
    """
    Gender detector class to predict gender from face images
    Uses a simple model or heuristics to determine gender
    """
    
    def __init__(self):
        """
        Initialize gender detector with pre-trained models if available
        """
        self.model_loaded = False
        
        # Try to load pre-trained caffe model for gender detection
        model_dir = '/app/models/gender'
        proto_paths = [
            '/app/models/gender/gender_deploy.prototxt',
            '/app/app/models/gender/gender_deploy.prototxt',
            './app/models/gender/gender_deploy.prototxt'
        ]
        
        model_paths = [
            '/app/models/gender/gender_net.caffemodel',
            '/app/app/models/gender/gender_net.caffemodel',
            './app/models/gender/gender_net.caffemodel'
        ]
        
        # Try to find prototxt and model files
        proto_file = None
        model_file = None
        
        for path in proto_paths:
            if os.path.exists(path):
                proto_file = path
                break
                
        for path in model_paths:
            if os.path.exists(path):
                model_file = path
                break
        
        # If both files found, load the model
        if proto_file and model_file:
            try:
                print(f"Loading gender detection model from: {model_file}")
                self.gender_net = cv2.dnn.readNet(proto_file, model_file)
                self.model_loaded = True
                print("Gender detection model loaded successfully")
            except Exception as e:
                print(f"Failed to load gender detection model: {str(e)}")
                self.model_loaded = False
        else:
            print("Gender detection model files not found, will use fallback method")
            
    def predict_gender(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Predict gender from face image
        
        Parameters:
        - face_image: Input face image
        
        Returns:
        - Tuple of (predicted_gender, confidence)
        """
        if face_image is None or face_image.size == 0:
            return "unknown", 0.0
            
        # Use pre-trained model if available
        if self.model_loaded:
            try:
                # Preprocess image for the model
                blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227),
                                           (78.4263377603, 87.7689143744, 114.895847746),
                                           swapRB=False)
                
                # Predict gender
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                
                # Get result
                gender_idx = gender_preds[0].argmax()
                confidence = gender_preds[0][gender_idx]
                
                gender = "male" if gender_idx == 1 else "female"
                return gender, float(confidence)
            except Exception as e:
                print(f"Error in gender prediction with model: {str(e)}")
                # Fall back to heuristic method
        
        # Fallback: Use simple facial features for gender estimation
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate face proportions
            height, width = face_image.shape[:2]
            aspect_ratio = width / height
            
            # Simple heuristics for demonstration
            if aspect_ratio > 0.95:  # More square face shape (typically male)
                gender = 'male'
                confidence = 0.55 + (aspect_ratio - 0.95) / 0.5  # Adjust confidence based on ratio
            else:  # More oval face shape (typically female)
                gender = 'female'
                confidence = 0.55 + (0.95 - aspect_ratio) / 0.5
            
            # Ensure confidence is between 0.5 and 0.85
            confidence = min(max(confidence, 0.5), 0.85)
            
            return gender, confidence
            
        except Exception as e:
            print(f"Error in fallback gender prediction: {str(e)}")
            return "unknown", 0.0