import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import os
import tensorflow as tf
import glob

class GenderDetector:
    """
    Gender detector class to predict gender from face images
    Uses multiple models for improved accuracy
    """
    
    def __init__(self):
        """
        Initialize gender detector with pre-trained models if available
        """
        self.model_loaded = False
        self.keras_model_loaded = False
        
        # Extended search path for Keras gender model
        model_paths = self._find_model_paths([
            '/home/suwit/FaceSocial/ai-services/face-recognition/app/models/keras-facenet-h5/model.h5',
            '/app/app/models/gender/model.h5',
            '/app/models/gender/model.h5',
            './app/models/gender/model.h5',
            # Add recursive search in standard directories
            '/app/**/*gender*.h5',
            '/app/**/*face*.h5'
        ])
        
        # Try loading Keras gender model
        for path in model_paths:
            if os.path.exists(path):
                try:
                    print(f"Attempting to load gender detection model from: {path}")
                    self.keras_gender_model = tf.keras.models.load_model(path)
                    
                    # Verify model works by testing it
                    if self._verify_keras_model():
                        self.keras_model_loaded = True
                        print(f"✓ Gender detection model loaded successfully from: {path}")
                        break
                    else:
                        print(f"✗ Model at {path} loaded but failed verification")
                        # Continue trying other models
                except Exception as e:
                    print(f"✗ Failed to load gender model from {path}: {str(e)}")
        
        if not self.keras_model_loaded:
            print("⚠ No Keras gender detection model loaded successfully")
        
        # Try to load Caffe model as backup
        proto_paths = self._find_model_paths([
            '/app/models/gender/gender_deploy.prototxt',
            '/app/app/models/gender/gender_deploy.prototxt',
            './app/models/gender/gender_deploy.prototxt',
            '/app/**/gender*.prototxt'
        ])
        
        model_files = self._find_model_paths([
            '/app/models/gender/gender_net.caffemodel',
            '/app/app/models/gender/gender_net.caffemodel',
            './app/models/gender/gender_net.caffemodel',
            '/app/**/gender*.caffemodel'
        ])
        
        # Try to find prototxt and model files
        proto_file = next((p for p in proto_paths if os.path.exists(p)), None)
        model_file = next((p for p in model_files if os.path.exists(p)), None)
        
        # If both files found, load the model
        if proto_file and model_file:
            try:
                print(f"Loading Caffe gender detection model:\n  - Proto: {proto_file}\n  - Model: {model_file}")
                self.gender_net = cv2.dnn.readNet(proto_file, model_file)
                
                # Test if model works
                if self._verify_caffe_model():
                    self.model_loaded = True
                    print("✓ Caffe gender detection model loaded and verified")
                else:
                    print("✗ Caffe model loaded but failed verification")
            except Exception as e:
                print(f"✗ Failed to load Caffe gender detection model: {str(e)}")
                self.model_loaded = False
        else:
            print("⚠ Caffe gender detection model files not found")
            
        # Create a logger to report status
        self._log_status()
            
    def _find_model_paths(self, path_patterns: List[str]) -> List[str]:
        """Find model files matching the given patterns with glob"""
        found_paths = []
        for pattern in path_patterns:
            if '**' in pattern:  # Recursive glob pattern
                found_paths.extend(glob.glob(pattern, recursive=True))
            else:  # Direct path
                found_paths.append(pattern)
        return found_paths
    
    def _verify_keras_model(self) -> bool:
        """Verify Keras model works by testing it with a dummy input"""
        try:
            # Create a dummy input of appropriate shape
            if not hasattr(self, 'keras_gender_model'):
                return False
                
            input_shape = self.keras_gender_model.input_shape
            if input_shape is None:
                return False
                
            # Remove batch dimension if it's None
            if input_shape[0] is None:
                input_shape = input_shape[1:]
                
            # Create a sample input - all zeros
            dummy_input = np.zeros((1,) + input_shape)
            
            # Try prediction
            _ = self.keras_gender_model.predict(dummy_input, verbose=0)
            return True
        except Exception as e:
            print(f"Keras model verification failed: {str(e)}")
            return False
            
    def _verify_caffe_model(self) -> bool:
        """Verify Caffe model works by testing it with a dummy input"""
        try:
            if not hasattr(self, 'gender_net'):
                return False
                
            # Create a dummy blob image (3x227x227)
            dummy_blob = np.zeros((1, 3, 227, 227), dtype=np.float32)
            
            # Try inference
            self.gender_net.setInput(dummy_blob)
            _ = self.gender_net.forward()
            return True
        except Exception as e:
            print(f"Caffe model verification failed: {str(e)}")
            return False
            
    def _log_status(self):
        """Log the status of loaded models"""
        status = {
            "keras_model": "Loaded ✓" if self.keras_model_loaded else "Not loaded ✗",
            "caffe_model": "Loaded ✓" if self.model_loaded else "Not loaded ✗",
            "fallback_method": "Enabled ✓" if not (self.keras_model_loaded or self.model_loaded) else "Disabled (using models) ✗"
        }
        
        print("\n=== Gender Detection Status ===")
        for k, v in status.items():
            print(f"  {k}: {v}")
        print("============================\n")
            
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
        
        # (1) ใช้โมเดล Keras ถ้าโหลดได้สำเร็จ (ความแม่นยำสูงกว่า)
        if self.keras_model_loaded:
            try:
                # Preprocess image for the Keras model
                resized = cv2.resize(face_image, (96, 96), interpolation=cv2.INTER_AREA)
                
                # Ensure RGB format
                if len(resized.shape) == 2:
                    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                elif resized.shape[2] == 4:
                    resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
                
                # Normalize pixel values
                normalized = resized / 255.0
                
                # Add batch dimension
                input_img = np.expand_dims(normalized, axis=0)
                
                # Predict gender
                predictions = self.keras_gender_model.predict(input_img, verbose=0)
                
                # If model output has multiple values, assume first one is gender
                # (usually models output [gender, age, ...])
                gender_prob = predictions[0][0] if isinstance(predictions, list) else predictions[0]
                
                # Interpret probability (common convention: >0.5 is female, <0.5 is male)
                gender = "female" if gender_prob > 0.5 else "male"
                confidence = gender_prob if gender == "female" else 1.0 - gender_prob
                
                return gender, float(confidence)
            except Exception as e:
                print(f"Error in Keras gender prediction: {str(e)}")
                # Fall back to next method
        
        # (2) ใช้โมเดล Caffe ถ้าโหลดได้สำเร็จ
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
        
        # (3) วิธีสำรองเมื่อไม่มีโมเดล: ใช้การวิเคราะห์ลักษณะใบหน้าอย่างง่าย
        try:
            # Facial feature-based gender detection (simple heuristics)
            height, width = face_image.shape[:2]
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # คำนวณอัตราส่วนของใบหน้า (สัดส่วนความกว้าง/ความสูง)
            face_ratio = width / height
            
            # ตรวจสอบความเข้มของสีผิว (ผู้ชายมักจะมีสีผิวเข้มกว่า)
            avg_intensity = np.mean(gray)
            
            # ตรวจหา Jawline (กรามของผู้ชายมักจะใหญ่กว่า)
            edges = cv2.Canny(gray, 100, 200)
            lower_half = edges[height//2:, :]
            jawline_strength = np.sum(lower_half) / (width * height / 2)
            
            # คำนวณคะแนนความเป็นผู้ชาย (ยิ่งมากยิ่งเป็นผู้ชาย)
            male_score = 0.0
            
            # กรามใหญ่ → มักเป็นผู้ชาย
            if jawline_strength > 10:
                male_score += 0.2
            
            # ใบหน้าเป็นสี่เหลี่ยมมากกว่า → มักเป็นผู้ชาย
            if face_ratio > 0.85:
                male_score += 0.2
            
            # ความเข้มของสีผิว → ผู้ชายมักมีสีผิวเข้มกว่า
            if avg_intensity < 120:
                male_score += 0.1
            
            # ตัดสินใจและคำนวณความเชื่อมั่น
            if male_score >= 0.3:
                return "male", min(0.55 + male_score, 0.85)
            else:
                return "female", min(0.55 + (0.5 - male_score), 0.85)
                
        except Exception as e:
            print(f"Error in fallback gender prediction: {str(e)}")
            return "unknown", 0.0