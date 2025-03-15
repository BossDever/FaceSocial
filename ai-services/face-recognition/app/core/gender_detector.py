# เพิ่มในไฟล์ app/core/gender_detector.py (สร้างไฟล์ใหม่)
import cv2
import numpy as np
import tensorflow as tf
import os

class GenderDetector:
    """
    A class for gender detection using a pre-trained model
    """
    def __init__(self):
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load gender detection model"""
        try:
            # พยายามโหลดโมเดล keras ที่ train ไว้สำหรับตรวจจับเพศ
            model_path = '/app/app/models/gender/gender_model.h5'
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"Loaded gender detection model from {model_path}")
            else:
                # ถ้าไม่มีโมเดล ใช้โมเดลอย่างง่าย
                self._create_simple_model()
                print("Using simplified gender detection model")
        except Exception as e:
            print(f"Error loading gender model: {str(e)}")
            self._create_simple_model()
            
    def _create_simple_model(self):
        """Create a simple gender detection CNN"""
        inputs = tf.keras.Input(shape=(64, 64, 3))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs, outputs)
        
    def predict_gender(self, face_image):
        """
        Predict gender from face image
        
        Parameters:
        - face_image: Input face image
        
        Returns:
        - Tuple of (predicted_gender, confidence)
        """
        if self.model is None:
            # ถ้าไม่มีโมเดล ให้ใช้วิธีวิเคราะห์แบบเดิม (facial features)
            return self._analyze_facial_features(face_image)
            
        # Preprocess image
        img = cv2.resize(face_image, (64, 64))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        try:
            pred = self.model.predict(img, verbose=0)[0][0]
            gender = 'male' if pred < 0.5 else 'female'
            # Convert to confidence (0-1)
            confidence = 1 - pred if pred < 0.5 else pred
            return gender, float(confidence)
        except Exception as e:
            print(f"Error predicting gender: {str(e)}")
            return self._analyze_facial_features(face_image)
            
    def _analyze_facial_features(self, face_image):
        """
        Analyze facial features for gender detection (fallback method)
        """
        try:
            # ทำ preprocessing เพื่อให้จับลักษณะของใบหน้าได้ชัดเจนขึ้น
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # ตัวแปรที่บ่งชี้เพศ
            h, w = face_image.shape[:2]
            aspect_ratio = w / h  # อัตราส่วนความกว้างต่อความสูง
            
            # ลองใช้ face landmark detector ถ้ามี
            try:
                face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_detector.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_area = face_image[y:y+h, x:x+w]
                    # นับพิกเซลสีผิวในบริเวณขากรรไกร
                    lower = np.array([0, 20, 70], dtype="uint8")
                    upper = np.array([20, 100, 255], dtype="uint8")
                    skin_mask = cv2.inRange(cv2.cvtColor(face_area, cv2.COLOR_BGR2HSV), lower, upper)
                    skin_pixels = cv2.countNonZero(skin_mask)
                    face_pixels = w * h
                    skin_ratio = skin_pixels / face_pixels
                    # ถ้ามีสัดส่วนพิกเซลสีผิวน้อยอาจเป็นเพราะมีหนวดเครา
                    has_beard = skin_ratio < 0.4
                    if has_beard:
                        return 'male', 0.85
            except:
                pass
                
            # คำนวณลักษณะอื่นๆ
            # 1. คำนวณความแตกต่างของสี (ผู้ชายมักมีความเข้มของสีมากกว่า)
            variance = np.var(gray)
            
            # 2. ดูอัตราส่วนใบหน้า
            face_ratio_score = 0.7 if aspect_ratio > 0.95 else 0.3
            
            # 3. คำนวณความเข้มของเฉดสี
            intensity_score = 0.7 if variance > 2000 else 0.3
            
            # รวมคะแนน
            male_score = (face_ratio_score + intensity_score) / 2
            
            # ตัดสินใจ
            gender = 'male' if male_score > 0.5 else 'female'
            confidence = male_score if gender == 'male' else (1 - male_score)
            
            return gender, confidence
            
        except Exception as e:
            print(f"Error in facial feature analysis: {str(e)}")
            # กรณีที่มีปัญหา ให้ตัดสินใจแบบไม่มั่นใจ
            return 'unknown', 0.51