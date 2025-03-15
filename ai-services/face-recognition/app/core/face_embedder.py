import os
import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any, Optional, Union
import time
import uuid

# ในสถานการณ์จริง คุณจะต้องดาวน์โหลดโมเดล FaceNet 20180402-114759 และเก็บไว้ในโฟลเดอร์ที่เหมาะสม
# URL: https://github.com/davidsandberg/facenet/tree/master/src/models/20180402-114759

class FaceEmbedder:
    """
    Face embedder class using FaceNet for generating face embeddings.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the face embedder.
        
        Parameters:
        - model_path: Path to FaceNet model. If None, uses a placeholder model for development.
        """
        # Set memory growth for GPU to avoid allocating all memory at once
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load the FaceNet model
        if model_path and os.path.exists(model_path):
            print(f"Loading FaceNet model from {model_path}")
            self.model = tf.saved_model.load(model_path)
        else:
            print("Model path not provided or does not exist. Using a placeholder model.")
            # Create a placeholder model for development
            self._create_placeholder_model()
        
        # Set the input shape required by the model
        self.input_shape = (160, 160)
    
    def _create_placeholder_model(self):
        """
        Create a placeholder model for development and testing.
        This generates random embeddings with the correct dimensionality.
        """
        # Create a simple model that returns random embeddings
        inputs = tf.keras.Input(shape=(160, 160, 3))
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(128, activation=None)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print("Created placeholder FaceNet model.")
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a face image for the FaceNet model.
        
        Parameters:
        - face_image: Input face image
        
        Returns:
        - Preprocessed face image
        """
        if face_image is None:
            raise ValueError("Face image is None")
        
        # Resize image to model input size
        if face_image.shape[:2] != self.input_shape:
            face_image = cv2.resize(face_image, self.input_shape, interpolation=cv2.INTER_CUBIC)
        
        # Ensure the image has 3 channels
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        elif face_image.shape[2] == 4:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
        
        # Convert to float and scale pixel values to [0, 1]
        face_image = face_image.astype(np.float32) / 255.0
        
        # Normalize the image (subtract mean and divide by std)
        mean = np.mean(face_image)
        std = np.std(face_image)
        face_image = (face_image - mean) / std
        
        return face_image
    
    def generate_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate a face embedding from a face image.
        
        Parameters:
        - face_image: Input face image
        
        Returns:
        - Face embedding vector
        """
        # Preprocess the face
        preprocessed_face = self.preprocess_face(face_image)
        
        # Add batch dimension
        input_tensor = np.expand_dims(preprocessed_face, axis=0)
        
        # Generate embedding
        if isinstance(self.model, tf.keras.Model):
            # For the placeholder model
            embedding = self.model.predict(input_tensor)[0]
        else:
            # For the loaded SavedModel
            embedding = self.model(input_tensor)[0]
        
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two face embeddings.
        
        Parameters:
        - embedding1: First face embedding
        - embedding2: Second face embedding
        
        Returns:
        - Similarity score (0-1, higher is more similar)
        """
        # Ensure embeddings are normalized
        if np.linalg.norm(embedding1) > 0:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
        if np.linalg.norm(embedding2) > 0:
            embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        # Ensure the similarity is between 0 and 1
        similarity = max(0, min(1, similarity))
        
        return float(similarity)
    
    def assess_quality(self, face_image: np.ndarray) -> float:
        """
        Assess the quality of a face image.
        
        Parameters:
        - face_image: Input face image
        
        Returns:
        - Quality score (0-1, higher is better)
        """
        if face_image is None:
            return 0.0
        
        # Get image dimensions
        height, width = face_image.shape[:2]
        
        # Check image resolution (higher is better)
        resolution_score = min(1.0, (height * width) / (250 * 250))
        
        # Check sharpness (higher is better)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, sharpness / 500)
        
        # Check brightness (middle values are better)
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - 2.0 * abs(brightness - 0.5)
        
        # Check contrast (higher is better, up to a point)
        contrast = np.std(gray) / 255.0
        contrast_score = min(1.0, contrast * 3.0)
        
        # Combine scores with weights
        quality_score = (
            0.3 * resolution_score +
            0.3 * sharpness_score +
            0.2 * brightness_score +
            0.2 * contrast_score
        )
        
        return float(quality_score)