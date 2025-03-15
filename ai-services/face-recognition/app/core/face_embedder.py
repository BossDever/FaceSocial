import os
import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any, Optional, Union
import time
import uuid

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
        
        # Check if model path is provided and exists
        if model_path is None:
            # Try to find the model in standard locations
            standard_paths = [
                "/app/models/facenet/20180402-114759.pb",
                "/app/app/models/facenet/20180402-114759.pb",
                "/home/suwit/FaceSocial/ai-services/face-recognition/app/models/facenet/20180402-114759.pb"
            ]
            
            for path in standard_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        self.model_path = model_path
        print(f"FaceNet model path: {model_path}")
        
        # Load the FaceNet model
        if model_path and os.path.exists(model_path):
            # Load SavedModel or frozen graph depending on the file
            if model_path.endswith('.pb'):
                print(f"Loading FaceNet frozen model from {model_path}")
                self._load_frozen_graph(model_path)
            else:
                print(f"Loading FaceNet checkpoint from {model_path}")
                self._load_checkpoint(model_path)
        else:
            print("Model path not provided or does not exist. Using a placeholder model.")
            # Create a placeholder model for development
            self._create_placeholder_model()
        
        # Set the input shape required by the model
        self.input_shape = (160, 160)
    
    def _load_frozen_graph(self, model_path):
        """
        Load a frozen graph TensorFlow model.
        """
        try:
            with tf.io.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            
            # Create a graph
            self.graph = tf.Graph()
            with self.graph.as_default():
                tf.import_graph_def(graph_def, name='')
                
                # Get input and output tensors
                self.input_tensor = self.graph.get_tensor_by_name('input:0')
                self.embeddings_tensor = self.graph.get_tensor_by_name('embeddings:0')
                self.phase_train_tensor = self.graph.get_tensor_by_name('phase_train:0')
                
                # Create a session
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.compat.v1.Session(graph=self.graph, config=config)
                
            print("FaceNet model loaded successfully (frozen graph)")
            self.model_type = 'frozen'
            
        except Exception as e:
            print(f"Error loading frozen graph: {str(e)}")
            self._create_placeholder_model()
    
    def _load_checkpoint(self, model_path):
        """
        Load a TensorFlow checkpoint model.
        """
        try:
            # Here we would implement the logic to load a checkpoint
            # For now, fall back to placeholder
            print("Checkpoint loading not fully implemented, using placeholder")
            self._create_placeholder_model()
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            self._create_placeholder_model()
    
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
        self.model_type = 'placeholder'
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
        
        # Standardize the image
        if self.model_type == 'frozen':
            # FaceNet expects standardized images
            standardized = (face_image - 127.5) / 128.0
            return standardized
        else:
            # For placeholder model
            face_image = face_image.astype(np.float32) / 255.0
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
        
        # Generate embedding
        if self.model_type == 'frozen':
            # Add batch dimension
            input_tensor = np.expand_dims(preprocessed_face, axis=0)
            
            # Generate embedding using the frozen graph
            feed_dict = {
                self.input_tensor: input_tensor,
                self.phase_train_tensor: False
            }
            embedding = self.sess.run(self.embeddings_tensor, feed_dict=feed_dict)[0]
        else:
            # For placeholder model
            input_tensor = np.expand_dims(preprocessed_face, axis=0)
            embedding = self.model.predict(input_tensor)[0]
        
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    # Rest of the methods remain unchanged
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
    def average_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Average multiple face embeddings.
        
        Parameters:
        - embeddings: List of face embeddings
        
        Returns:
        - Average embedding vector
        """
        if not embeddings:
            raise ValueError("No embeddings provided")
        
        # Ensure all embeddings are normalized
        normalized_embeddings = []
        for emb in embeddings:
            if np.linalg.norm(emb) > 0:
                normalized_embeddings.append(emb / np.linalg.norm(emb))
            else:
                normalized_embeddings.append(emb)
        
        # Calculate average
        avg_embedding = np.mean(normalized_embeddings, axis=0)
        
        # Normalize the average
        if np.linalg.norm(avg_embedding) > 0:
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return avg_embedding

    def calculate_similarity_with_multiple(self, embedding: np.ndarray, target_embeddings: List[np.ndarray]) -> float:
        """
        Calculate similarity between one embedding and multiple target embeddings.
        Returns the maximum similarity score.
        
        Parameters:
        - embedding: Face embedding to compare
        - target_embeddings: List of target face embeddings
        
        Returns:
        - Maximum similarity score (0-1, higher is more similar)
        """
        if not target_embeddings:
            return 0.0
        
        # Calculate similarity with each target embedding
        similarities = [self.calculate_similarity(embedding, target_emb) for target_emb in target_embeddings]
        
        # Return maximum similarity
        return max(similarities)

    def calculate_similarity_with_average(self, embedding: np.ndarray, target_embeddings: List[np.ndarray]) -> float:
        """
        Calculate similarity between one embedding and the average of multiple target embeddings.
        
        Parameters:
        - embedding: Face embedding to compare
        - target_embeddings: List of target face embeddings
        
        Returns:
        - Similarity score with the average embedding (0-1, higher is more similar)
        """
        if not target_embeddings:
            return 0.0
        
        # Calculate average embedding
        avg_embedding = self.average_embeddings(target_embeddings)
        
        # Calculate similarity with the average embedding
        return self.calculate_similarity(embedding, avg_embedding)

    def calculate_top_n_average_similarity(self, embedding: np.ndarray, target_embeddings: List[np.ndarray], top_n: int = 3) -> float:
        """
        Calculate similarity between one embedding and multiple target embeddings
        using Top-N Average method.
        
        Parameters:
        - embedding: Face embedding to compare
        - target_embeddings: List of target face embeddings
        - top_n: Number of highest similarity embeddings to use for average
        
        Returns:
        - Average similarity score of top N matches (0-1, higher is more similar)
        """
        if not target_embeddings:
            return 0.0
        
        # Calculate similarity with each target embedding
        similarities = [self.calculate_similarity(embedding, target_emb) for target_emb in target_embeddings]
        
        # Sort similarities in descending order and take top N
        similarities.sort(reverse=True)
        top_similarities = similarities[:min(top_n, len(similarities))]
        
        # Return average of top N similarities
        if top_similarities:
            return sum(top_similarities) / len(top_similarities)
        else:
            return 0.0
            
    def estimate_gender(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Estimate gender from face image.
        
        Parameters:
        - face_image: Input face image
        
        Returns:
        - Tuple of (predicted_gender, confidence)
        """
        # A simple gender estimation based on face features
        # In a real-world system, you would use a dedicated gender classification model
        # This is a placeholder implementation
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Extract simple features
        face_width = face_image.shape[1]
        face_height = face_image.shape[0]
        
        # Calculate basic face proportions
        ratio = face_width / face_height
        
        # Simple heuristics for demonstration
        # This should be replaced with a proper gender classification model
        if ratio > 0.95:  # More square face shape
            gender = 'male'
            confidence = 0.6 + (ratio - 0.95) / 0.5  # Adjust confidence based on ratio
        else:  # More oval face shape
            gender = 'female'
            confidence = 0.6 + (0.95 - ratio) / 0.5
        
        # Ensure confidence is between 0.5 and 1.0
        confidence = min(max(confidence, 0.5), 0.95)
        
        return gender, confidence

    def calculate_weighted_top_n_average_similarity(self, embedding: np.ndarray, 
                                                 target_embeddings: List[np.ndarray], 
                                                 target_images: List[np.ndarray] = None,
                                                 top_n: int = 3) -> Dict[str, Any]:
        """
        Calculate similarity using weighted Top-N Average method.
        
        Parameters:
        - embedding: Face embedding to compare
        - target_embeddings: List of target face embeddings
        - target_images: Optional list of original face images for additional analysis
        - top_n: Number of highest similarity embeddings to use for average
        
        Returns:
        - Dictionary with similarity scores and additional information
        """
        if not target_embeddings:
            return {
                "similarity": 0.0,
                "top_similarities": [],
                "gender_match": None,
                "quality_scores": []
            }
        
        # Calculate similarity and quality for each target embedding
        similarities = []
        quality_scores = []
        genders = []
        
        for i, target_emb in enumerate(target_embeddings):
            # Calculate base similarity
            sim = self.calculate_similarity(embedding, target_emb)
            
            # If we have the original images, perform additional analysis
            quality = 0.8  # Default quality score
            gender = None
            
            if target_images and i < len(target_images) and target_images[i] is not None:
                # Assess image quality
                quality = self.assess_quality(target_images[i])
                
                # Estimate gender
                if quality > 0.4:  # Only estimate gender for reasonable quality images
                    gender, _ = self.estimate_gender(target_images[i])
                    genders.append(gender)
                
                # Apply quality-based weighting
                # Higher quality images get slightly more weight
                weighted_sim = sim * (0.8 + 0.2 * quality)
            else:
                weighted_sim = sim
            
            similarities.append((weighted_sim, sim, i))  # Store weighted_sim, original_sim, and index
            quality_scores.append(quality)
        
        # Sort by weighted similarity (descending)
        similarities.sort(reverse=True)
        
        # Extract top N
        top_similarities = similarities[:min(top_n, len(similarities))]
        
        # Calculate average of top N (using original similarities, not weighted ones)
        original_similarities = [item[1] for item in top_similarities]
        avg_similarity = sum(original_similarities) / len(original_similarities) if original_similarities else 0.0
        
        # Determine if genders match
        gender_match = None
        if target_images and len(genders) > 0:
            query_gender, _ = self.estimate_gender(target_images[0])  # Assuming first image is the query
            gender_match = all(g == query_gender for g in genders) if genders else None
        
        return {
            "similarity": avg_similarity,
            "top_similarities": original_similarities,
            "gender_match": gender_match,
            "quality_scores": quality_scores
        }

    def adaptive_threshold(self, num_references: int, quality_scores: List[float]) -> float:
        """
        Calculate adaptive threshold based on the number of reference images and their quality.
        
        Parameters:
        - num_references: Number of reference images
        - quality_scores: List of quality scores for reference images
        
        Returns:
        - Adaptive threshold value
        """
        # Base threshold
        base_threshold = 0.63
        
        # Adjust based on number of references
        if num_references < 3:
            ref_adjustment = -0.05  # Lower threshold if few references
        elif num_references > 7:
            ref_adjustment = 0.02  # Raise threshold if many references
        else:
            ref_adjustment = 0.0
        
        # Adjust based on average quality
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.7
        if avg_quality < 0.5:
            quality_adjustment = -0.03  # Lower threshold for poor quality images
        elif avg_quality > 0.8:
            quality_adjustment = 0.02  # Raise threshold for high quality images
        else:
            quality_adjustment = 0.0
        
        # Calculate final threshold
        threshold = base_threshold + ref_adjustment + quality_adjustment
        
        # Ensure threshold is in reasonable range
        threshold = max(min(threshold, 0.75), 0.57)
        
        return threshold