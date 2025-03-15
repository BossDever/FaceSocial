import os
import numpy as np
import tensorflow as tf
import cv2
import onnxruntime as ort
from typing import List, Dict, Any, Tuple, Optional, Union
import json

class ModelEnsemble:
    """
    Ensemble of face recognition models for improved accuracy.
    Combines multiple models including FaceNet, ArcFace, and CosFace.
    """
    
    def __init__(self, base_model_path: str = '/app/models'):
        """
        Initialize the model ensemble.
        
        Parameters:
        - base_model_path: Base path to the models directory
        """
        self.base_path = base_model_path
        print(f"Initializing ModelEnsemble with base path: {base_model_path}")
        
        # List all files in model directory tree
        if os.path.exists(base_model_path):
            for root, dirs, files in os.walk(base_model_path):
                print(f"Models directory {root} contains: {files}")
        else:
            print(f"Models base path does not exist: {base_model_path}")
        
        self.models = {}
        self.model_weights = {}
        
        # Set default weights for models
        self.default_weights = {
            'facenet': 0.35,
            'arcface': 0.35,
            'cosface': 0.3
        }
        
        # Load available models
        self._load_facenet()
        self._load_arcface()
        self._load_cosface()
        
        # Initialize model weights
        self._init_weights()
        
        print(f"Loaded {len(self.models)} models for ensemble: {list(self.models.keys())}")
    
    def _load_facenet(self):
        """Load FaceNet model if available"""
        # Define possible paths for FaceNet model
        possible_pb_paths = [
            os.path.join(self.base_path, 'facenet/20180402-114759.pb'),
            '/app/app/models/facenet/20180402-114759.pb',
            '/app/models/facenet/20180402-114759.pb',
            './app/models/facenet/20180402-114759.pb'
        ]
        
        possible_keras_paths = [
            os.path.join(self.base_path, 'keras-facenet-h5/facenet_keras.h5'),
            '/app/app/models/keras-facenet-h5/facenet_keras.h5',
            '/app/models/keras-facenet-h5/facenet_keras.h5',
            './app/models/keras-facenet-h5/facenet_keras.h5'
        ]
        
        # Try TensorFlow FaceNet model paths
        for facenet_pb_path in possible_pb_paths:
            if os.path.exists(facenet_pb_path):
                try:
                    print(f"Found FaceNet TF model at: {facenet_pb_path}")
                    with tf.Graph().as_default() as graph:
                        with tf.io.gfile.GFile(facenet_pb_path, 'rb') as f:
                            graph_def = tf.compat.v1.GraphDef()
                            graph_def.ParseFromString(f.read())
                            tf.import_graph_def(graph_def, name='')

                        # Get input and output tensors
                        self.facenet_input = graph.get_tensor_by_name('input:0')
                        self.facenet_embeddings = graph.get_tensor_by_name('embeddings:0')
                        self.facenet_phase_train = graph.get_tensor_by_name('phase_train:0')

                        # Create a session
                        config = tf.compat.v1.ConfigProto()
                        config.gpu_options.allow_growth = True
                        self.facenet_sess = tf.compat.v1.Session(graph=graph, config=config)
                    
                    self.models['facenet'] = {
                        'type': 'tf_graph', 
                        'dim': 128,
                        'path': facenet_pb_path
                    }
                    print("Loaded FaceNet TensorFlow model successfully")
                    return  # Exit if loaded successfully
                except Exception as e:
                    print(f"Error loading FaceNet TensorFlow model from {facenet_pb_path}: {str(e)}")
        
        # Try Keras FaceNet model paths if TF model failed
        for keras_facenet_path in possible_keras_paths:
            if os.path.exists(keras_facenet_path):
                try:
                    print(f"Found FaceNet Keras model at: {keras_facenet_path}")
                    model = tf.keras.models.load_model(keras_facenet_path)
                    self.models['facenet'] = {
                        'type': 'keras', 
                        'model': model, 
                        'dim': 128,
                        'path': keras_facenet_path
                    }
                    print("Loaded FaceNet Keras model successfully")
                    return  # Exit if loaded successfully
                except Exception as e:
                    print(f"Error loading FaceNet Keras model from {keras_facenet_path}: {str(e)}")
        
        print("Failed to load FaceNet model from any location")
    
    def _load_arcface(self):
        """Load ArcFace model if available"""
        # Define possible paths for ArcFace model
        possible_paths = [
            os.path.join(self.base_path, 'arcface/arcface.onnx'),
            '/app/app/models/arcface/arcface.onnx',
            '/app/models/arcface/arcface.onnx',
            './app/models/arcface/arcface.onnx'
        ]
        
        for arcface_path in possible_paths:
            if os.path.exists(arcface_path):
                try:
                    print(f"Found ArcFace model at: {arcface_path}")
                    
                    # Import the ONNX helper to get the best providers
                    from app.core.onnx_helper import create_onnx_session
                    session = create_onnx_session(arcface_path)
                    
                    # Store session
                    self.models['arcface'] = {
                        'type': 'onnx',
                        'session': session,
                        'dim': 512,  # ArcFace typically has 512-dimensional embeddings
                        'path': arcface_path
                    }
                    print("Loaded ArcFace ONNX model successfully")
                    return  # Exit once loaded successfully
                except Exception as e:
                    print(f"Error loading ArcFace model from {arcface_path}: {str(e)}")
        
        print("Failed to load ArcFace model from any location")
    
    def _load_cosface(self):
        """Load CosFace model if available"""
        # Define possible paths for CosFace model
        possible_paths = [
            os.path.join(self.base_path, 'cosface/glint360k_cosface_r50.onnx'),
            '/app/app/models/cosface/glint360k_cosface_r50.onnx',
            '/app/models/cosface/glint360k_cosface_r50.onnx',
            './app/models/cosface/glint360k_cosface_r50.onnx'
        ]
        
        for cosface_path in possible_paths:
            if os.path.exists(cosface_path):
                try:
                    print(f"Found CosFace model at: {cosface_path}")
                    
                    # Import the ONNX helper to get the best providers
                    from app.core.onnx_helper import create_onnx_session
                    session = create_onnx_session(cosface_path)
                    
                    # Store session
                    self.models['cosface'] = {
                        'type': 'onnx',
                        'session': session,
                        'dim': 512,  # CosFace typically has 512-dimensional embeddings
                        'path': cosface_path
                    }
                    print("Loaded CosFace ONNX model successfully")
                    return  # Exit once loaded successfully
                except Exception as e:
                    print(f"Error loading CosFace model from {cosface_path}: {str(e)}")
        
        print("Failed to load CosFace model from any location")
    
    def _init_weights(self):
        """Initialize model weights based on available models"""
        # ค่าน้ำหนักเริ่มต้นใหม่ตามประสิทธิภาพของแต่ละโมเดล
        # ให้น้ำหนักกับ ArcFace มากขึ้น เนื่องจากมีความแม่นยำสูงกว่า
        self.default_weights = {
            'facenet': 0.20,  # ลดลงจาก 0.25
            'arcface': 0.60,  # เพิ่มขึ้นจาก 0.50
            'cosface': 0.20   # ลดลงจาก 0.25
        }
        
        # นำค่าน้ำหนักมาใช้กับโมเดลที่โหลดได้สำเร็จ
        total_weight = 0
        for model_name in self.models.keys():
            if model_name in self.default_weights:
                self.model_weights[model_name] = self.default_weights[model_name]
                total_weight += self.default_weights[model_name]
        
        # ปรับให้ผลรวมเป็น 1.0 เสมอ
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight
                
        # Log the final weights
        print(f"Model weights: {self.model_weights}")
    
    def preprocess_for_model(self, face_image: np.ndarray, model_name: str) -> np.ndarray:
        """
        Preprocess face image for specific model.
        
        Parameters:
        - face_image: Input face image
        - model_name: Name of the model to preprocess for
        
        Returns:
        - Preprocessed face image
        """
        if face_image is None:
            raise ValueError("Face image is None")
        
        if model_name == 'facenet':
            # Resize to 160x160 for FaceNet
            if face_image.shape[:2] != (160, 160):
                face_image = cv2.resize(face_image, (160, 160), interpolation=cv2.INTER_CUBIC)
            
            # Ensure the image has 3 channels
            if len(face_image.shape) == 2:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 4:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
            
            # Standardize
            preprocessed = (face_image - 127.5) / 128.0
            return preprocessed
        
        elif model_name == 'arcface':
            # Resize to 112x112 for ArcFace
            if face_image.shape[:2] != (112, 112):
                face_image = cv2.resize(face_image, (112, 112), interpolation=cv2.INTER_CUBIC)
            
            # Convert to RGB if needed
            if len(face_image.shape) == 2:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 4:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
            
            # Transpose to NCHW format for ONNX
            preprocessed = np.transpose(face_image, (2, 0, 1))
            # Convert to float32 and normalize
            preprocessed = preprocessed.astype(np.float32) / 255.0
            # Add batch dimension
            preprocessed = np.expand_dims(preprocessed, axis=0)
            return preprocessed
        
        elif model_name == 'cosface':
            # Resize to 112x112 for CosFace
            if face_image.shape[:2] != (112, 112):
                face_image = cv2.resize(face_image, (112, 112), interpolation=cv2.INTER_CUBIC)
            
            # Convert to RGB if needed
            if len(face_image.shape) == 2:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 4:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
            
            # Transpose to NCHW format for ONNX
            preprocessed = np.transpose(face_image, (2, 0, 1))
            # Convert to float32 and normalize
            preprocessed = preprocessed.astype(np.float32) / 255.0
            # Add batch dimension
            preprocessed = np.expand_dims(preprocessed, axis=0)
            return preprocessed
        
        else:
            # Default preprocessing
            if face_image.shape[:2] != (160, 160):
                face_image = cv2.resize(face_image, (160, 160), interpolation=cv2.INTER_CUBIC)
            
            # Ensure 3 channels
            if len(face_image.shape) == 2:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            
            # Normalize
            preprocessed = face_image.astype(np.float32) / 255.0
            return preprocessed
    
    def generate_embedding(self, face_image: np.ndarray, model_name: str) -> np.ndarray:
        """
        Generate embedding for a specific model.
        
        Parameters:
        - face_image: Input face image
        - model_name: Name of the model to use
        
        Returns:
        - Face embedding vector
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model_info = self.models[model_name]
        preprocessed = self.preprocess_for_model(face_image, model_name)
        
        if model_name == 'facenet':
            if model_info['type'] == 'tf_graph':
                # Generate embedding using TensorFlow graph
                feed_dict = {
                    self.facenet_input: np.expand_dims(preprocessed, axis=0),
                    self.facenet_phase_train: False
                }
                embedding = self.facenet_sess.run(self.facenet_embeddings, feed_dict=feed_dict)[0]
            else:  # Keras model
                # Add batch dimension if not already present
                if len(preprocessed.shape) == 3:
                    preprocessed = np.expand_dims(preprocessed, axis=0)
                embedding = model_info['model'].predict(preprocessed)[0]
        
        elif model_name in ['arcface', 'cosface']:
            # Get input name
            input_name = self.models[model_name]['session'].get_inputs()[0].name
            
            # Run inference
            outputs = self.models[model_name]['session'].run(None, {input_name: preprocessed})
            embedding = outputs[0][0]
        
        else:
            # Fallback
            raise ValueError(f"Model {model_name} processing not implemented")
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def generate_ensemble_embedding(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Generate embeddings from all available models and combine them.
        
        Parameters:
        - face_image: Input face image
        
        Returns:
        - Dictionary with ensemble embedding and individual model embeddings
        """
        model_embeddings = {}
        model_names = list(self.models.keys())
        
        # Generate embeddings for each model
        for model_name in model_names:
            try:
                embedding = self.generate_embedding(face_image, model_name)
                model_embeddings[model_name] = embedding
            except Exception as e:
                print(f"Error generating embedding with {model_name}: {str(e)}")
        
        return {
            "model_embeddings": model_embeddings,
            "model_weights": {k: v for k, v in self.model_weights.items() if k in model_embeddings}
        }
    
    def calculate_similarity(self, embedding1: Dict[str, Any], embedding2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate similarity between two ensemble embeddings.
        
        Parameters:
        - embedding1: First ensemble embedding
        - embedding2: Second ensemble embedding
        
        Returns:
        - Dictionary with similarity scores and additional information
        """
        # Find common models
        common_models = set(embedding1["model_embeddings"].keys()) & set(embedding2["model_embeddings"].keys())
        
        if not common_models:
            return {
                "ensemble_similarity": 0.0,
                "model_similarities": {},
                "common_models": []
            }
        
        # Calculate similarity for each common model
        model_similarities = {}
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for model_name in common_models:
            emb1 = embedding1["model_embeddings"][model_name]
            emb2 = embedding2["model_embeddings"][model_name]
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2)
            similarity = max(0, min(1, similarity))  # Clip to 0-1 range
            
            model_similarities[model_name] = similarity
            
            # Apply weight
            weight = self.model_weights.get(model_name, 0.0)
            weighted_sum += similarity * weight
            weight_sum += weight
        
        # Calculate weighted average similarity
        ensemble_similarity = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        return {
            "ensemble_similarity": ensemble_similarity,
            "model_similarities": model_similarities,
            "common_models": list(common_models)
        }
    
    def adjust_weights(self, new_weights: Dict[str, float]):
        """
        Adjust model weights for ensemble.
        
        Parameters:
        - new_weights: Dictionary with new weights for models
        """
        # Update weights for available models
        total_weight = 0.0
        for model_name, weight in new_weights.items():
            if model_name in self.models:
                self.model_weights[model_name] = weight
                total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight