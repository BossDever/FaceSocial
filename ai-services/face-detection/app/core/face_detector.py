import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf

class FaceDetector:
    """
    Face detector class using MTCNN for face detection and alignment.
    """
    
    def __init__(self):
        """
        Initialize the face detector.
        """
        # Set memory growth for GPU to avoid allocating all memory at once
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Initialize MTCNN detector
        self.detector = MTCNN()
    
    def detect(self, image, min_face_size=50, threshold=0.85):
        """
        Detect faces in an image.
        
        Parameters:
        - image: Input image
        - min_face_size: Minimum face size to detect
        - threshold: Detection confidence threshold
        
        Returns:
        - List of detected faces with bounding boxes, confidence, and landmarks
        """
        # Convert image to RGB (MTCNN expects RGB)
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector.detect_faces(image)
        
        # Filter faces by size and confidence
        filtered_faces = []
        for face in faces:
            box = face['box']
            confidence = face['confidence']
            
            if confidence >= threshold and box[2] >= min_face_size and box[3] >= min_face_size:
                filtered_faces.append(face)
        
        return filtered_faces
    
    def align(self, image, landmarks, output_size=(160, 160)):
    """
    Align face based on facial landmarks.
    
    Parameters:
    - image: Input image
    - landmarks: 5 point landmarks (left eye, right eye, nose, left mouth, right mouth)
    - output_size: Size of the output image
    
    Returns:
    - Aligned face image
    """
    try:
        # Define reference points for alignment
        # These reference points correspond to FaceNet's expected 5 points
        # [left_eye, right_eye, nose, left_mouth, right_mouth]
        reference = np.array([
            [30.2946, 51.6963],  # Left eye
            [65.5318, 51.6963],  # Right eye
            [48.0252, 71.7366],  # Nose
            [33.5493, 92.3655],  # Left mouth corner
            [62.7299, 92.3655]   # Right mouth corner
        ], dtype=np.float32)
        
        # Scale reference points to match the output size
        reference[:, 0] *= output_size[0] / 96.0
        reference[:, 1] *= output_size[1] / 96.0
        
        # Ensure landmarks have the correct format
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Calculate the transformation matrix with full=False for more robust estimation
        transformation_matrix, _ = cv2.estimateAffinePartial2D(landmarks, reference, method=cv2.RANSAC)
        
        if transformation_matrix is None:
            # Fallback: use simplified alignment (just translation and scaling)
            src_mean = np.mean(landmarks, axis=0)
            dst_mean = np.mean(reference, axis=0)
            src_scale = np.std(landmarks)
            dst_scale = np.std(reference)
            
            scale = dst_scale / src_scale if src_scale > 0 else 1.0
            
            transformation_matrix = np.array([
                [scale, 0, dst_mean[0] - scale * src_mean[0]],
                [0, scale, dst_mean[1] - scale * src_mean[1]]
            ], dtype=np.float32)
        
        # Apply the transformation
        aligned_face = cv2.warpAffine(image, transformation_matrix, output_size, flags=cv2.INTER_CUBIC)
        
        return aligned_face
    
    except Exception as e:
        print(f"Error in align function: {str(e)}")
        # Return a simple resized version of the image as fallback
        return cv2.resize(image, output_size, interpolation=cv2.INTER_CUBIC)
    
    def extract(self, image, bbox, margin=0.2, output_size=(160, 160)):
        """
        Extract a face from an image using the bounding box.
        
        Parameters:
        - image: Input image
        - bbox: Bounding box [x, y, width, height]
        - margin: Margin to add around the bounding box
        - output_size: Size of the output image
        
        Returns:
        - Extracted face image
        """
        try:
            # Get image dimensions
            img_height, img_width = image.shape[:2]
            
            # Ensure bbox is in the correct format with 4 elements
            if len(bbox) != 4:
                raise ValueError(f"Invalid bbox format: {bbox}, expected [x, y, width, height]")
            
            # Extract coordinates
            x, y, width, height = bbox
            
            # Calculate margin
            margin_x = int(width * margin)
            margin_y = int(height * margin)
            
            # Calculate coordinates with margin
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img_width, x + width + margin_x)
            y2 = min(img_height, y + height + margin_y)
            
            # Ensure the region is valid
            if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0 or x1 >= img_width or y1 >= img_height:
                raise ValueError(f"Invalid face region: [{x1}, {y1}, {x2}, {y2}]")
            
            # Extract face region
            face_img = image[y1:y2, x1:x2]
            
            # Resize to output size
            face_img = cv2.resize(face_img, output_size, interpolation=cv2.INTER_CUBIC)
            
            return face_img
        
        except Exception as e:
            print(f"Error in extract function: {str(e)}")
            # Return a blank image as fallback
            return np.zeros((*output_size, 3), dtype=np.uint8)
    
    def preprocess_image(self, image):
        """
        Preprocess an image for better face detection.
        
        Parameters:
        - image: Input image
        
        Returns:
        - Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        equalized = cv2.equalizeHist(gray)
        
        # Convert back to BGR
        equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        
        return equalized_bgr