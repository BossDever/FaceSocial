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
        # Print image information for debugging
        print(f"DEBUG detect - Image shape: {image.shape}, dtype: {image.dtype}, min-max values: {np.min(image)}-{np.max(image)}")
        
        # Convert image to RGB (MTCNN expects RGB)
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector.detect_faces(image)
        print(f"DEBUG detect - Found {len(faces)} faces")
        
        # Filter faces by size and confidence
        filtered_faces = []
        for face in faces:
            box = face['box']
            confidence = face['confidence']
            
            if confidence >= threshold and box[2] >= min_face_size and box[3] >= min_face_size:
                filtered_faces.append(face)
        
        print(f"DEBUG detect - After filtering: {len(filtered_faces)} faces")
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
            # Print debug information
            print(f"DEBUG align - Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"DEBUG align - Landmarks: {landmarks}")
            print(f"DEBUG align - Landmarks shape: {np.array(landmarks).shape}")
            print(f"DEBUG align - Output size: {output_size}")
            
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
            landmarks_array = np.array(landmarks, dtype=np.float32)
            
            # Flexible landmarks handling
            if landmarks_array.ndim == 1:
                # If flat array, try to reshape to (5, 2)
                if len(landmarks_array) >= 10:
                    landmarks_array = landmarks_array[:10].reshape(5, 2)
                else:
                    # Not enough points, create dummy landmarks
                    print("DEBUG align - Not enough landmarks, creating dummy points")
                    h, w = image.shape[:2]
                    landmarks_array = np.array([
                        [w * 0.3, h * 0.3],  # Left eye
                        [w * 0.7, h * 0.3],  # Right eye
                        [w * 0.5, h * 0.5],  # Nose
                        [w * 0.3, h * 0.7],  # Left mouth
                        [w * 0.7, h * 0.7]   # Right mouth
                    ], dtype=np.float32)
            
            # If we get 1 point array with more than 2 elements, reshape to 5 points
            elif landmarks_array.shape[0] == 1 and len(landmarks_array[0]) >= 10:
                landmarks_array = landmarks_array[0][:10].reshape(5, 2)
                
            # Ensure we have exactly 5 points
            if landmarks_array.shape[0] > 5:
                landmarks_array = landmarks_array[:5]
            elif landmarks_array.shape[0] < 5:
                # Create missing points
                print("DEBUG align - Not enough landmarks, filling missing points")
                h, w = image.shape[:2]
                missing_points = 5 - landmarks_array.shape[0]
                dummy_landmarks = np.array([
                    [w * 0.3, h * 0.3],  # Left eye
                    [w * 0.7, h * 0.3],  # Right eye
                    [w * 0.5, h * 0.5],  # Nose
                    [w * 0.3, h * 0.7],  # Left mouth
                    [w * 0.7, h * 0.7]   # Right mouth
                ], dtype=np.float32)
                landmarks_array = np.vstack([landmarks_array, dummy_landmarks[:missing_points]])
            
            # Ensure each point has 2 coordinates
            if landmarks_array.shape[1] != 2:
                if landmarks_array.shape[1] > 2:
                    landmarks_array = landmarks_array[:, :2]
                else:
                    # Pad with zeros
                    print("DEBUG align - Points don't have 2 coordinates, padding with zeros")
                    padded = np.zeros((5, 2), dtype=np.float32)
                    padded[:, :landmarks_array.shape[1]] = landmarks_array
                    landmarks_array = padded
            
            print(f"DEBUG align - Final landmarks array: {landmarks_array}")
            
            # Calculate the transformation matrix with RANSAC for more robust estimation
            try:
                transformation_matrix, _ = cv2.estimateAffinePartial2D(
                    landmarks_array, reference, method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                print(f"DEBUG align - Transformation matrix: {transformation_matrix}")
            except Exception as e:
                print(f"DEBUG align - Error calculating transformation matrix: {str(e)}")
                # Create a simple transformation matrix (scale and translation)
                s_x = output_size[0] / image.shape[1]
                s_y = output_size[1] / image.shape[0]
                transformation_matrix = np.array([
                    [s_x, 0, 0],
                    [0, s_y, 0]
                ], dtype=np.float32)
            
            if transformation_matrix is None:
                print("DEBUG align - Got None transformation matrix, creating a simple one")
                # Create a simple transformation matrix (scale and translation)
                s_x = output_size[0] / image.shape[1]
                s_y = output_size[1] / image.shape[0]
                transformation_matrix = np.array([
                    [s_x, 0, 0],
                    [0, s_y, 0]
                ], dtype=np.float32)
            
            # Apply the transformation
            aligned_face = cv2.warpAffine(image, transformation_matrix, output_size, flags=cv2.INTER_CUBIC)
            
            # Debug the output
            print(f"DEBUG align - Output image shape: {aligned_face.shape}, dtype: {aligned_face.dtype}")
            print(f"DEBUG align - Output image min-max values: {np.min(aligned_face)}-{np.max(aligned_face)}")
            
            # Ensure the output isn't completely black
            if np.max(aligned_face) < 10:
                print("DEBUG align - Warning: Output image is almost black, using simple resize instead")
                aligned_face = cv2.resize(image, output_size, interpolation=cv2.INTER_CUBIC)
            
            return aligned_face
        
        except Exception as e:
            print(f"DEBUG align - Critical error: {str(e)}")
            # Return a colored image instead of black
            dummy = np.ones((*output_size, 3), dtype=np.uint8) * 128  # Gray
            cv2.putText(dummy, "Error in align", (10, output_size[1]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            return dummy
    
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
            # Print debug information
            print(f"DEBUG extract - Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"DEBUG extract - Bbox: {bbox}")
            print(f"DEBUG extract - Margin: {margin}, Output size: {output_size}")
            
            # Get image dimensions
            img_height, img_width = image.shape[:2]
            
            # Ensure bbox is in the correct format with 4 elements
            if not isinstance(bbox, list) and hasattr(bbox, "__iter__"):
                bbox = list(bbox)
            
            if len(bbox) != 4:
                print(f"DEBUG extract - Invalid bbox format: {bbox}, expected [x, y, width, height]. Creating default bbox.")
                # Create a default bbox in the center of the image
                default_size = min(img_width, img_height) // 2
                x = (img_width - default_size) // 2
                y = (img_height - default_size) // 2
                bbox = [x, y, default_size, default_size]
            
            # Extract coordinates
            x, y, width, height = bbox
            
            print(f"DEBUG extract - After conversion: x={x}, y={y}, width={width}, height={height}")
            
            # Ensure width and height are positive
            width = max(1, width)
            height = max(1, height)
            
            # Calculate margin
            margin_x = int(width * margin)
            margin_y = int(height * margin)
            
            # Calculate coordinates with margin
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img_width, x + width + margin_x)
            y2 = min(img_height, y + height + margin_y)
            
            print(f"DEBUG extract - Region with margin: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Ensure the region is valid
            if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0 or x1 >= img_width or y1 >= img_height:
                print(f"DEBUG extract - Invalid face region: [{x1}, {y1}, {x2}, {y2}]")
                # Create a default region
                x1 = 0
                y1 = 0
                x2 = min(img_width, 100)
                y2 = min(img_height, 100)
                print(f"DEBUG extract - Using default region: [{x1}, {y1}, {x2}, {y2}]")
            
            # Extract face region
            face_img = image[y1:y2, x1:x2]
            print(f"DEBUG extract - Extracted face shape: {face_img.shape}")
            
            # Resize to output size
            face_img = cv2.resize(face_img, output_size, interpolation=cv2.INTER_CUBIC)
            
            # Debug the output
            print(f"DEBUG extract - Output image shape: {face_img.shape}, dtype: {face_img.dtype}")
            print(f"DEBUG extract - Output image min-max values: {np.min(face_img)}-{np.max(face_img)}")
            
            return face_img
        
        except Exception as e:
            print(f"DEBUG extract - Critical error: {str(e)}")
            # Return a colored image instead of black
            dummy = np.ones((*output_size, 3), dtype=np.uint8) * 64  # Dark gray
            cv2.putText(dummy, "Error in extract", (10, output_size[1]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            return dummy
    
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