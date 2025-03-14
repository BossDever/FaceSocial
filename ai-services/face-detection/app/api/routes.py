from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
import numpy as np
import base64
import cv2
import time

from app.core.face_detector import FaceDetector
from app.models.schemas import (
    FaceDetectionRequest,
    FaceDetectionResponse,
    FaceAlignmentRequest,
    FaceAlignmentResponse,
    FaceExtractionRequest,
    FaceExtractionResponse,
)

router = APIRouter()
face_detector = FaceDetector()

@router.post("/detect", response_model=FaceDetectionResponse)
async def detect_faces(request: FaceDetectionRequest):
    """
    Detect faces in an image.
    
    Parameters:
    - image: Base64 encoded image
    - min_face_size: Minimum face size to detect (default: 50)
    - threshold: Detection confidence threshold (default: 0.85)
    - return_landmarks: Whether to return facial landmarks (default: True)
    
    Returns:
    - faces: List of detected faces with bounding boxes, confidence, and landmarks
    - count: Number of faces detected
    - processing_time_ms: Processing time in milliseconds
    """
    try:
        # Decode base64 image
        start_time = time.time()
        
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Detect faces
        faces = face_detector.detect(
            image, 
            min_face_size=request.min_face_size, 
            threshold=request.threshold
        )
        
        # Format response
        result = []
        for face in faces:
            bbox = face['box']
            confidence = face['confidence']
            landmarks = face['keypoints'] if request.return_landmarks else None
            
            face_data = {
                "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                "confidence": float(confidence),
            }
            
            if request.return_landmarks:
                # Convert landmarks to list format
                landmark_list = []
                for key, value in landmarks.items():
                    landmark_list.append([int(value[0]), int(value[1])])
                face_data["landmarks"] = landmark_list
                
            result.append(face_data)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "faces": result,
            "count": len(result),
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting faces: {str(e)}")

@router.post("/align", response_model=FaceAlignmentResponse)
async def align_face(request: FaceAlignmentRequest):
    """
    Align a face using facial landmarks.
    
    Parameters:
    - image: Base64 encoded image
    - landmarks: List of 5 facial landmarks (eyes, nose, mouth corners)
        - Format 1 (recommended): [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]
        - Format 2 (legacy): [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
        - Format 3 (legacy): any array that can be converted to 5 landmarks
    - output_size: Output image size [width, height] (default: [160, 160])
    
    Returns:
    - aligned_face: Base64 encoded aligned face image
    - processing_time_ms: Processing time in milliseconds
    """
    try:
        start_time = time.time()
        
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Flexible landmarks handling - try to make it work with different formats
        landmarks_data = request.landmarks
        
        # Try to convert landmarks to the format we need: array of 5 points
        landmarks_array = np.array(landmarks_data, dtype=np.float32)
        
        # Reshape if needed
        if landmarks_array.ndim == 1:
            # If flat array, try to reshape to (5, 2)
            if len(landmarks_array) >= 10:
                landmarks_array = landmarks_array[:10].reshape(5, 2)
            else:
                # If not enough points, duplicate the first point
                first_point = landmarks_array[:2] if len(landmarks_array) >= 2 else np.array([0, 0])
                landmarks_array = np.array([first_point] * 5)
        
        # If we get 1 point array with more than 2 elements, reshape to 5 points
        elif landmarks_array.shape[0] == 1 and len(landmarks_array[0]) >= 10:
            landmarks_array = landmarks_array[0][:10].reshape(5, 2)
            
        # Ensure we have exactly 5 points
        if landmarks_array.shape[0] > 5:
            landmarks_array = landmarks_array[:5]
        elif landmarks_array.shape[0] < 5:
            # Duplicate the first point to make 5 points
            first_point = landmarks_array[0] if landmarks_array.shape[0] > 0 else np.array([0, 0])
            missing_points = 5 - landmarks_array.shape[0]
            landmarks_array = np.vstack([landmarks_array, np.tile(first_point, (missing_points, 1))])
        
        # Ensure each point has 2 coordinates
        if landmarks_array.shape[1] != 2:
            if landmarks_array.shape[1] > 2:
                landmarks_array = landmarks_array[:, :2]
            else:
                # Pad with zeros
                padded = np.zeros((5, 2), dtype=np.float32)
                padded[:, :landmarks_array.shape[1]] = landmarks_array
                landmarks_array = padded
        
        # Align face
        aligned_face = face_detector.align(image, landmarks_array, request.output_size)
        
        # Encode the aligned face as base64
        _, buffer = cv2.imencode('.jpg', aligned_face)
        aligned_face_base64 = base64.b64encode(buffer).decode('utf-8')
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "aligned_face": aligned_face_base64,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error aligning face: {str(e)}")

@router.post("/extract", response_model=FaceExtractionResponse)
async def extract_face(request: FaceExtractionRequest):
    """
    Extract a face from an image using the bounding box.
    
    Parameters:
    - image: Base64 encoded image
    - bbox: Bounding box [x1, y1, x2, y2]
    - margin: Margin to add around the bounding box (default: 0.2)
    - output_size: Output image size [width, height] (default: [160, 160])
    
    Returns:
    - face_image: Base64 encoded extracted face image
    - processing_time_ms: Processing time in milliseconds
    
    Example:
    ```
    {
      "image": "base64_encoded_image",
      "bbox": [846, 873, 1108, 1230],  # [x1, y1, x2, y2]
      "margin": 0.2,
      "output_size": [160, 160]
    }
    ```
    """
    try:
        start_time = time.time()
        
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Validate bbox format
        if not isinstance(request.bbox, list):
            raise HTTPException(
                status_code=400, 
                detail=f"bbox must be a list, got {type(request.bbox).__name__}"
            )
        
        if len(request.bbox) != 4:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 4 values in bbox, got {len(request.bbox)}. " +
                "bbox should be [x1, y1, x2, y2]"
            )
        
        # Extract face
        bbox = request.bbox  # [x1, y1, x2, y2]
        
        # Convert to the format expected by the extract function [x, y, width, height]
        bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        
        face_image = face_detector.extract(image, bbox_xywh, request.margin, request.output_size)
        
        # Encode the face image as base64
        _, buffer = cv2.imencode('.jpg', face_image)
        face_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "face_image": face_image_base64,
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions with status codes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting face: {str(e)}")