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
    - output_size: Output image size [width, height] (default: [160, 160])
    
    Returns:
    - aligned_face: Base64 encoded aligned face image
    - processing_time_ms: Processing time in milliseconds
    
    Example:
    ```
    {
      "image": "base64_encoded_image",
      "landmarks": [
        [936, 1095],  # Left eye
        [1005, 1149], # Right eye
        [1019, 1024], # Nose
        [902, 1021],  # Left mouth corner
        [904, 1145]   # Right mouth corner
      ],
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
        
        # Validate landmarks format
        if not isinstance(request.landmarks, list):
            raise HTTPException(
                status_code=400, 
                detail=f"landmarks must be a list, got {type(request.landmarks).__name__}"
            )
        
        if len(request.landmarks) != 5:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 5 landmarks, got {len(request.landmarks)}. " +
                "Landmarks should be 5 points: [left_eye, right_eye, nose, left_mouth, right_mouth]"
            )
        
        for i, point in enumerate(request.landmarks):
            if not isinstance(point, list) or len(point) != 2:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Landmark point {i} should be a list of 2 coordinates [x, y], got {point}"
                )
        
        # Convert landmarks to numpy array
        landmarks = np.array(request.landmarks, dtype=np.float32)
        
        # Align face
        aligned_face = face_detector.align(image, landmarks, request.output_size)
        
        # Encode the aligned face as base64
        _, buffer = cv2.imencode('.jpg', aligned_face)
        aligned_face_base64 = base64.b64encode(buffer).decode('utf-8')
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "aligned_face": aligned_face_base64,
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions with status codes
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
    """
    try:
        start_time = time.time()
        
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Validate bbox
        if len(request.bbox) != 4:
            raise HTTPException(status_code=400, detail=f"Expected 4 values in bbox, got {len(request.bbox)}")
        
        # Extract face
        bbox = request.bbox  # [x1, y1, x2, y2]
        
        # Convert to the format expected by the extract function [x, y, width, height]
        bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        
        # Log for debugging
        print(f"Input bbox: {request.bbox}, Converted bbox: {bbox_xywh}")
        
        face_image = face_detector.extract(image, bbox_xywh, request.margin, request.output_size)
        
        # Encode the face image as base64
        _, buffer = cv2.imencode('.jpg', face_image)
        face_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "face_image": face_image_base64,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting face: {str(e)}")