from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class FaceDetectionRequest(BaseModel):
    """
    Request model for face detection.
    """
    image: str = Field(..., description="Base64 encoded image")
    min_face_size: int = Field(50, description="Minimum face size to detect")
    threshold: float = Field(0.85, description="Detection confidence threshold")
    return_landmarks: bool = Field(True, description="Whether to return facial landmarks")

class Face(BaseModel):
    """
    Face detection result.
    """
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence")
    landmarks: Optional[List[List[int]]] = Field(None, description="Facial landmarks")

class FaceDetectionResponse(BaseModel):
    """
    Response model for face detection.
    """
    faces: List[Face] = Field(..., description="List of detected faces")
    count: int = Field(..., description="Number of faces detected")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class FaceAlignmentRequest(BaseModel):
    """
    Request model for face alignment.
    """
    image: str = Field(..., description="Base64 encoded image")
    landmarks: List[List[int]] = Field(..., description="List of 5 facial landmarks (eyes, nose, mouth corners)")
    output_size: Tuple[int, int] = Field((160, 160), description="Output image size [width, height]")

class FaceAlignmentResponse(BaseModel):
    """
    Response model for face alignment.
    """
    aligned_face: str = Field(..., description="Base64 encoded aligned face image")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class FaceExtractionRequest(BaseModel):
    """
    Request model for face extraction.
    """
    image: str = Field(..., description="Base64 encoded image")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    margin: float = Field(0.2, description="Margin to add around the bounding box")
    output_size: Tuple[int, int] = Field((160, 160), description="Output image size [width, height]")

class FaceExtractionResponse(BaseModel):
    """
    Response model for face extraction.
    """
    face_image: str = Field(..., description="Base64 encoded extracted face image")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")