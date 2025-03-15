from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple, Union

class FaceEmbeddingRequest(BaseModel):
    """
    Request model for face embedding generation.
    """
    image: str = Field(..., description="Base64 encoded image")
    detect_and_align: bool = Field(True, description="Automatically detect and align face before embedding")

class FaceEmbeddingResponse(BaseModel):
    """
    Response model for face embedding generation.
    """
    embedding: List[float] = Field(..., description="Face embedding vector")
    embedding_size: int = Field(..., description="Size of embedding vector")
    quality_score: float = Field(..., description="Quality score of the face image")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class FaceComparisonRequest(BaseModel):
    """
    Request model for face comparison.
    """
    embedding1: Optional[List[float]] = Field(None, description="First face embedding")
    embedding2: Optional[List[float]] = Field(None, description="Second face embedding")
    or_face1: Optional[str] = Field(None, description="Base64 encoded first face image (alternative to embedding1)")
    or_face2: Optional[str] = Field(None, description="Base64 encoded second face image (alternative to embedding2)")

class FaceComparisonResponse(BaseModel):
    """
    Response model for face comparison.
    """
    similarity: float = Field(..., description="Similarity score between faces (0-1)")
    is_same_person: bool = Field(..., description="Whether the faces belong to the same person")
    threshold_used: float = Field(..., description="Threshold used for determining if same person")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class FaceIdentificationRequest(BaseModel):
    """
    Request model for face identification.
    """
    embedding: Optional[List[float]] = Field(None, description="Face embedding to identify")
    or_face: Optional[str] = Field(None, description="Base64 encoded face image (alternative to embedding)")
    top_k: int = Field(5, description="Number of top matches to return")
    collection_name: str = Field("users", description="Milvus collection name to search in")
    min_similarity: float = Field(0.75, description="Minimum similarity threshold")

class FaceMatch(BaseModel):
    """
    Face match result.
    """
    user_id: int = Field(..., description="User ID of the matched face")
    similarity: float = Field(..., description="Similarity score")
    embedding_id: str = Field(..., description="ID of the matched embedding")

class FaceIdentificationResponse(BaseModel):
    """
    Response model for face identification.
    """
    matches: List[FaceMatch] = Field(..., description="List of matched faces")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class FaceRegistrationRequest(BaseModel):
    """
    Request model for face registration.
    """
    user_id: int = Field(..., description="User ID to register faces for")
    embeddings: Optional[List[List[float]]] = Field(None, description="List of face embeddings to register")
    or_faces: Optional[List[str]] = Field(None, description="List of base64 encoded face images (alternative to embeddings)")
    collection_name: str = Field("users", description="Milvus collection name to store in")

class FaceRegistrationResponse(BaseModel):
    """
    Response model for face registration.
    """
    success: bool = Field(..., description="Whether registration was successful")
    embedding_ids: List[str] = Field(..., description="IDs of registered embeddings")
    quality_scores: List[float] = Field(..., description="Quality scores of registered faces")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")