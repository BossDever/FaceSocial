from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
import numpy as np
import base64
import cv2
import time
import os

from fastapi.responses import HTMLResponse 
from app.core.face_embedder import FaceEmbedder
from app.core.milvus_client import MilvusClient
from app.models.schemas import (
    FaceEmbeddingRequest,
    FaceEmbeddingResponse,
    FaceComparisonRequest,
    FaceComparisonResponse,
    FaceIdentificationRequest,
    FaceIdentificationResponse,
    FaceRegistrationRequest,
    FaceRegistrationResponse,
)

router = APIRouter()

# Initialize FaceEmbedder with model path
model_path = os.getenv("FACENET_MODEL_PATH", None)
face_embedder = FaceEmbedder(model_path)

# Initialize MilvusClient
milvus_host = os.getenv("MILVUS_HOST", "milvus")
milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
milvus_client = MilvusClient(host=milvus_host, port=milvus_port)

# Default similarity threshold
DEFAULT_SIMILARITY_THRESHOLD = 0.75

@router.post("/embed", response_model=FaceEmbeddingResponse)
async def generate_embedding(request: FaceEmbeddingRequest):
    """
    Generate a face embedding from an image.
    
    Parameters:
    - image: Base64 encoded image
    - detect_and_align: Whether to automatically detect and align the face
    
    Returns:
    - embedding: Face embedding vector
    - embedding_size: Size of the embedding vector
    - quality_score: Quality score of the face image
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
        
        # If detect_and_align is True, use Face Detection Service
        if request.detect_and_align:
            # In a production environment, you would call the Face Detection Service API
            # For now, we'll just use the image as-is
            print("Detect and align functionality would be integrated with Face Detection Service")
            # In the future: face_image = await call_face_detection_service(image)
            face_image = image
        else:
            face_image = image
        
        # Generate face embedding
        embedding = face_embedder.generate_embedding(face_image)
        
        # Assess face quality
        quality_score = face_embedder.assess_quality(face_image)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "embedding": embedding.tolist(),
            "embedding_size": len(embedding),
            "quality_score": quality_score,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

@router.post("/compare", response_model=FaceComparisonResponse)
async def compare_faces(request: FaceComparisonRequest):
    """
    Compare two faces and calculate similarity.
    
    Parameters:
    - embedding1: First face embedding (optional)
    - embedding2: Second face embedding (optional)
    - or_face1: Base64 encoded first face image (alternative to embedding1)
    - or_face2: Base64 encoded second face image (alternative to embedding2)
    
    Returns:
    - similarity: Similarity score between faces (0-1)
    - is_same_person: Whether the faces belong to the same person
    - threshold_used: Threshold used for determining if same person
    - processing_time_ms: Processing time in milliseconds
    """
    try:
        start_time = time.time()
        
        # Get embeddings - either from request or generate from images
        embedding1 = None
        embedding2 = None
        
        if request.embedding1 is not None:
            embedding1 = np.array(request.embedding1)
        elif request.or_face1 is not None:
            # Decode base64 image
            image_data = base64.b64decode(request.or_face1)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image data for first face")
            
            # Generate embedding
            embedding1 = face_embedder.generate_embedding(image)
        else:
            raise HTTPException(status_code=400, detail="Must provide either embedding1 or or_face1")
        
        if request.embedding2 is not None:
            embedding2 = np.array(request.embedding2)
        elif request.or_face2 is not None:
            # Decode base64 image
            image_data = base64.b64decode(request.or_face2)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image data for second face")
            
            # Generate embedding
            embedding2 = face_embedder.generate_embedding(image)
        else:
            raise HTTPException(status_code=400, detail="Must provide either embedding2 or or_face2")
        
        # Calculate similarity
        similarity = face_embedder.calculate_similarity(embedding1, embedding2)
        
        # Determine if same person
        is_same_person = similarity >= DEFAULT_SIMILARITY_THRESHOLD
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "similarity": similarity,
            "is_same_person": is_same_person,
            "threshold_used": DEFAULT_SIMILARITY_THRESHOLD,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing faces: {str(e)}")

@router.post("/identify", response_model=FaceIdentificationResponse)
async def identify_face(request: FaceIdentificationRequest):
    """
    Identify a face by searching for similar faces in the database.
    
    Parameters:
    - embedding: Face embedding to identify (optional)
    - or_face: Base64 encoded face image (alternative to embedding)
    - top_k: Number of top matches to return
    - collection_name: Milvus collection name to search in
    - min_similarity: Minimum similarity threshold
    
    Returns:
    - matches: List of matched faces with user_id, similarity, and embedding_id
    - processing_time_ms: Processing time in milliseconds
    """
    try:
        start_time = time.time()
        
        # Get embedding - either from request or generate from image
        embedding = None
        
        if request.embedding is not None:
            embedding = np.array(request.embedding)
        elif request.or_face is not None:
            # Decode base64 image
            image_data = base64.b64decode(request.or_face)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image data")
            
            # Generate embedding
            embedding = face_embedder.generate_embedding(image)
        else:
            raise HTTPException(status_code=400, detail="Must provide either embedding or or_face")
        
        # Search for similar faces in Milvus
        matches = milvus_client.search_embeddings(
            query_embedding=embedding,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            collection_name=request.collection_name
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "matches": matches,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error identifying face: {str(e)}")

@router.post("/register", response_model=FaceRegistrationResponse)
async def register_face(request: FaceRegistrationRequest):
    """
    Register face embeddings for a user.
    
    Parameters:
    - user_id: User ID to register faces for
    - embeddings: List of face embeddings to register (optional)
    - or_faces: List of base64 encoded face images (alternative to embeddings)
    - collection_name: Milvus collection name to store in
    
    Returns:
    - success: Whether registration was successful
    - embedding_ids: IDs of registered embeddings
    - quality_scores: Quality scores of registered faces
    - processing_time_ms: Processing time in milliseconds
    """
    try:
        start_time = time.time()
        
        embeddings = []
        quality_scores = []
        
        if request.embeddings is not None:
            # Use provided embeddings
            embeddings = [np.array(emb) for emb in request.embeddings]
            
            # Generate placeholder quality scores
            quality_scores = [0.9] * len(embeddings)  # Default quality score
            
        elif request.or_faces is not None:
            # Generate embeddings from images
            for face_base64 in request.or_faces:
                # Decode base64 image
                image_data = base64.b64decode(face_base64)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    continue  # Skip invalid images
                
                # Generate embedding
                embedding = face_embedder.generate_embedding(image)
                embeddings.append(embedding)
                
                # Assess quality
                quality = face_embedder.assess_quality(image)
                quality_scores.append(quality)
        else:
            raise HTTPException(status_code=400, detail="Must provide either embeddings or or_faces")
        
        if not embeddings:
            raise HTTPException(status_code=400, detail="No valid embeddings to register")
        
        # Ensure the collection exists
        milvus_client.create_collection(request.collection_name)
        
        # Insert embeddings into Milvus
        embedding_ids = milvus_client.insert_embeddings(
            user_id=request.user_id,
            embeddings=embeddings,
            qualities=quality_scores,
            collection_name=request.collection_name
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "success": len(embedding_ids) > 0,
            "embedding_ids": embedding_ids,
            "quality_scores": quality_scores,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering face: {str(e)}")