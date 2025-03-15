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
model_path = "/home/suwit/FaceSocial/ai-services/face-recognition/app/models/facenet/20180402-114759.pb"
if not os.path.exists(model_path):
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

@router.post("/compare-multiple", response_model=FaceComparisonResponse)
async def compare_faces_multiple(request: dict):
    """
    Compare a face with multiple reference faces and find the best match.
    
    Parameters:
    - query_face: Base64 encoded query face image
    - reference_faces: List of base64 encoded reference face images
    - method: Comparison method ("max" or "average")
    
    Returns:
    - similarity: Similarity score between faces (0-1)
    - is_same_person: Whether the faces belong to the same person
    - threshold_used: Threshold used for determining if same person
    - processing_time_ms: Processing time in milliseconds
    """
    try:
        start_time = time.time()
        
        # Validate request
        if 'query_face' not in request or 'reference_faces' not in request:
            raise HTTPException(status_code=400, detail="Must provide query_face and reference_faces")
        
        if not isinstance(request['reference_faces'], list) or len(request['reference_faces']) == 0:
            raise HTTPException(status_code=400, detail="reference_faces must be a non-empty list")
        
        # Get method (default: max)
        method = request.get('method', 'max').lower()
        if method not in ['max', 'average']:
            raise HTTPException(status_code=400, detail="method must be 'max' or 'average'")
        
        # Decode query face
        query_base64 = request['query_face']
        query_image_data = base64.b64decode(query_base64)
        query_nparr = np.frombuffer(query_image_data, np.uint8)
        query_image = cv2.imdecode(query_nparr, cv2.IMREAD_COLOR)
        
        if query_image is None:
            raise HTTPException(status_code=400, detail="Invalid query image data")
        
        # Generate embedding for query face
        query_embedding = face_embedder.generate_embedding(query_image)
        
        # Decode and generate embeddings for all reference faces
        reference_embeddings = []
        for ref_base64 in request['reference_faces']:
            ref_image_data = base64.b64decode(ref_base64)
            ref_nparr = np.frombuffer(ref_image_data, np.uint8)
            ref_image = cv2.imdecode(ref_nparr, cv2.IMREAD_COLOR)
            
            if ref_image is not None:
                ref_embedding = face_embedder.generate_embedding(ref_image)
                reference_embeddings.append(ref_embedding)
        
        # Calculate similarity based on method
        if method == 'max':
            similarity = max(face_embedder.calculate_similarity(query_embedding, ref_emb) for ref_emb in reference_embeddings)
        else:  # average
            similarity = np.mean([face_embedder.calculate_similarity(query_embedding, ref_emb) for ref_emb in reference_embeddings])
        
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

@router.post("/compare-top-n", response_model=FaceComparisonResponse)
async def compare_faces_top_n(request: dict):
    """
    Compare a face with multiple reference faces using Top-N Average method.
    
    Parameters:
    - query_face: Base64 encoded query face image
    - reference_faces: List of base64 encoded reference face images
    - top_n: Number of best matches to use for average (default: 3)
    - threshold: Similarity threshold to determine if same person (default: 0.65)
    
    Returns:
    - similarity: Top-N Average similarity score between faces (0-1)
    - is_same_person: Whether the faces belong to the same person
    - threshold_used: Threshold used for determining if same person
    - processing_time_ms: Processing time in milliseconds
    """
    try:
        start_time = time.time()
        
        # Validate request
        if 'query_face' not in request or 'reference_faces' not in request:
            raise HTTPException(status_code=400, detail="Must provide query_face and reference_faces")
        
        if not isinstance(request['reference_faces'], list) or len(request['reference_faces']) == 0:
            raise HTTPException(status_code=400, detail="reference_faces must be a non-empty list")
        
        # Get top_n parameter (default: 3)
        top_n = request.get('top_n', 3)
        if not isinstance(top_n, int) or top_n <= 0:
            raise HTTPException(status_code=400, detail="top_n must be a positive integer")
        
        # Get threshold parameter (default: 0.65)
        threshold = request.get('threshold', DEFAULT_SIMILARITY_THRESHOLD)
        
        # Decode query face
        query_base64 = request['query_face']
        query_image_data = base64.b64decode(query_base64)
        query_nparr = np.frombuffer(query_image_data, np.uint8)
        query_image = cv2.imdecode(query_nparr, cv2.IMREAD_COLOR)
        
        if query_image is None:
            raise HTTPException(status_code=400, detail="Invalid query image data")
        
        # Generate embedding for query face
        query_embedding = face_embedder.generate_embedding(query_image)
        
        # Decode and generate embeddings for all reference faces
        reference_embeddings = []
        for ref_base64 in request['reference_faces']:
            ref_image_data = base64.b64decode(ref_base64)
            ref_nparr = np.frombuffer(ref_image_data, np.uint8)
            ref_image = cv2.imdecode(ref_nparr, cv2.IMREAD_COLOR)
            
            if ref_image is not None:
                ref_embedding = face_embedder.generate_embedding(ref_image)
                reference_embeddings.append(ref_embedding)
        
        # Calculate Top-N Average similarity
        similarity = face_embedder.calculate_top_n_average_similarity(
            query_embedding, 
            reference_embeddings,
            top_n=top_n
        )
        
        # Determine if same person based on threshold
        is_same_person = similarity >= threshold
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "similarity": similarity,
            "is_same_person": is_same_person,
            "threshold_used": threshold,
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
    
    from fastapi.responses import HTMLResponse  # เพิ่มบรรทัดนี้ที่ด้านบนของไฟล์ ในส่วน imports

# Helper function for image enhancement
def improve_image_quality(image: np.ndarray) -> np.ndarray:
    """
    Improve image quality for better face recognition.
    
    Parameters:
    - image: Input image
    
    Returns:
    - Enhanced image
    """
    if image is None:
        return None
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        equalized = cv2.equalizeHist(gray)
        
        # Convert back to BGR
        equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        
        # Blend with original for more natural look (70% original, 30% equalized)
        enhanced = cv2.addWeighted(image, 0.7, equalized_bgr, 0.3, 0)
        
        return enhanced
    except Exception as e:
        print(f"Error enhancing image: {str(e)}")
        return image  # Return original if enhancement fails

# Helper function to calculate confidence level
def calculate_confidence_level(similarity: float, threshold: float, gender_warning: bool) -> str:
    """
    Calculate a human-readable confidence level based on similarity score.
    
    Parameters:
    - similarity: Similarity score
    - threshold: Threshold used
    - gender_warning: Whether there was a gender mismatch warning
    
    Returns:
    - Confidence level string
    """
    margin = similarity - threshold
    
    if gender_warning:
        # Lower confidence if gender mismatch
        if margin >= 0.1:
            return "MEDIUM"
        elif margin >= 0:
            return "LOW"
        else:
            return "VERY LOW"
    else:
        # Normal confidence calculation
        if margin >= 0.15:
            return "VERY HIGH"
        elif margin >= 0.08:
            return "HIGH"
        elif margin >= 0.03:
            return "MEDIUM"
        elif margin >= 0:
            return "LOW"
        else:
            return "VERY LOW"

@router.post("/smart-compare", response_model=Dict[str, Any])
async def smart_compare_faces(request: dict):
    """
    Compare faces using smart adaptive comparison with multiple techniques.
    
    Parameters:
    - query_face: Base64 encoded query face image
    - reference_faces: List of base64 encoded reference face images
    - threshold: Optional custom threshold (default: adaptive)
    - top_n: Number of best matches to use (default: 3)
    
    Returns:
    - Enhanced comparison results with detailed information
    """
    try:
        start_time = time.time()
        
        # Validate request
        if 'query_face' not in request or 'reference_faces' not in request:
            raise HTTPException(status_code=400, detail="Must provide query_face and reference_faces")
        
        if not isinstance(request['reference_faces'], list) or len(request['reference_faces']) == 0:
            raise HTTPException(status_code=400, detail="reference_faces must be a non-empty list")
        
        # Get parameters
        top_n = request.get('top_n', 3)
        custom_threshold = request.get('threshold', None)
        
        # Decode query face
        query_base64 = request['query_face']
        query_image_data = base64.b64decode(query_base64)
        query_nparr = np.frombuffer(query_image_data, np.uint8)
        query_image = cv2.imdecode(query_nparr, cv2.IMREAD_COLOR)
        
        if query_image is None:
            raise HTTPException(status_code=400, detail="Invalid query image data")
        
        # Enhanced preprocessing for query image
        query_image = improve_image_quality(query_image)
        
        # Generate embedding for query face
        query_embedding = face_embedder.generate_embedding(query_image)
        
        # Process reference faces
        reference_embeddings = []
        reference_images = []
        reference_qualities = []
        
        for ref_base64 in request['reference_faces']:
            ref_image_data = base64.b64decode(ref_base64)
            ref_nparr = np.frombuffer(ref_image_data, np.uint8)
            ref_image = cv2.imdecode(ref_nparr, cv2.IMREAD_COLOR)
            
            if ref_image is not None:
                # Enhance image
                ref_image = improve_image_quality(ref_image)
                
                # Assess quality
                quality_score = face_embedder.assess_quality(ref_image)
                reference_qualities.append(quality_score)
                
                # Only use images with reasonable quality
                if quality_score > 0.3:  # Very low quality threshold
                    ref_embedding = face_embedder.generate_embedding(ref_image)
                    reference_embeddings.append(ref_embedding)
                    reference_images.append(ref_image)
        
        # Different comparison methods
        result = {}
        
        # 1. Weighted Top-N Average (main method)
        weighted_result = face_embedder.calculate_weighted_top_n_average_similarity(
            query_embedding, 
            reference_embeddings,
            reference_images,
            top_n=top_n
        )
        
        # 2. Standard Top-N Average (backup)
        top_n_similarity = face_embedder.calculate_top_n_average_similarity(
            query_embedding, 
            reference_embeddings,
            top_n=top_n
        )
        
        # 3. Maximum Similarity (for comparison)
        max_similarity = face_embedder.calculate_similarity_with_multiple(
            query_embedding, 
            reference_embeddings
        ) if reference_embeddings else 0.0
        
        # Determine threshold (adaptive or custom)
        threshold = custom_threshold
        if threshold is None:
            threshold = face_embedder.adaptive_threshold(
                len(reference_embeddings),
                reference_qualities
            )
        
        # Make final determination
        is_same_person = weighted_result["similarity"] >= threshold
        
        # Check for gender mismatch as a warning flag
        gender_warning = False
        if weighted_result["gender_match"] is not None and weighted_result["gender_match"] is False:
            gender_warning = True
            # If strong gender mismatch and similarity is borderline, adjust decision
            if weighted_result["similarity"] < (threshold + 0.05):
                is_same_person = False
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Prepare detailed response
        response = {
            "primary_similarity": weighted_result["similarity"],
            "is_same_person": is_same_person,
            "threshold_used": threshold,
            "processing_time_ms": processing_time,
            "reference_count": len(reference_embeddings),
            "details": {
                "top_n_similarity": top_n_similarity,
                "max_similarity": max_similarity,
                "top_similarities": weighted_result["top_similarities"],
                "quality_scores": reference_qualities,
                "gender_warning": gender_warning,
                "adaptive_threshold": threshold if custom_threshold is None else None
            },
            "confidence_level": calculate_confidence_level(
                weighted_result["similarity"], 
                threshold,
                gender_warning
            )
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in smart comparison: {str(e)}")

@router.post("/ensemble-compare", response_model=Dict[str, Any])
async def ensemble_compare_faces(request: dict):
    """
    Compare faces using ensemble of multiple face recognition models.
    
    Parameters:
    - query_face: Base64 encoded query face image
    - reference_faces: List of base64 encoded reference face images
    - threshold: Optional custom threshold (default: 0.63)
    
    Returns:
    - Enhanced comparison results with ensemble model details
    """
    try:
        start_time = time.time()
        
        # Validate request
        if 'query_face' not in request or 'reference_faces' not in request:
            raise HTTPException(status_code=400, detail="Must provide query_face and reference_faces")
        
        if not isinstance(request['reference_faces'], list) or len(request['reference_faces']) == 0:
            raise HTTPException(status_code=400, detail="reference_faces must be a non-empty list")
        
        # Get threshold parameter (default: 0.63)
        threshold = request.get('threshold', 0.63)
        
        # Decode query face
        query_base64 = request['query_face']
        query_image_data = base64.b64decode(query_base64)
        query_nparr = np.frombuffer(query_image_data, np.uint8)
        query_image = cv2.imdecode(query_nparr, cv2.IMREAD_COLOR)
        
        if query_image is None:
            raise HTTPException(status_code=400, detail="Invalid query image data")
        
        # Enhanced preprocessing for query image
        query_image = improve_image_quality(query_image)
        
        # Generate ensemble embedding for query face
        query_ensemble_embedding = face_embedder.generate_ensemble_embedding(query_image)
        
        # Process reference faces
        reference_ensemble_embeddings = []
        reference_qualities = []
        
        for ref_base64 in request['reference_faces']:
            ref_image_data = base64.b64decode(ref_base64)
            ref_nparr = np.frombuffer(ref_image_data, np.uint8)
            ref_image = cv2.imdecode(ref_nparr, cv2.IMREAD_COLOR)
            
            if ref_image is not None:
                # Enhance image
                ref_image = improve_image_quality(ref_image)
                
                # Assess quality
                quality_score = face_embedder.assess_quality(ref_image)
                reference_qualities.append(quality_score)
                
                # Generate ensemble embedding
                ref_ensemble_embedding = face_embedder.generate_ensemble_embedding(ref_image)
                reference_ensemble_embeddings.append(ref_ensemble_embedding)
        
        # Calculate similarities for each reference
        similarities = []
        model_details = {}
        
        for ref_ensemble_embedding in reference_ensemble_embeddings:
            # Get detailed similarity results
            similarity_result = face_embedder.ensemble.calculate_similarity(
                query_ensemble_embedding,
                ref_ensemble_embedding
            )
            
            similarities.append(similarity_result["ensemble_similarity"])
            
            # Collect model-specific details
            for model_name, similarity in similarity_result["model_similarities"].items():
                if (model_name not in model_details):
                    model_details[model_name] = []
                model_details[model_name].append(similarity)
        
        # Calculate average similarities for each model
        model_avg_similarities = {}
        for model_name, sims in model_details.items():
            model_avg_similarities[model_name] = sum(sims) / len(sims) if sims else 0.0
        
        # Sort and get top similarities
        similarities.sort(reverse=True)
        top_similarities = similarities[:min(3, len(similarities))]
        
        # Final similarity score (can be max or average of top 3)
        final_similarity = top_similarities[0] if top_similarities else 0.0
        
        # Determine if same person
        is_same_person = final_similarity >= threshold
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Prepare response
        response = {
            "similarity": final_similarity,
            "is_same_person": is_same_person,
            "threshold_used": threshold,
            "processing_time_ms": processing_time,
            "reference_count": len(reference_ensemble_embeddings),
            "top_similarities": top_similarities,
            "models_used": list(face_embedder.ensemble.models.keys()),
            "model_weights": face_embedder.ensemble.model_weights,
            "model_similarities": model_avg_similarities,
            "confidence_level": calculate_confidence_level(final_similarity, threshold, False)
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in ensemble comparison: {str(e)}")

# เพิ่ม endpoint นี้ที่ด้านล่างของไฟล์ (ต่อจาก endpoint อื่นๆ)
@router.get("/demo")
async def face_recognition_demo():
    """
    Show a demo page for testing Face Recognition Service.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition Demo</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; max-width: 1200px; margin: 0 auto; }
            h1, h2, h3 { color: #333; }
            .container { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; flex: 1; min-width: 300px; background: #f9f9f9; }
            .result-box { margin-top: 15px; padding: 10px; border: 1px solid #eee; border-radius: 5px; background: white; }
            img { max-width: 100%; border: 1px solid #eee; border-radius: 4px; }
            button { padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px; }
            button:hover { background-color: #45a049; }
            input, select { padding: 8px; margin: 5px 0; width: 100%; box-sizing: border-box; }
            .image-preview { height: 200px; display: flex; align-items: center; justify-content: center; }
            .log { font-family: monospace; font-size: 12px; overflow: auto; max-height: 200px; background: #f5f5f5; padding: 10px; margin-top: 10px; border-radius: 4px; }
            .tabs { display: flex; border-bottom: 1px solid #ccc; margin-bottom: 20px; }
            .tab { padding: 10px 20px; cursor: pointer; }
            .tab.active { background: #f1f1f1; border: 1px solid #ccc; border-bottom: none; border-radius: 4px 4px 0 0; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .similarity-high { color: green; font-weight: bold; }
            .similarity-med { color: orange; }
            .similarity-low { color: red; }
        </style>
    </head>
    <body>
        <h1>Face Recognition Demo</h1>
        
        <div class="tabs">
            <div class="tab active" onclick="openTab(event, 'embed-tab')">Generate Embedding</div>
            <div class="tab" onclick="openTab(event, 'compare-tab')">Compare Faces</div>
            <div class="tab" onclick="openTab(event, 'register-tab')">Register Face</div>
            <div class="tab" onclick="openTab(event, 'identify-tab')">Identify Face</div>
        </div>
        
        <!-- Generate Embedding Tab -->
        <div id="embed-tab" class="tab-content active">
            <h2>Generate Face Embedding</h2>
            <div class="container">
                <div class="card">
                    <h3>Input Face</h3>
                    <input type="file" id="embed-image-upload" accept="image/*" onchange="previewImage('embed-image-upload', 'embed-preview')">
                    <div class="image-preview">
                        <img id="embed-preview" src="" alt="Preview" style="display: none">
                    </div>
                    <button onclick="generateEmbedding()">Generate Embedding</button>
                </div>
                
                <div class="card">
                    <h3>Embedding Result</h3>
                    <div class="result-box" id="embed-result">
                        <p>Embedding will appear here...</p>
                    </div>
                    <h4>Quality Score: <span id="quality-score">-</span></h4>
                    <div class="log" id="embed-log"></div>
                </div>
            </div>
        </div>
        
        <!-- Compare Faces Tab -->
        <div id="compare-tab" class="tab-content">
            <h2>Compare Two Faces</h2>
            <div class="container">
                <div class="card">
                    <h3>First Face</h3>
                    <input type="file" id="face1-upload" accept="image/*" onchange="previewImage('face1-upload', 'face1-preview')">
                    <div class="image-preview">
                        <img id="face1-preview" src="" alt="Preview" style="display: none">
                    </div>
                </div>
                
                <div class="card">
                    <h3>Second Face</h3>
                    <input type="file" id="face2-upload" accept="image/*" onchange="previewImage('face2-upload', 'face2-preview')">
                    <div class="image-preview">
                        <img id="face2-preview" src="" alt="Preview" style="display: none">
                    </div>
                </div>
                
                <div class="card">
                    <h3>Comparison Result</h3>
                    <button onclick="compareFaces()">Compare Faces</button>
                    <div class="result-box" id="compare-result">
                        <p>Comparison results will appear here...</p>
                    </div>
                    <div class="log" id="compare-log"></div>
                </div>
            </div>
        </div>
        
        <!-- Register Face Tab -->
        <div id="register-tab" class="tab-content">
            <h2>Register Face</h2>
            <div class="container">
                <div class="card">
                    <h3>Face to Register</h3>
                    <input type="file" id="register-image-upload" accept="image/*" onchange="previewImage('register-image-upload', 'register-preview')">
                    <div class="image-preview">
                        <img id="register-preview" src="" alt="Preview" style="display: none">
                    </div>
                    
                    <h3>User Information</h3>
                    <label for="user-id">User ID:</label>
                    <input type="number" id="user-id" placeholder="Enter user ID (e.g., 123)">
                    
                    <label for="collection-name">Collection:</label>
                    <input type="text" id="collection-name" value="users" placeholder="Collection name">
                    
                    <button onclick="registerFace()">Register Face</button>
                </div>
                
                <div class="card">
                    <h3>Registration Result</h3>
                    <div class="result-box" id="register-result">
                        <p>Registration results will appear here...</p>
                    </div>
                    <div class="log" id="register-log"></div>
                </div>
            </div>
        </div>
        
        <!-- Identify Face Tab -->
        <div id="identify-tab" class="tab-content">
            <h2>Identify Face</h2>
            <div class="container">
                <div class="card">
                    <h3>Face to Identify</h3>
                    <input type="file" id="identify-image-upload" accept="image/*" onchange="previewImage('identify-image-upload', 'identify-preview')">
                    <div class="image-preview">
                        <img id="identify-preview" src="" alt="Preview" style="display: none">
                    </div>
                    
                    <h3>Search Parameters</h3>
                    <label for="top-k">Top K Results:</label>
                    <input type="number" id="top-k" value="5" min="1" max="20">
                    
                    <label for="min-similarity">Minimum Similarity:</label>
                    <input type="range" id="min-similarity" min="0" max="100" value="75" oninput="updateSimilarityValue()">
                    <span id="min-similarity-value">0.75</span>
                    
                    <label for="identify-collection">Collection:</label>
                    <input type="text" id="identify-collection" value="users" placeholder="Collection name">
                    
                    <button onclick="identifyFace()">Identify Face</button>
                </div>
                
                <div class="card">
                    <h3>Identification Results</h3>
                    <div class="result-box" id="identify-result">
                        <p>Identification results will appear here...</p>
                    </div>
                    <div class="log" id="identify-log"></div>
                </div>
            </div>
        </div>
        
        <script>
            // Tab functionality
            function openTab(evt, tabName) {
                var tabcontents = document.getElementsByClassName("tab-content");
                for (var i = 0; i < tabcontents.length; i++) {
                    tabcontents[i].classList.remove("active");
                }
                
                var tabs = document.getElementsByClassName("tab");
                for (var i = 0; i < tabs.length; i++) {
                    tabs[i].classList.remove("active");
                }
                
                document.getElementById(tabName).classList.add("active");
                evt.currentTarget.classList.add("active");
            }
            
            // Image preview functionality
            function previewImage(inputId, previewId) {
                const file = document.getElementById(inputId).files[0];
                const preview = document.getElementById(previewId);
                
                if (file) {
                    const reader = new FileReader();
                    
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }
                    
                    reader.readAsDataURL(file);
                }
            }
            
            // Log functionality
            function log(elementId, message) {
                const logElement = document.getElementById(elementId);
                const timestamp = new Date().toLocaleTimeString();
                logElement.innerHTML += `[${timestamp}] ${message}<br>`;
                logElement.scrollTop = logElement.scrollHeight;
            }
            
            // Update similarity slider value
            function updateSimilarityValue() {
                const slider = document.getElementById('min-similarity');
                const value = slider.value / 100;
                document.getElementById('min-similarity-value').textContent = value.toFixed(2);
            }
            
            // Generate embedding functionality
            async function generateEmbedding() {
                const fileInput = document.getElementById('embed-image-upload');
                if (!fileInput.files || fileInput.files.length === 0) {
                    alert('Please select an image first');
                    return;
                }
                
                const file = fileInput.files[0];
                const reader = new FileReader();
                
                reader.onload = async function(e) {
                    try {
                        log('embed-log', 'Generating embedding...');
                        
                        // Get base64 data (remove data URL prefix)
                        const base64Image = e.target.result.split(',')[1];
                        
                        // Call API
                        const response = await fetch('/v1/face/embed', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                image: base64Image,
                                detect_and_align: true
                            })
                        });
                        
                        if (!response.ok) {
                            const errorText = await response.text();
                            throw new Error(`API error: ${errorText}`);
                        }
                        
                        const data = await response.json();
                        log('embed-log', `Generated embedding with size ${data.embedding_size}`);
                        
                        // Display results
                        document.getElementById('quality-score').textContent = data.quality_score.toFixed(2);
                        
                        // Show abbreviated embedding vector
                        const embedding = data.embedding;
                        const embedStr = JSON.stringify(embedding.slice(0, 5)) + 
                                        ` ... (${embedding.length - 10} more values) ... ` + 
                                        JSON.stringify(embedding.slice(-5));
                        
                        document.getElementById('embed-result').innerHTML = `
                            <p><strong>Embedding Vector:</strong> ${embedStr}</p>
                            <p><strong>Vector Size:</strong> ${data.embedding_size}</p>
                            <p><strong>Processing Time:</strong> ${data.processing_time_ms.toFixed(2)} ms</p>
                        `;
                        
                    } catch (error) {
                        log('embed-log', `ERROR: ${error.message}`);
                        document.getElementById('embed-result').innerHTML = `
                            <p style="color: red;">Error: ${error.message}</p>
                        `;
                    }
                };
                
                reader.readAsDataURL(file);
            }
            
            // Compare faces functionality
            async function compareFaces() {
                const face1Input = document.getElementById('face1-upload');
                const face2Input = document.getElementById('face2-upload');
                
                if (!face1Input.files || !face1Input.files.length === 0 ||
                    !face2Input.files || !face2Input.files.length === 0) {
                    alert('Please select both face images');
                    return;
                }
                
                const face1File = face1Input.files[0];
                const face2File = face2Input.files[0];
                
                // Read first face
                const reader1 = new FileReader();
                reader1.onload = function(e1) {
                    const base64Image1 = e1.target.result.split(',')[1];
                    
                    // Read second face
                    const reader2 = new FileReader();
                    reader2.onload = async function(e2) {
                        try {
                            log('compare-log', 'Comparing faces...');
                            
                            const base64Image2 = e2.target.result.split(',')[1];
                            
                            // Call API
                            const response = await fetch('/v1/face/compare', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({
                                    or_face1: base64Image1,
                                    or_face2: base64Image2
                                })
                            });
                            
                            if (!response.ok) {
                                const errorText = await response.text();
                                throw new Error(`API error: ${errorText}`);
                            }
                            
                            const data = await response.json();
                            log('compare-log', `Comparison complete: similarity = ${data.similarity.toFixed(4)}`);
                            
                            // Determine similarity class for styling
                            let similarityClass = 'similarity-low';
                            if (data.similarity >= 0.85) {
                                similarityClass = 'similarity-high';
                            } else if (data.similarity >= 0.75) {
                                similarityClass = 'similarity-med';
                            }
                            
                            // Display results
                            document.getElementById('compare-result').innerHTML = `
                                <p><strong>Similarity Score:</strong> <span class="${similarityClass}">${data.similarity.toFixed(4)}</span></p>
                                <p><strong>Same Person:</strong> ${data.is_same_person ? 'Yes ✓' : 'No ✗'}</p>
                                <p><strong>Threshold Used:</strong> ${data.threshold_used}</p>
                                <p><strong>Processing Time:</strong> ${data.processing_time_ms.toFixed(2)} ms</p>
                            `;
                            
                        } catch (error) {
                            log('compare-log', `ERROR: ${error.message}`);
                            document.getElementById('compare-result').innerHTML = `
                                <p style="color: red;">Error: ${error.message}</p>
                            `;
                        }
                    };
                    
                    reader2.readAsDataURL(face2File);
                };
                
                reader1.readAsDataURL(face1File);
            }
            
            // Register face functionality
            async function registerFace() {
                const fileInput = document.getElementById('register-image-upload');
                const userIdInput = document.getElementById('user-id');
                const collectionInput = document.getElementById('collection-name');
                
                if (!fileInput.files || fileInput.files.length === 0) {
                    alert('Please select an image first');
                    return;
                }
                
                const userId = parseInt(userIdInput.value);
                if (isNaN(userId) || userId <= 0) {
                    alert('Please enter a valid user ID (positive number)');
                    return;
                }
                
                const collection = collectionInput.value.trim();
                if (!collection) {
                    alert('Please enter a collection name');
                    return;
                }
                
                const file = fileInput.files[0];
                const reader = new FileReader();
                
                reader.onload = async function(e) {
                    try {
                        log('register-log', `Registering face for user ID: ${userId}...`);
                        
                        // Get base64 data
                        const base64Image = e.target.result.split(',')[1];
                        
                        // Call API
                        const response = await fetch('/v1/face/register', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                user_id: userId,
                                or_faces: [base64Image],
                                collection_name: collection
                            })
                        });
                        
                        if (!response.ok) {
                            const errorText = await response.text();
                            throw new Error(`API error: ${errorText}`);
                        }
                        
                        const data = await response.json();
                        log('register-log', `Registration ${data.success ? 'successful' : 'failed'}`);
                        
                        // Display results
                        document.getElementById('register-result').innerHTML = `
                            <p><strong>Registration:</strong> ${data.success ? 'Successful ✓' : 'Failed ✗'}</p>
                            <p><strong>Embedding IDs:</strong> ${data.embedding_ids.join(', ')}</p>
                            <p><strong>Quality Scores:</strong> ${data.quality_scores.map(q => q.toFixed(2)).join(', ')}</p>
                            <p><strong>Processing Time:</strong> ${data.processing_time_ms.toFixed(2)} ms</p>
                        `;
                        
                    } catch (error) {
                        log('register-log', `ERROR: ${error.message}`);
                        document.getElementById('register-result').innerHTML = `
                            <p style="color: red;">Error: ${error.message}</p>
                        `;
                    }
                };
                
                reader.readAsDataURL(file);
            }
            
            // Identify face functionality
            async function identifyFace() {
                const fileInput = document.getElementById('identify-image-upload');
                const topK = parseInt(document.getElementById('top-k').value);
                const minSimilarity = parseFloat(document.getElementById('min-similarity').value) / 100;
                const collection = document.getElementById('identify-collection').value.trim();
                
                if (!fileInput.files || fileInput.files.length === 0) {
                    alert('Please select an image first');
                    return;
                }
                
                if (isNaN(topK) || topK <= 0) {
                    alert('Please enter a valid number for Top K');
                    return;
                }
                
                if (!collection) {
                    alert('Please enter a collection name');
                    return;
                }
                
                const file = fileInput.files[0];
                const reader = new FileReader();
                
                reader.onload = async function(e) {
                    try {
                        log('identify-log', 'Identifying face...');
                        
                        // Get base64 data
                        const base64Image = e.target.result.split(',')[1];
                        
                        // Call API
                        const response = await fetch('/v1/face/identify', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                or_face: base64Image,
                                top_k: topK,
                                min_similarity: minSimilarity,
                                collection_name: collection
                            })
                        });
                        
                        if (!response.ok) {
                            const errorText = await response.text();
                            throw new Error(`API error: ${errorText}`);
                        }
                        
                        const data = await response.json();
                        log('identify-log', `Found ${data.matches.length} matches`);
                        
                        // Display results
                        let resultHtml = '';
                        
                        if (data.matches.length === 0) {
                            resultHtml = '<p>No matches found.</p>';
                        } else {
                            resultHtml = `
                                <p>Found ${data.matches.length} potential matches:</p>
                                <table>
                                    <thead>
                                        <tr>
                                            <th>User ID</th>
                                            <th>Similarity</th>
                                            <th>Embedding ID</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                            `;
                            
                            for (const match of data.matches) {
                                // Determine similarity class for styling
                                let similarityClass = 'similarity-low';
                                if (match.similarity >= 0.85) {
                                    similarityClass = 'similarity-high';
                                } else if (match.similarity >= 0.75) {
                                    similarityClass = 'similarity-med';
                                }
                                
                                resultHtml += `
                                    <tr>
                                        <td>${match.user_id}</td>
                                        <td class="${similarityClass}">${match.similarity.toFixed(4)}</td>
                                        <td>${match.embedding_id}</td>
                                    </tr>
                                `;
                            }
                            
                            resultHtml += `
                                    </tbody>
                                </table>
                                <p><strong>Processing Time:</strong> ${data.processing_time_ms.toFixed(2)} ms</p>
                            `;
                        }
                        
                        document.getElementById('identify-result').innerHTML = resultHtml;
                        
                    } catch (error) {
                        log('identify-log', `ERROR: ${error.message}`);
                        document.getElementById('identify-result').innerHTML = `
                            <p style="color: red;">Error: ${error.message}</p>
                        `;
                    }
                };
                
                reader.readAsDataURL(file);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.get("/demo-multiple")
async def face_recognition_demo_multiple():
    """
    Show a demo page for testing Face Recognition with multiple faces.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition Multi-Face Demo</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; max-width: 1200px; margin: 0 auto; }
            h1, h2, h3 { color: #333; }
            .container { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; flex: 1; min-width: 300px; background: #f9f9f9; }
            .result-box { margin-top: 15px; padding: 10px; border: 1px solid #eee; border-radius: 5px; background: white; }
            img { max-width: 100%; border: 1px solid #eee; border-radius: 4px; max-height: 200px; object-fit: cover; }
            button { padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px; }
            button:hover { background-color: #45a049; }
            input, select { padding: 8px; margin: 5px 0; width: 100%; box-sizing: border-box; }
            .face-gallery { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px; }
            .face-item { position: relative; width: 100px; height: 100px; margin-bottom: 10px; }
            .face-item img { width: 100%; height: 100%; object-fit: cover; }
            .face-item .remove { position: absolute; top: -5px; right: -5px; background: red; color: white; border-radius: 50%; width: 20px; height: 20px; line-height: 20px; text-align: center; cursor: pointer; }
            .log { font-family: monospace; font-size: 12px; overflow: auto; max-height: 200px; background: #f5f5f5; padding: 10px; margin-top: 10px; border-radius: 4px; }
            .similarity-high { color: green; font-weight: bold; }
            .similarity-med { color: orange; }
            .similarity-low { color: red; }
            .details-section { margin-top: 15px; padding: 10px; background: #f8f8f8; border-radius: 5px; }
            .warning { color: #e74c3c; }
            .confidence-veryhigh { color: #2ecc71; font-weight: bold; }
            .confidence-high { color: #27ae60; font-weight: bold; }
            .confidence-medium { color: #f39c12; }
            .confidence-low { color: #e67e22; }
            .confidence-verylow { color: #e74c3c; }
            .model-info {
                margin-top: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
                border: 1px solid #dee2e6;
            }
            .model-badge {
                display: inline-block;
                padding: 3px 8px;
                margin: 2px;
                border-radius: 12px;
                font-size: 12px;
                color: white;
                background-color: #6c757d;
            }
            .model-facenet { background-color: #007bff; }
            .model-arcface { background-color: #28a745; }
            .model-cosface { background-color: #fd7e14; }
            .refresh-btn {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 12px;
            }
            .refresh-btn:hover { background-color: #5a6268; }
        </style>
    </head>
    <body>
        <h1>Face Recognition with Multiple Reference Faces</h1>
        
        <!-- Model Info Section -->
        <div class="model-info" id="models-info">
            <h3>Models Status <button class="refresh-btn" onclick="checkModelsStatus()">Refresh</button></h3>
            <p>Loading model information...</p>
        </div>
        
        <div class="container">
            <div class="card">
                <h3>Query Face</h3>
                <input type="file" id="query-face-upload" accept="image/*" onchange="previewImage('query-face-upload', 'query-face-preview')">
                <div class="image-preview">
                    <img id="query-face-preview" src="" alt="Preview" style="display: none">
                </div>
            </div>
            
            <div class="card">
                <h3>Reference Faces (Upload up to 5)</h3>
                <input type="file" id="reference-face-upload" accept="image/*" onchange="addReferenceImage()" multiple>
                <div class="face-gallery" id="reference-faces-gallery"></div>
                <p><strong>Count:</strong> <span id="face-count">0</span>/5</p>
                <button onclick="clearReferenceImages()">Clear All</button>
            </div>
            
            <div class="card">
                <h3>Comparison Settings</h3>
                <label for="comparison-method">Comparison Method:</label>
                <select id="comparison-method">
                    <option value="max">Maximum Similarity (Best Match)</option>
                    <option value="average">Average Similarity</option>
                    <option value="top_n">Top-N Average (N=3)</option>
                    <option value="smart">Smart Adaptive Comparison</option>
                    <option value="ensemble">Model Ensemble (Multiple Models)</option>
                </select>
                <br>
                <button onclick="compareWithMultiple()">Compare Face</button>
                
                <div class="result-box" id="comparison-result">
                    <p>Comparison results will appear here...</p>
                </div>
                <div class="log" id="comparison-log"></div>
            </div>
        </div>
        
        <script>
            // Global variables
            const maxFaces = 5;
            let referenceFaces = [];
            
            // Preview image function
            function previewImage(inputId, previewId) {
                const file = document.getElementById(inputId).files[0];
                const preview = document.getElementById(previewId);
                
                if (file) {
                    const reader = new FileReader();
                    
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }
                    
                    reader.readAsDataURL(file);
                }
            }
            
            // Add reference image
            function addReferenceImage() {
                const fileInput = document.getElementById('reference-face-upload');
                if (!fileInput.files || fileInput.files.length === 0) return;
                
                for (const file of fileInput.files) {
                    if (referenceFaces.length >= maxFaces) {
                        alert(`Maximum ${maxFaces} reference faces allowed. Please remove some faces first.`);
                        break;
                    }
                    
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const faceId = Date.now() + Math.floor(Math.random() * 1000);
                        referenceFaces.push({
                            id: faceId,
                            base64: e.target.result.split(',')[1],
                            dataUrl: e.target.result
                        });
                        
                        updateReferenceGallery();
                    };
                    
                    reader.readAsDataURL(file);
                }
                
                // Reset file input
                fileInput.value = "";
            }
            
            // Update reference gallery
            function updateReferenceGallery() {
                const gallery = document.getElementById('reference-faces-gallery');
                const counter = document.getElementById('face-count');
                
                gallery.innerHTML = '';
                counter.textContent = referenceFaces.length;
                
                referenceFaces.forEach(face => {
                    const faceItem = document.createElement('div');
                    faceItem.className = 'face-item';
                    
                    const img = document.createElement('img');
                    img.src = face.dataUrl;
                    img.alt = 'Reference face';
                    
                    const removeBtn = document.createElement('div');
                    removeBtn.className = 'remove';
                    removeBtn.textContent = 'X';
                    removeBtn.onclick = () => removeReferenceImage(face.id);
                    
                    faceItem.appendChild(img);
                    faceItem.appendChild(removeBtn);
                    gallery.appendChild(faceItem);
                });
            }
            
            // Remove reference image
            function removeReferenceImage(id) {
                referenceFaces = referenceFaces.filter(face => face.id !== id);
                updateReferenceGallery();
            }
            
            // Clear all reference images
            function clearReferenceImages() {
                referenceFaces = [];
                updateReferenceGallery();
            }
            
            // Log to console
            function log(message) {
                const logElement = document.getElementById('comparison-log');
                const timestamp = new Date().toLocaleTimeString();
                logElement.innerHTML += `[${timestamp}] ${message}<br>`;
                logElement.scrollTop = logElement.scrollHeight;
            }

            // Get similarity class based on value
            function getSimilarityClass(similarity) {
                if (similarity >= 0.85) return 'similarity-high';
                if (similarity >= 0.75) return 'similarity-med';
                return 'similarity-low';
            }
            
            // Compare with multiple faces
            async function compareWithMultiple() {
                const queryFileInput = document.getElementById('query-face-upload');
                if (!queryFileInput.files || queryFileInput.files.length === 0) {
                    alert('Please select a query face image');
                    return;
                }
                
                if (referenceFaces.length === 0) {
                    alert('Please upload at least one reference face');
                    return;
                }
                
                try {
                    log('Comparing face against ' + referenceFaces.length + ' reference faces...');
                    
                    // Get query face base64
                    const queryReader = new FileReader();
                    queryReader.onload = async function(e) {
                        const queryBase64 = e.target.result.split(',')[1];
                        const referenceBase64List = referenceFaces.map(face => face.base64);
                        const method = document.getElementById('comparison-method').value;
                        
                        // Determine which API endpoint to call based on the method
                        let apiEndpoint = '/v1/face/compare-multiple';
                        let requestBody = {
                            query_face: queryBase64,
                            reference_faces: referenceBase64List
                        };
                        
                        if (method === 'top_n') {
                            apiEndpoint = '/v1/face/compare-top-n';
                            requestBody.top_n = 3;  // Default to top 3 matches
                        } else if (method === 'smart') {
                            apiEndpoint = '/v1/face/smart-compare';
                            requestBody.top_n = 3;  // Default to top 3 matches
                        } else if (method === 'ensemble') {
                            apiEndpoint = '/v1/face/ensemble-compare';
                        } else {
                            requestBody.method = method;
                        }
                        
                        // Call API
                        const response = await fetch(apiEndpoint, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(requestBody)
                        });
                        
                        if (!response.ok) {
                            const errorText = await response.text();
                            throw new Error(`API error: ${errorText}`);
                        }
                        
                        const data = await response.json();
                        
                        // Process and display results based on method
                        if (method === 'smart') {
                            log(`Comparison complete: similarity = ${data.primary_similarity.toFixed(4)}`);
                            
                            document.getElementById('comparison-result').innerHTML = `
                                <p><strong>Similarity Score:</strong> <span class="${getSimilarityClass(data.primary_similarity)}">${data.primary_similarity.toFixed(4)}</span></p>
                                <p><strong>Same Person:</strong> ${data.is_same_person ? 'Yes ✓' : 'No ✗'}</p>
                                <p><strong>Confidence Level:</strong> <span class="confidence-${data.confidence_level.toLowerCase()}">${data.confidence_level}</span></p>
                                <p><strong>Threshold:</strong> ${data.threshold_used.toFixed(4)} (${data.details.adaptive_threshold ? 'Adaptive' : 'Fixed'})</p>
                                <p><strong>Reference Images:</strong> ${data.reference_count}</p>
                                <p><strong>Processing Time:</strong> ${data.processing_time_ms.toFixed(2)} ms</p>
                                ${data.details.gender_warning ? '<p class="warning"><strong>Warning:</strong> Possible gender mismatch detected</p>' : ''}
                                <div class="details-section">
                                    <h4>Additional Details</h4>
                                    <p>Max Similarity: ${data.details.max_similarity.toFixed(4)}</p>
                                    <p>Top-N Similarity: ${data.details.top_n_similarity.toFixed(4)}</p>
                                    <p>Top Similarities: ${data.details.top_similarities.map(s => s.toFixed(4)).join(', ')}</p>
                                </div>
                            `;
                        } else if (method === 'ensemble') {
                            log(`Ensemble comparison complete: similarity = ${data.similarity.toFixed(4)}`);
                            
                            document.getElementById('comparison-result').innerHTML = `
                                <p><strong>Similarity Score:</strong> <span class="${getSimilarityClass(data.similarity)}">${data.similarity.toFixed(4)}</span></p>
                                <p><strong>Same Person:</strong> ${data.is_same_person ? 'Yes ✓' : 'No ✗'}</p>
                                <p><strong>Confidence Level:</strong> <span class="confidence-${data.confidence_level.toLowerCase()}">${data.confidence_level}</span></p>
                                <p><strong>Threshold:</strong> ${data.threshold_used}</p>
                                <p><strong>Models Used:</strong> ${data.models_used.join(', ')}</p>
                                <p><strong>Processing Time:</strong> ${data.processing_time_ms.toFixed(2)} ms</p>
                                
                                <div class="details-section">
                                    <h4>Model Details</h4>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Model</th>
                                                <th>Weight</th>
                                                <th>Avg. Similarity</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${Object.entries(data.model_weights).map(([model, weight]) => `
                                                <tr>
                                                    <td>${model}</td>
                                                    <td>${weight.toFixed(2)}</td>
                                                    <td>${data.model_similarities[model]?.toFixed(4) || 'N/A'}</td>
                                                </tr>
                                            `).join('')}
                                        </tbody>
                                    </table>
                                    <p><strong>Top Similarities:</strong> ${data.top_similarities.map(s => s.toFixed(4)).join(', ')}</p>
                                </div>
                            `;
                        } else {
                            log(`Comparison complete: similarity = ${data.similarity.toFixed(4)}`);
                            















































































































    return HTMLResponse(content=html_content)    """    </html>    </body>        </script>            });                checkModelsStatus();                // Check model status                                // ...                // Original initialization code            document.addEventListener('DOMContentLoaded', function() {            // Call when page loads                        }                    });                             <p>Error loading model information: ${error}</p>`;                            `<h3>Models Status <button class="refresh-btn" onclick="checkModelsStatus()">Refresh</button></h3>                        document.getElementById('models-info').innerHTML =                         console.error('Error checking models status:', error);                    .catch(error => {                    })                        modelsInfo.innerHTML = content;                                                }                            content += '</p>';                            content += data.onnx_providers.join(', ');                            content += '<p><strong>ONNX Providers:</strong> ';                        if (data.onnx_providers && data.onnx_providers.length > 0) {                        // ONNX providers                                                content += `<p><strong>Using Ensemble:</strong> ${data.use_ensemble ? 'Yes ✓' : 'No ✗'}</p>`;                                                }                            content += '</ul>';                            }                                content += `<li>${name}: ${weight.toFixed(2)}</li>`;                            for (const [name, weight] of Object.entries(data.model_weights)) {                            content += '<p><strong>Model Weights:</strong></p><ul>';                        if (Object.keys(data.model_weights).length > 0) {                        // Weights                                                }                            }                                content += '</ul>';                                }                                    content += `<li>${file}</li>`;                                for (const file of data.model_files_found) {                                content += '<p><strong>Model Files Found:</strong></p><ul>';                            if (data.model_files_found && data.model_files_found.length > 0) {                            // Show files found to help with debugging                                                        content += '<span class="model-badge" style="background-color: #dc3545;">None loaded</span></p>';                        } else {                            content += '</ul>';                            }                                           Type: ${info.type}, Dim: ${info.dim}, Path: ${info.path}</li>`;                                content += `<li><span class="model-badge model-${name}">${name}</span>                             for (const [name, info] of Object.entries(data.ensemble_models)) {                            content += '<ul>';                            content += `<span class="model-badge">${Object.keys(data.ensemble_models).length} loaded</span></p>`;                        if (Object.keys(data.ensemble_models).length > 0) {                        content += '<p><strong>Ensemble Models:</strong> ';                        // Ensemble models                                                content += '</p>';                        }                            content += ` - Path: ${data.facenet.path}`;                        if (data.facenet.path) {                                    (${data.facenet.type || 'unknown'})`;                        content += `<p><strong>FaceNet:</strong> ${data.facenet.loaded ? 'Loaded ✓' : 'Not loaded ✗'}                         // FaceNet status                                                let content = '<h3>Models Status <button class="refresh-btn" onclick="checkModelsStatus()">Refresh</button></h3>';                        let modelsInfo = document.getElementById('models-info');                    .then(data => {                    .then(response => response.json())                fetch('/v1/face/models-status')            function checkModelsStatus() {            // Add model status checking function            }                }                    `;                        <p style="color: red;">Error: ${error.message}</p>                    document.getElementById('comparison-result').innerHTML = `                    log(`ERROR: ${error.message}`);                } catch (error) {                                        queryReader.readAsDataURL(queryFileInput.files[0]);                                        };                        }                            `;                                <p><strong>Reference Faces:</strong> ${referenceFaces.length}</p>                                <p><strong>Method Used:</strong> ${methodDisplay}</p>                                <p><strong>Processing Time:</strong> ${data.processing_time_ms.toFixed(2)} ms</p>                                <p><strong>Threshold Used:</strong> ${data.threshold_used}</p>                                <p><strong>Same Person:</strong> ${data.is_same_person ? 'Yes ✓' : 'No ✗'}</p>                                <p><strong>Similarity Score:</strong> <span class="${similarityClass}">${data.similarity.toFixed(4)}</span></p>                            document.getElementById('comparison-result').innerHTML = `                            // Display results                                                        else if (method === 'top_n') methodDisplay = "Top-3 Average Similarity";                            else if (method === 'average') methodDisplay = "Average Similarity";                            if (method === 'max') methodDisplay = "Maximum Similarity";                            let methodDisplay = "Unknown";                            // Display method used in a user-friendly way                                                        let similarityClass = getSimilarityClass(data.similarity);                            // Determine similarity class for styling
@router.get("/models-status")
async def models_status():
    """
    Show status of all face recognition models.
    """
    # Check FaceNet
    facenet_status = {"loaded": False, "path": None}
    if hasattr(face_embedder, 'model_type'):
        facenet_status["loaded"] = True
        facenet_status["type"] = face_embedder.model_type
        facenet_status["path"] = face_embedder.model_path
    
    # Check ensemble models
    ensemble_models = {}
    if hasattr(face_embedder, 'ensemble') and hasattr(face_embedder.ensemble, 'models'):
        for model_name, model_info in face_embedder.ensemble.models.items():
            ensemble_models[model_name] = {
                "type": model_info.get("type", "unknown"),
                "dim": model_info.get("dim", "unknown"),
                "path": model_info.get("path", "unknown")
            }
    
    # Check weights
    weights = {}
    if hasattr(face_embedder, 'ensemble') and hasattr(face_embedder.ensemble, 'model_weights'):
        weights = face_embedder.ensemble.model_weights
    
    # Search for model files
    model_files = []
    search_paths = [
        '/app/models', 
        '/app/app/models', 
        '/home/suwit/FaceSocial/ai-services/face-recognition/app/models'
    ]
    
    for base_path in search_paths:
        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith(('.onnx', '.pb', '.h5')):
                        model_files.append(os.path.join(root, file))
    
    # Check ONNX providers
    onnx_providers = []
    try:
        import onnxruntime as ort
        onnx_providers = ort.get_available_providers()
    except Exception as e:
        onnx_providers = [f"Error getting providers: {str(e)}"]
    
    return {
        "facenet": facenet_status,
        "ensemble_models": ensemble_models,
        "model_weights": weights,
        "use_ensemble": face_embedder.use_ensemble if hasattr(face_embedder, 'use_ensemble') else False,
        "model_files_found": model_files,
        "onnx_providers": onnx_providers
    }