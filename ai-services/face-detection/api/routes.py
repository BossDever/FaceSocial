"""API routes for the face detection service."""
from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import JSONResponse
import time
import numpy as np
import cv2
import base64
import logging
from typing import List, Dict, Any, Optional
import os
import sys

# จัดการ import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
api_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(api_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from shared.config.settings import settings
from shared.utils.logging import structured_log
from shared.middleware.auth import get_api_key

# Import local modules correctly
from face_detection.models.scrfd import SCRFDDetector
from face_detection.core.aligner import FaceAligner
from face_detection.core.extractor import FaceExtractor

# Import image utilities (create this file if missing)
from face_detection.utils.image_utils import base64_to_image, image_to_base64

from face_detection.api.models import (
    FaceDetectionRequest, FaceDetectionResponse, 
    BatchDetectionRequest, BatchDetectionResponse, BatchDetectionResult,
    FaceAlignRequest, FaceAlignResponse,
    FaceExtractRequest, FaceExtractResponse,
    FaceBoundingBox
)

# สร้าง router
router = APIRouter()

# สร้าง logger
logger = logging.getLogger(__name__)

# อ้างอิงถึงโมเดลที่โหลดในไฟล์ main.py - จะนำมาเชื่อมต่อในขั้นตอนต่อไป
face_detector = None

# สร้าง face aligner และ extractor
face_aligner = FaceAligner()
face_extractor = FaceExtractor()

@router.post("/detect", response_model=FaceDetectionResponse)
async def detect_face(
    request: FaceDetectionRequest,
    api_key: str = Depends(get_api_key)
):
    """
    ตรวจจับใบหน้าในรูปภาพ
    
    หลังจากส่งรูปภาพเข้ามา API จะตรวจจับใบหน้าทั้งหมดในรูปภาพและส่งกลับตำแหน่งพร้อมความมั่นใจ
    """
    start_time = time.time()
    
    # Import face_detector จาก main module ถ้าไม่ได้ตั้งค่าไว้
    global face_detector
    if face_detector is None:
        from main import face_detector
    
    try:
        # ตรวจสอบว่าโหลดโมเดลสำเร็จแล้วหรือยัง
        if face_detector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Face detector model is not loaded. Please try again later."
            )
        
        # แปลงรูปภาพจาก base64 เป็น numpy array
        image = base64_to_image(request.image)
        
        # ตรวจจับใบหน้า
        use_gpu = request.use_gpu if request.use_gpu is not None else settings.USE_GPU
        faces = face_detector.detect_faces(
            image,
            min_face_size=request.min_face_size,
            confidence_threshold=request.threshold,
            return_landmarks=request.return_landmarks
        )
        
        # แปลงผลลัพธ์ให้เข้ากับ response model
        face_boxes = []
        for face in faces:
            face_box = FaceBoundingBox(
                bbox=face["bbox"],
                confidence=face["confidence"]
            )
            
            if "landmarks" in face and face["landmarks"] is not None:
                face_box.landmarks = face["landmarks"]
            
            face_boxes.append(face_box)
        
        # สร้าง response
        response = FaceDetectionResponse(
            faces=face_boxes,
            count=len(face_boxes),
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # บันทึก log
        structured_log(
            level="info",
            message=f"Detected {len(face_boxes)} faces",
            service="face-detection",
            method="POST",
            path="/v1/face/detect",
            duration_ms=response.processing_time_ms,
            extra={"num_faces": len(face_boxes)}
        )
        
        return response
    
    except Exception as e:
        # บันทึก log ข้อผิดพลาด
        processing_time = (time.time() - start_time) * 1000
        structured_log(
            level="error",
            message=f"Error detecting faces: {str(e)}",
            service="face-detection",
            method="POST",
            path="/v1/face/detect",
            duration_ms=processing_time,
            error={"message": str(e)}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting faces: {str(e)}"
        )

# ส่วนที่เหลือของไฟล์ routes.py คงเดิม