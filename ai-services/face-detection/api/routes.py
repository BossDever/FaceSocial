"""API routes for the face detection service."""
from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import JSONResponse
import time
import numpy as np
import cv2
import base64
import logging
from typing import List, Dict, Any, Optional

from shared.config.settings import settings
from shared.utils.logging import structured_log
from shared.middleware.auth import get_api_key

from models.scrfd import SCRFDDetector
from core.aligner import FaceAligner
from core.extractor import FaceExtractor
from utils.image_utils import base64_to_image, image_to_base64

from api.models import (
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

# อ้างอิงถึงโมเดลที่โหลดในไฟล์ main.py
from main import face_detector

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

@router.post("/batch-detect", response_model=BatchDetectionResponse)
async def batch_detect(
    request: BatchDetectionRequest,
    api_key: str = Depends(get_api_key)
):
    """
    ตรวจจับใบหน้าในรูปภาพหลายรูปพร้อมกัน
    
    รองรับการประมวลผลรูปภาพจำนวนมากพร้อมกันด้วย batch processing
    """
    start_time = time.time()
    
    try:
        # ตรวจสอบว่าโหลดโมเดลสำเร็จแล้วหรือยัง
        if face_detector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Face detector model is not loaded. Please try again later."
            )
        
        # แปลงรูปภาพทั้งหมดจาก base64 เป็น numpy array
        images = []
        for i, image_base64 in enumerate(request.images):
            try:
                image = base64_to_image(image_base64)
                images.append(image)
            except Exception as e:
                logger.error(f"Error decoding image {i}: {str(e)}")
                # ถ้ามีบางรูปภาพที่แปลงไม่สำเร็จ ให้ใส่ None แทน
                images.append(None)
        
        # กรองออกรูปภาพที่เป็น None
        valid_images = [img for img in images if img is not None]
        
        # ตรวจจับใบหน้าในรูปภาพทั้งหมด
        if not valid_images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid images to process"
            )
        
        # ตรวจจับใบหน้าพร้อมกันด้วย batch processing
        batch_results = face_detector.batch_detect(
            valid_images,
            min_face_size=request.min_face_size,
            confidence_threshold=request.threshold,
            return_landmarks=request.return_landmarks,
            batch_size=request.batch_size
        )
        
        # แปลงผลลัพธ์ให้เข้ากับ response model
        results = []
        valid_idx = 0
        
        for i, image in enumerate(images):
            if image is None:
                # ถ้ารูปภาพไม่ถูกต้อง ให้ส่งผลลัพธ์ว่าง
                results.append(BatchDetectionResult(
                    image_index=i,
                    faces=[],
                    count=0
                ))
            else:
                # แปลงผลลัพธ์ของรูปภาพที่ถูกต้อง
                face_boxes = []
                faces = batch_results[valid_idx]
                
                for face in faces:
                    face_box = FaceBoundingBox(
                        bbox=face["bbox"],
                        confidence=face["confidence"]
                    )
                    
                    if "landmarks" in face and face["landmarks"] is not None:
                        face_box.landmarks = face["landmarks"]
                    
                    face_boxes.append(face_box)
                
                results.append(BatchDetectionResult(
                    image_index=i,
                    faces=face_boxes,
                    count=len(face_boxes)
                ))
                
                valid_idx += 1
        
        # จำนวนรูปภาพที่ตรวจจับสำเร็จ (มีอย่างน้อย 1 ใบหน้า)
        successful_detections = sum(1 for result in results if result.count > 0)
        
        # สร้าง response
        response = BatchDetectionResponse(
            results=results,
            total_images=len(request.images),
            successful_detections=successful_detections,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # บันทึก log
        structured_log(
            level="info",
            message=f"Batch processed {len(request.images)} images",
            service="face-detection",
            method="POST",
            path="/v1/face/batch-detect",
            duration_ms=response.processing_time_ms,
            extra={
                "total_images": len(request.images),
                "successful_detections": successful_detections
            }
        )
        
        return response
    
    except Exception as e:
        # บันทึก log ข้อผิดพลาด
        processing_time = (time.time() - start_time) * 1000
        structured_log(
            level="error",
            message=f"Error batch detecting faces: {str(e)}",
            service="face-detection",
            method="POST",
            path="/v1/face/batch-detect",
            duration_ms=processing_time,
            error={"message": str(e)}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error batch detecting faces: {str(e)}"
        )

@router.post("/align", response_model=FaceAlignResponse)
async def align_face(
    request: FaceAlignRequest,
    api_key: str = Depends(get_api_key)
):
    """
    ปรับตำแหน่งใบหน้าให้อยู่ในตำแหน่งมาตรฐาน
    
    ใช้ landmarks 5 จุดเพื่อปรับตำแหน่งใบหน้าให้อยู่ในตำแหน่งมาตรฐานสำหรับการจดจำใบหน้า
    """
    start_time = time.time()
    
    try:
        # แปลงรูปภาพจาก base64 เป็น numpy array
        image = base64_to_image(request.image)
        
        # ถ้ามีการระบุ target_size ให้สร้าง aligner ใหม่
        if request.target_size and request.target_size != [112, 112]:
            aligner = FaceAligner(target_size=tuple(request.target_size))
        else:
            aligner = face_aligner
        
        # ปรับตำแหน่งใบหน้า
        aligned_face = aligner.align_face(image, request.landmarks)
        
        # แปลงรูปภาพกลับเป็น base64
        aligned_face_base64 = image_to_base64(aligned_face)
        
        # สร้าง response
        response = FaceAlignResponse(
            aligned_image=aligned_face_base64,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # บันทึก log
        structured_log(
            level="info",
            message="Face aligned successfully",
            service="face-detection",
            method="POST",
            path="/v1/face/align",
            duration_ms=response.processing_time_ms
        )
        
        return response
    
    except Exception as e:
        # บันทึก log ข้อผิดพลาด
        processing_time = (time.time() - start_time) * 1000
        structured_log(
            level="error",
            message=f"Error aligning face: {str(e)}",
            service="face-detection",
            method="POST",
            path="/v1/face/align",
            duration_ms=processing_time,
            error={"message": str(e)}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error aligning face: {str(e)}"
        )

@router.post("/extract", response_model=FaceExtractResponse)
async def extract_face(
    request: FaceExtractRequest,
    api_key: str = Depends(get_api_key)
):
    """
    ตัดใบหน้าจากรูปภาพ
    
    ตัดเฉพาะใบหน้าจากรูปภาพด้วยกรอบที่กำหนด
    """
    start_time = time.time()
    
    try:
        # แปลงรูปภาพจาก base64 เป็น numpy array
        image = base64_to_image(request.image)
        
        # ตัดใบหน้า
        extracted = face_extractor.extract_face(
            image,
            request.bbox,
            request.margin
        )
        
        # แปลงรูปภาพกลับเป็น base64
        face_image_base64 = image_to_base64(extracted["face_image"])
        
        # สร้าง response
        response = FaceExtractResponse(
            face_image=face_image_base64,
            original_bbox=extracted["original_bbox"],
            final_bbox=extracted["final_bbox"],
            image_size=extracted["image_size"],
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # บันทึก log
        structured_log(
            level="info",
            message="Face extracted successfully",
            service="face-detection",
            method="POST",
            path="/v1/face/extract",
            duration_ms=response.processing_time_ms
        )
        
        return response
    
    except Exception as e:
        # บันทึก log ข้อผิดพลาด
        processing_time = (time.time() - start_time) * 1000
        structured_log(
            level="error",
            message=f"Error extracting face: {str(e)}",
            service="face-detection",
            method="POST",
            path="/v1/face/extract",
            duration_ms=processing_time,
            error={"message": str(e)}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting face: {str(e)}"
        )