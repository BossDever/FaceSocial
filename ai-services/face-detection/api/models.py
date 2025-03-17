"""API models for the face detection service."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import base64

class FaceDetectionRequest(BaseModel):
    """ข้อมูลสำหรับการตรวจจับใบหน้าในรูปภาพเดียว"""
    image: str = Field(..., description="รูปภาพในรูปแบบ base64")
    min_face_size: int = Field(50, description="ขนาดใบหน้าขั้นต่ำที่ต้องการตรวจจับ (พิกเซล)")
    threshold: float = Field(0.85, description="ค่า threshold สำหรับความมั่นใจในการตรวจจับ (0-1)")
    return_landmarks: bool = Field(True, description="ต้องการ landmarks หรือไม่")
    use_gpu: Optional[bool] = Field(None, description="ใช้ GPU หรือไม่ (ถ้าไม่ระบุจะใช้ค่าจาก settings)")
    
    @validator('image')
    def validate_image(cls, v):
        if not v:
            raise ValueError("Image cannot be empty")
        
        # ตรวจสอบว่าเป็น base64 ที่ถูกต้องหรือไม่
        try:
            # ถ้ามี header (เช่น data:image/jpeg;base64,) ให้ตัดออก
            if ',' in v:
                v = v.split(',', 1)[1]
            
            # ทดลอง decode
            base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 image")
        
        return v
    
    @validator('threshold')
    def validate_threshold(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v

class BatchDetectionRequest(BaseModel):
    """ข้อมูลสำหรับการตรวจจับใบหน้าในหลายรูปภาพ"""
    images: List[str] = Field(..., description="รายการรูปภาพในรูปแบบ base64")
    min_face_size: int = Field(50, description="ขนาดใบหน้าขั้นต่ำที่ต้องการตรวจจับ (พิกเซล)")
    threshold: float = Field(0.85, description="ค่า threshold สำหรับความมั่นใจในการตรวจจับ (0-1)")
    return_landmarks: bool = Field(True, description="ต้องการ landmarks หรือไม่")
    batch_size: int = Field(16, description="ขนาด batch สำหรับการประมวลผล")
    
    @validator('images')
    def validate_images(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Images list cannot be empty")
        
        if len(v) > 100:
            raise ValueError("Maximum of 100 images allowed per request")
            
        return v
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v < 1:
            raise ValueError("Batch size must be at least 1")
        if v > 64:
            raise ValueError("Batch size cannot exceed 64")
        return v

class FaceAlignRequest(BaseModel):
    """ข้อมูลสำหรับการปรับตำแหน่งใบหน้า"""
    image: str = Field(..., description="รูปภาพในรูปแบบ base64")
    landmarks: List[List[float]] = Field(..., description="จุด landmarks 5 จุด (ตา 2 จุด, จมูก 1 จุด, มุมปาก 2 จุด)")
    target_size: Optional[List[int]] = Field([112, 112], description="ขนาดภาพเป้าหมาย [width, height]")

class FaceExtractRequest(BaseModel):
    """ข้อมูลสำหรับการตัดใบหน้าจากรูปภาพ"""
    image: str = Field(..., description="รูปภาพในรูปแบบ base64")
    bbox: List[int] = Field(..., description="พิกัดกรอบใบหน้า [x1, y1, x2, y2]")
    margin: float = Field(0.3, description="ขอบเพิ่มเติม (เป็นสัดส่วนของขนาดกล่อง)")

class FaceBoundingBox(BaseModel):
    """ข้อมูลกรอบใบหน้าที่ตรวจพบ"""
    bbox: List[int] = Field(..., description="พิกัดกรอบใบหน้า [x1, y1, x2, y2]")
    confidence: float = Field(..., description="ความมั่นใจในการตรวจจับ (0-1)")
    landmarks: Optional[List[List[float]]] = Field(None, description="จุด landmarks 5 จุด (ตา 2 จุด, จมูก 1 จุด, มุมปาก 2 จุด)")

class FaceDetectionResponse(BaseModel):
    """ผลลัพธ์การตรวจจับใบหน้า"""
    faces: List[FaceBoundingBox] = Field(..., description="รายการใบหน้าที่ตรวจพบ")
    count: int = Field(..., description="จำนวนใบหน้าที่ตรวจพบ")
    processing_time_ms: float = Field(..., description="เวลาในการประมวลผล (มิลลิวินาที)")

class BatchDetectionResult(BaseModel):
    """ผลลัพธ์การตรวจจับใบหน้าสำหรับรูปภาพหนึ่งใบ"""
    image_index: int = Field(..., description="ลำดับของรูปภาพในรายการ")
    faces: List[FaceBoundingBox] = Field(..., description="รายการใบหน้าที่ตรวจพบ")
    count: int = Field(..., description="จำนวนใบหน้าที่ตรวจพบ")

class BatchDetectionResponse(BaseModel):
    """ผลลัพธ์การตรวจจับใบหน้าในหลายรูปภาพ"""
    results: List[BatchDetectionResult] = Field(..., description="ผลลัพธ์สำหรับแต่ละรูปภาพ")
    total_images: int = Field(..., description="จำนวนรูปภาพทั้งหมด")
    successful_detections: int = Field(..., description="จำนวนรูปภาพที่ตรวจจับสำเร็จ")
    processing_time_ms: float = Field(..., description="เวลาในการประมวลผลทั้งหมด (มิลลิวินาที)")

class FaceAlignResponse(BaseModel):
    """ผลลัพธ์การปรับตำแหน่งใบหน้า"""
    aligned_image: str = Field(..., description="รูปภาพใบหน้าที่ปรับตำแหน่งแล้วในรูปแบบ base64")
    processing_time_ms: float = Field(..., description="เวลาในการประมวลผล (มิลลิวินาที)")

class FaceExtractResponse(BaseModel):
    """ผลลัพธ์การตัดใบหน้าจากรูปภาพ"""
    face_image: str = Field(..., description="รูปภาพใบหน้าที่ตัดออกมาในรูปแบบ base64")
    original_bbox: List[int] = Field(..., description="พิกัดกรอบใบหน้าดั้งเดิม [x1, y1, x2, y2]")
    final_bbox: List[int] = Field(..., description="พิกัดกรอบใบหน้าสุดท้ายที่รวมขอบเพิ่มเติม [x1, y1, x2, y2]")
    image_size: List[int] = Field(..., description="ขนาดของรูปภาพที่ตัดออกมา [width, height]")
    processing_time_ms: float = Field(..., description="เวลาในการประมวลผล (มิลลิวินาที)")