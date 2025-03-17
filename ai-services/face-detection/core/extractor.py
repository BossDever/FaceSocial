"""Feature extractor implementation."""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import time
import logging

logger = logging.getLogger(__name__)

class FaceExtractor:
    """
    คลาสสำหรับตัดใบหน้าจากรูปภาพ
    """
    
    def __init__(self):
        """
        Initialize face extractor
        """
        logger.info("Face extractor initialized")
    
    def extract_face(
        self,
        image: np.ndarray,
        bbox: List[int],
        margin: float = 0.3
    ) -> Dict[str, Any]:
        """
        ตัดใบหน้าจากรูปภาพด้วยกรอบที่กำหนด
        
        Args:
            image: รูปภาพต้นฉบับในรูปแบบ numpy array (BGR)
            bbox: พิกัดกรอบใบหน้า [x1, y1, x2, y2]
            margin: ขอบเพิ่มเติม (เป็นสัดส่วนของขนาดกล่อง)
            
        Returns:
            Dictionary containing:
                - face_image: รูปภาพใบหน้าที่ตัดออกมา
                - original_bbox: พิกัดกรอบใบหน้าดั้งเดิม [x1, y1, x2, y2]
                - final_bbox: พิกัดกรอบใบหน้าสุดท้ายที่รวมขอบเพิ่มเติม [x1, y1, x2, y2]
                - image_size: ขนาดของรูปภาพที่ตัดออกมา [width, height]
        """
        start_time = time.time()
        
        # รับขนาดของรูปภาพ
        height, width = image.shape[:2]
        
        # แปลง bbox เป็น integer
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # คำนวณขนาดของกล่อง
        box_width = x2 - x1
        box_height = y2 - y1
        
        # คำนวณขอบเพิ่มเติม
        margin_x = int(box_width * margin)
        margin_y = int(box_height * margin)
        
        # คำนวณพิกัดใหม่พร้อมขอบเพิ่มเติม
        new_x1 = max(0, x1 - margin_x)
        new_y1 = max(0, y1 - margin_y)
        new_x2 = min(width, x2 + margin_x)
        new_y2 = min(height, y2 + margin_y)
        
        # ตัดรูปภาพ
        face_image = image[new_y1:new_y2, new_x1:new_x2]
        
        logger.debug(f"Face extracted in {(time.time() - start_time) * 1000:.2f} ms")
        
        return {
            "face_image": face_image,
            "original_bbox": [x1, y1, x2, y2],
            "final_bbox": [new_x1, new_y1, new_x2, new_y2],
            "image_size": [new_x2 - new_x1, new_y2 - new_y1]
        }
    
    def batch_extract(
        self,
        images: List[np.ndarray],
        bboxes: List[List[int]],
        margin: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        ตัดใบหน้าจากหลายรูปภาพพร้อมกัน
        
        Args:
            images: รายการรูปภาพต้นฉบับ
            bboxes: รายการพิกัดกรอบใบหน้า
            margin: ขอบเพิ่มเติม (เป็นสัดส่วนของขนาดกล่อง)
            
        Returns:
            รายการ dictionary ของใบหน้าที่ตัดออกมา
        """
        start_time = time.time()
        extracted_faces = []
        
        for i, (image, bbox) in enumerate(zip(images, bboxes)):
            try:
                extracted_face = self.extract_face(image, bbox, margin)
                extracted_faces.append(extracted_face)
            except Exception as e:
                logger.error(f"Error extracting face {i}: {str(e)}")
                extracted_faces.append(None)
        
        logger.debug(f"Batch extracted {len(images)} faces in {(time.time() - start_time) * 1000:.2f} ms")
        return extracted_faces