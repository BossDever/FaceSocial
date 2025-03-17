"""Face alignment implementation."""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import time
import logging

logger = logging.getLogger(__name__)

class FaceAligner:
    """
    คลาสสำหรับปรับตำแหน่งใบหน้าให้อยู่ในตำแหน่งมาตรฐาน
    """
    
    def __init__(self, target_size: Tuple[int, int] = (112, 112)):
        """
        Initialize face aligner
        
        Args:
            target_size: ขนาดภาพเป้าหมาย (width, height)
        """
        self.target_size = target_size
        
        # จุด landmark มาตรฐาน
        # จุดมาตรฐานสำหรับโมเดล face recognition ขนาด 112x112
        self.reference_landmarks = np.array([
            [38.2946, 51.6963],  # ตาซ้าย
            [73.5318, 51.5014],  # ตาขวา
            [56.0252, 71.7366],  # จมูก
            [41.5493, 92.3655],  # มุมปากซ้าย
            [70.7299, 92.2041]   # มุมปากขวา
        ], dtype=np.float32)
        
        # ปรับตำแหน่งจุดมาตรฐานตามขนาดภาพเป้าหมาย
        scale_x = target_size[0] / 112.0
        scale_y = target_size[1] / 112.0
        
        self.reference_landmarks[:, 0] *= scale_x
        self.reference_landmarks[:, 1] *= scale_y
        
        logger.info(f"Face aligner initialized with target size: {target_size}")
    
    def align_face(self, image: np.ndarray, landmarks: List[List[float]]) -> np.ndarray:
        """
        ปรับตำแหน่งใบหน้าให้อยู่ในตำแหน่งมาตรฐาน
        
        Args:
            image: รูปภาพต้นฉบับในรูปแบบ numpy array (BGR)
            landmarks: จุด landmarks 5 จุด (ตา 2 จุด, จมูก 1 จุด, มุมปาก 2 จุด)
            
        Returns:
            รูปภาพใบหน้าที่ปรับตำแหน่งแล้ว
        """
        start_time = time.time()
        
        # แปลง landmarks เป็น numpy array
        src = np.array(landmarks, dtype=np.float32)
        
        # คำนวณ transformation matrix
        M = cv2.estimateAffinePartial2D(src, self.reference_landmarks)[0]
        
        # แปลงภาพ
        aligned_face = cv2.warpAffine(
            image,
            M,
            (self.target_size[0], self.target_size[1]),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        logger.debug(f"Face aligned in {(time.time() - start_time) * 1000:.2f} ms")
        return aligned_face
    
    def batch_align(self, images: List[np.ndarray], landmarks_list: List[List[List[float]]]) -> List[np.ndarray]:
        """
        ปรับตำแหน่งใบหน้าหลายใบพร้อมกัน
        
        Args:
            images: รายการรูปภาพต้นฉบับ
            landmarks_list: รายการของ landmarks สำหรับแต่ละรูปภาพ
            
        Returns:
            รายการของรูปภาพใบหน้าที่ปรับตำแหน่งแล้ว
        """
        start_time = time.time()
        aligned_faces = []
        
        for i, (image, landmarks) in enumerate(zip(images, landmarks_list)):
            try:
                aligned_face = self.align_face(image, landmarks)
                aligned_faces.append(aligned_face)
            except Exception as e:
                logger.error(f"Error aligning face {i}: {str(e)}")
                # ถ้าปรับตำแหน่งไม่สำเร็จ ให้ resize รูปภาพแทน
                aligned_face = cv2.resize(image, self.target_size)
                aligned_faces.append(aligned_face)
        
        logger.debug(f"Batch aligned {len(images)} faces in {(time.time() - start_time) * 1000:.2f} ms")
        return aligned_faces