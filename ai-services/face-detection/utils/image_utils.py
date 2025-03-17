"""Utility functions for image processing."""
import cv2
import numpy as np
import base64
from typing import Tuple, List, Optional, Union

def base64_to_image(base64_str: str) -> np.ndarray:
    """
    แปลงรูปภาพ base64 เป็น numpy array
    
    Args:
        base64_str: รูปภาพในรูปแบบ base64 string
        
    Returns:
        np.ndarray: รูปภาพในรูปแบบ numpy array (BGR)
    """
    try:
        # ถ้า base64 string มี header (data:image/jpeg;base64,) ให้ตัดออก
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
            
        # Decode base64 เป็น bytes
        img_bytes = base64.b64decode(base64_str)
        
        # แปลง bytes เป็น numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode เป็นรูปภาพ
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode base64 image")
            
        return img
    except Exception as e:
        raise ValueError(f"Error decoding base64 image: {str(e)}")

def image_to_base64(image: np.ndarray, format: str = 'jpeg') -> str:
    """
    แปลงรูปภาพ numpy array เป็น base64 string
    
    Args:
        image: รูปภาพในรูปแบบ numpy array (BGR)
        format: รูปแบบไฟล์ภาพ ('jpeg' หรือ 'png')
        
    Returns:
        str: รูปภาพในรูปแบบ base64 string
    """
    # กำหนด encode params สำหรับ jpeg
    if format.lower() == 'jpeg':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        ext = '.jpg'
    else:  # png
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        ext = '.png'
    
    # Encode รูปภาพเป็น bytes
    _, buffer = cv2.imencode(ext, image, encode_param)
    
    # แปลงเป็น base64
    base64_str = base64.b64encode(buffer).decode('utf-8')
    
    return base64_str