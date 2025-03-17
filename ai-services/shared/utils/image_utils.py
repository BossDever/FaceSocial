import cv2
import numpy as np
import base64
from typing import Tuple, List, Optional, Union
from PIL import Image
import io

# ฟังก์ชันแปลง base64 เป็น numpy array
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

# ฟังก์ชันแปลง numpy array เป็น base64
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

# ฟังก์ชันปรับขนาดภาพ
def resize_image(image: np.ndarray, size: Tuple[int, int], keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    ปรับขนาดรูปภาพ
    
    Args:
        image: รูปภาพในรูปแบบ numpy array (BGR)
        size: (width, height) ที่ต้องการ
        keep_aspect_ratio: รักษาสัดส่วนภาพหรือไม่
        
    Returns:
        np.ndarray: รูปภาพที่ปรับขนาดแล้ว
    """
    h, w = image.shape[:2]
    target_w, target_h = size
    
    if keep_aspect_ratio:
        # คำนวณสัดส่วนที่เหมาะสม
        aspect = w / h
        
        if w > h:
            target_h = int(target_w / aspect)
        else:
            target_w = int(target_h * aspect)
    
    # ปรับขนาดรูปภาพ
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    return resized

# ฟังก์ชันตัดภาพใบหน้า
def crop_face(image: np.ndarray, bbox: List[int], margin: float = 0.3) -> np.ndarray:
    """
    ตัดภาพใบหน้าจากรูปภาพ
    
    Args:
        image: รูปภาพในรูปแบบ numpy array (BGR)
        bbox: [x1, y1, x2, y2] พิกัดของใบหน้า
        margin: ขอบเพิ่มเติม (เป็นสัดส่วนของขนาดกล่อง)
        
    Returns:
        np.ndarray: ภาพใบหน้าที่ตัดแล้ว
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # คำนวณขนาดของกล่อง
    box_w = x2 - x1
    box_h = y2 - y1
    
    # เพิ่มขอบ
    margin_x = int(box_w * margin)
    margin_y = int(box_h * margin)
    
    # คำนวณพิกัดใหม่พร้อมขอบ
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)
    
    # ตัดภาพ
    face_crop = image[y1:y2, x1:x2]
    
    return face_crop

# ฟังก์ชันปรับสีและความสว่าง
def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    ปรับสีและความสว่างของรูปภาพ
    
    Args:
        image: รูปภาพในรูปแบบ numpy array (BGR)
        
    Returns:
        np.ndarray: รูปภาพที่ปรับแล้ว
    """
    # แปลงเป็น LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # แยกช่องสี
    l, a, b = cv2.split(lab)
    
    # ทำ CLAHE (Contrast Limited Adaptive Histogram Equalization) กับช่อง L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # รวมช่องสีกลับ
    lab = cv2.merge((l, a, b))
    
    # แปลงกลับเป็น BGR
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return normalized

# ฟังก์ชัน preprocess สำหรับโมเดล face recognition
def preprocess_for_face_recognition(image: np.ndarray, model_type: str) -> np.ndarray:
    """
    Preprocess รูปภาพสำหรับโมเดล face recognition
    
    Args:
        image: รูปภาพในรูปแบบ numpy array (BGR)
        model_type: ชื่อโมเดล ('facenet', 'arcface', 'elasticface', 'adaface')
        
    Returns:
        np.ndarray: รูปภาพที่ผ่านการ preprocess แล้ว
    """
    # ขนาดภาพสำหรับแต่ละโมเดล
    model_size = {
        'facenet': (160, 160),
        'arcface': (112, 112),
        'elasticface': (112, 112),
        'adaface': (112, 112)
    }
    
    if model_type not in model_size:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # ปรับขนาดรูปภาพตามที่โมเดลต้องการ
    resized = resize_image(image, model_size[model_type], keep_aspect_ratio=False)
    
    # แปลงเป็น RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    if model_type == 'facenet':
        # FaceNet: scale to [0, 1] and then normalize
        preprocessed = rgb / 255.0
        preprocessed = (preprocessed - 0.5) / 0.5
    elif model_type in ['arcface', 'elasticface', 'adaface']:
        # ArcFace, ElasticFace, AdaFace: normalize to [-1, 1]
        preprocessed = rgb.astype(np.float32) / 127.5 - 1.0
    
    # แปลงเป็นรูปแบบที่เหมาะสมสำหรับโมเดล (NCHW)
    preprocessed = preprocessed.transpose(2, 0, 1)  # HWC to CHW format
    preprocessed = np.expand_dims(preprocessed, axis=0)  # add batch dimension
    
    return preprocessed.astype(np.float32)