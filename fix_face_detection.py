#!/usr/bin/env python3
import os

# เนื้อหาของไฟล์ app.py ใหม่
new_app_content = '''
from flask import Flask, request, jsonify
import json  # เพิ่มการ import json module มาตรฐาน
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import time

app = Flask(__name__)
CORS(app)

# Custom JSON Encoder สำหรับแปลงค่า NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

# ฟังก์ชันแปลงค่า NumPy types เป็น Python types มาตรฐาน
def convert_numpy_types(obj):
    """แปลงค่า NumPy types เป็น Python types มาตรฐาน เพื่อให้สามารถแปลงเป็น JSON ได้"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# โหลดโมเดล Haar Cascade สำหรับตรวจจับใบหน้า (มาพร้อมกับ OpenCV)
def load_face_detector():
    """โหลดโมเดล Face Detector ใช้ Haar Cascade"""
    try:
        print("โหลด Haar Cascade สำหรับตรวจจับใบหน้า...")
        model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return {"type": "haar", "model": model}
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการโหลดโมเดล Haar Cascade: {str(e)}")
        return None

# ตรวจจับใบหน้าด้วย Haar Cascade
def detect_faces_haar(img, model, confidence_threshold=0.5):
    """ตรวจจับใบหน้าด้วย Haar Cascade"""
    # แปลงเป็นภาพขาวดำ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ตรวจจับใบหน้า
    faces_rect = model.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    # แปลงผลลัพธ์
    faces = []
    for (x, y, w, h) in faces_rect:
        face_info = {
            "bbox": [int(x), int(y), int(w), int(h)],
            "confidence": 0.9,  # ค่าความเชื่อมั่นเริ่มต้น
            "landmarks": []
        }
        faces.append(face_info)
    
    return faces

# ฟังก์ชันวิเคราะห์เพศและอายุ (สร้างข้อมูลจำลอง)
def analyze_face_attributes(img, face):
    """วิเคราะห์เพศและอายุ (สร้างข้อมูลจำลอง)"""
    import random
    
    # สร้างข้อมูลจำลอง
    gender = "male" if random.random() > 0.5 else "female"
    age = int(random.uniform(18, 60))
    
    return {
        "gender": gender,
        "age": age
    }

# แปลงรูปภาพ base64 เป็น cv2 image
def decode_base64_image(base64_str):
    """แปลงรูปภาพ base64 เป็น cv2 image"""
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# โหลดโมเดลเมื่อเริ่มต้น
face_detector = load_face_detector()

@app.route('/health', methods=['GET'])
def health_check():
    """ตรวจสอบสถานะของ service"""
    status = {
        "status": "online" if face_detector else "limited",
        "version": "1.0.0",
        "models": []
    }
    
    if face_detector:
        status["models"].append(face_detector["type"])
    
    return jsonify(status)

@app.route('/detect', methods=['POST'])
def detect_faces():
    """API สำหรับตรวจจับใบหน้า"""
    data = request.json
    
    # ตรวจสอบว่ามีไฟล์รูปภาพหรือไม่
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    # ตรวจสอบว่าโหลดโมเดลสำเร็จหรือไม่
    if face_detector is None:
        return jsonify({'error': 'Face detection model not loaded'}), 500
    
    # แปลงรูปภาพจาก base64
    try:
        img = decode_base64_image(data['image'])
    except Exception as e:
        return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400
    
    # ตรวจจับใบหน้า
    try:
        start_time = time.time()
        
        # ใช้ Haar Cascade
        faces = detect_faces_haar(img, face_detector["model"])
        
        # เพิ่มข้อมูลเพศและอายุถ้าต้องการ
        include_attributes = data.get('include_attributes', False)
        if include_attributes:
            for face in faces:
                # ตัดเฉพาะส่วนใบหน้า
                x, y, w, h = face["bbox"]
                face_img = img[y:y+h, x:x+w]
                
                # วิเคราะห์เพศและอายุ
                if face_img.size > 0:  # ตรวจสอบว่ารูปไม่ว่างเปล่า
                    attributes = analyze_face_attributes(face_img, face)
                    face.update(attributes)
        
        processing_time = time.time() - start_time
        
        # แปลงข้อมูลเป็น Python types มาตรฐาน
        faces = convert_numpy_types(faces)
        
        # ส่งผลลัพธ์
        result = {
            "faces": faces,
            "count": len(faces),
            "processing_time": processing_time
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Face detection failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
'''

# เขียนไฟล์
with open('services/face-detection/app.py', 'w') as f:
    f.write(new_app_content.strip())

print("✅ แก้ไขไฟล์ services/face-detection/app.py สำเร็จ")
print("รีบิลด์และรีสตาร์ท service ด้วย: docker-compose up -d --build face-detection")