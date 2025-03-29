from flask import Flask, request, jsonify, json
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import time

app = Flask(__name__)
CORS(app)

# ตั้งค่าพาธของโมเดล face detection
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "retinaface_r50_v1.onnx")

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

app.json_encoder = NumpyEncoder  # เปลี่ยนจาก app.json.encoder เป็น app.json_encoder

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

# โหลดโมเดล Face Detector
def load_face_detector():
    """โหลดโมเดล Face Detector (RetinaFace หรือ SCRFD)"""
    model_files = [
        os.path.join(MODEL_DIR, "retinaface_r50_v1.onnx"),
        os.path.join(MODEL_DIR, "scrfd_10g_bnkps.onnx")
    ]
    
    # ตรวจสอบไฟล์โมเดลที่มีอยู่
    model_path = None
    for path in model_files:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("⚠️ ไม่พบไฟล์โมเดล Face Detection")
        return None
    
    # โหลดโมเดลขึ้นอยู่กับชนิดของโมเดล
    try:
        if "retinaface" in model_path:
            print(f"โหลดโมเดล RetinaFace จาก {model_path}")
            model = cv2.dnn.readNetFromONNX(model_path)
            return {"type": "retinaface", "model": model}
        elif "scrfd" in model_path:
            print(f"โหลดโมเดล SCRFD จาก {model_path}")
            model = cv2.dnn.readNetFromONNX(model_path)
            return {"type": "scrfd", "model": model}
        else:
            print(f"⚠️ ไม่รองรับโมเดลชนิดนี้: {model_path}")
            return None
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
        return None

# ตรวจจับใบหน้าด้วย RetinaFace
def detect_faces_retinaface(img, model, confidence_threshold=0.5):
    """ตรวจจับใบหน้าด้วย RetinaFace"""
    # เตรียมรูปภาพ
    h, w, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1.0, (640, 640), [0, 0, 0], True, False)
    
    # ส่งรูปภาพเข้าโมเดล
    model.setInput(blob)
    
    # รับผลลัพธ์จากโมเดล
    boxes, scores, landmarks = model.forward(["face_rpn_bbox_pred_stride8", "face_rpn_cls_prob_reshape_stride8", "face_rpn_landmark_pred_stride8"])
    
    # แปลงผลลัพธ์
    faces = []
    for i, score in enumerate(scores):
        if score > confidence_threshold:
            # แปลง box coordinates
            box = boxes[i]
            x1, y1, x2, y2 = (
                int(box[0] * w),
                int(box[1] * h),
                int(box[2] * w),
                int(box[3] * h)
            )
            
            # สร้างข้อมูลใบหน้า
            face_info = {
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "confidence": float(score),
                "landmarks": []
            }
            
            # เพิ่ม landmarks ถ้ามี
            if landmarks is not None and i < len(landmarks):
                for j in range(0, 10, 2):
                    x = int(landmarks[i][j] * w)
                    y = int(landmarks[i][j+1] * h)
                    face_info["landmarks"].append([x, y])
            
            faces.append(face_info)
    
    return faces

# ตรวจจับใบหน้าด้วย SCRFD
def detect_faces_scrfd(img, model, confidence_threshold=0.5):
    """ตรวจจับใบหน้าด้วย SCRFD"""
    # เตรียมรูปภาพ
    h, w, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1.0, (640, 640), [127.5, 127.5, 127.5], True, False)
    
    # ส่งรูปภาพเข้าโมเดล
    model.setInput(blob)
    
    # รับผลลัพธ์จากโมเดล
    outputs = model.forward(model.getUnconnectedOutLayersNames())
    
    # แปลงผลลัพธ์
    faces = []
    for i in range(0, len(outputs), 2):
        boxes = outputs[i][0]
        scores = outputs[i+1][0]
        
        for j, score in enumerate(scores):
            if score > confidence_threshold:
                # แปลง box coordinates
                box = boxes[j]
                x1, y1, x2, y2 = (
                    int(box[0] * w),
                    int(box[1] * h),
                    int(box[2] * w),
                    int(box[3] * h)
                )
                
                # สร้างข้อมูลใบหน้า
                face_info = {
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "confidence": float(score),
                    "landmarks": []
                }
                faces.append(face_info)
    
    return faces

# ฟังก์ชันวิเคราะห์เพศและอายุ (สร้างข้อมูลจำลอง)
def analyze_face_attributes(img, face):
    """วิเคราะห์เพศและอายุ (ตัวอย่าง - สร้างข้อมูลจำลอง)"""
    # ในโค้ดจริงควรมีการใช้โมเดลวิเคราะห์เพศและอายุ
    # แต่ในที่นี้สร้างข้อมูลจำลอง
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
        
        if face_detector["type"] == "retinaface":
            faces = detect_faces_retinaface(img, face_detector["model"])
        elif face_detector["type"] == "scrfd":
            faces = detect_faces_scrfd(img, face_detector["model"])
        else:
            return jsonify({'error': 'Unsupported face detection model'}), 500
        
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