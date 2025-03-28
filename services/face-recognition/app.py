from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import onnxruntime as ort
import os
from scipy.spatial.distance import cosine
import json

app = Flask(__name__)
CORS(app)

# โหลดโมเดล
MODELS = {
    "arcface": {
        "path": os.path.join('models', 'arcface_r100.onnx'),
        "session": None,
        "default_weight": 0.33  # เพิ่มน้ำหนักจาก 0.25 เป็น 0.33
    },
    "adaface": {
        "path": os.path.join('models', 'adaface_ir101_webface12m.onnx'),
        "session": None,
        "default_weight": 0.33  # เพิ่มน้ำหนักจาก 0.25 เป็น 0.33
    },
    "elasticface": {
        "path": os.path.join('models', 'elasticface.onnx'),
        "session": None,
        "default_weight": 0.34  # เพิ่มน้ำหนักจาก 0.20 เป็น 0.34
    }
    # FaceNet ถูกลบออกเนื่องจากไม่มีโมเดล
}

# โหลดโมเดลที่มีอยู่
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

def load_available_models():
    for name, model_info in MODELS.items():
        if os.path.exists(model_info["path"]):
            try:
                model_info["session"] = ort.InferenceSession(model_info["path"], providers=providers)
                print(f"โหลดโมเดล {name} สำเร็จ")
            except Exception as e:
                print(f"ไม่สามารถโหลดโมเดล {name}: {str(e)}")
        else:
            print(f"ไม่พบไฟล์โมเดล {name} ที่ {model_info['path']}")

load_available_models()

def preprocess_face(face_img, target_size=(112, 112)):
    # ปรับขนาดภาพ
    if face_img.shape[0] != target_size[0] or face_img.shape[1] != target_size[1]:
        face_img = cv2.resize(face_img, target_size)
    
    # แปลงให้เป็น RGB ถ้าเป็น BGR
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    face_img = face_img.astype(np.float32) / 255.0
    face_img = (face_img - 0.5) / 0.5
    
    # เปลี่ยนรูปร่างเป็น NCHW
    face_img = np.transpose(face_img, (2, 0, 1))
    face_img = np.expand_dims(face_img, axis=0)
    
    return face_img

def get_embedding(model_name, face_img):
    model_info = MODELS[model_name]
    if model_info["session"] is None:
        return None
    
    # เตรียมรูปภาพ
    preprocessed = preprocess_face(face_img)
    
    # รัน inference
    input_name = model_info["session"].get_inputs()[0].name
    embedding = model_info["session"].run(None, {input_name: preprocessed})[0]
    
    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    
    return embedding

def ensemble_face_recognition(face_img, weights=None):
    # ถ้าไม่ได้กำหนด weights ให้ใช้ค่าเริ่มต้น
    if weights is None:
        weights = {name: model_info["default_weight"] for name, model_info in MODELS.items() 
                  if model_info["session"] is not None}
    
    # ทำให้น้ำหนักรวมกันเป็น 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
    else:
        return None
    
    # คำนวณ embedding จากแต่ละโมเดลและรวมกัน
    combined_embedding = None
    embedding_size = None
    
    for model_name, weight in normalized_weights.items():
        if weight > 0 and model_name in MODELS and MODELS[model_name]["session"] is not None:
            embedding = get_embedding(model_name, face_img)
            if embedding is not None:
                if combined_embedding is None:
                    embedding_size = embedding.shape[1]
                    combined_embedding = np.zeros((1, embedding_size), dtype=np.float32)
                combined_embedding += embedding * weight
    
    # Normalize อีกครั้ง
    if combined_embedding is not None:
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
    
    return combined_embedding

def decode_base64_image(base64_str):
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/compare', methods=['POST'])
def compare_faces():
    data = request.json
    
    # แปลงรูปภาพจาก base64
    try:
        img1 = decode_base64_image(data['image1'])
        img2 = decode_base64_image(data['image2'])
    except Exception as e:
        return jsonify({'error': f'Failed to decode images: {str(e)}'}), 400
    
    # ตรวจสอบน้ำหนักโมเดล
    weights = data.get('model_weights', None)
    
    # กรองเอาเฉพาะโมเดลที่มีอยู่
    if weights:
        weights = {k: v for k, v in weights.items() if k in MODELS and MODELS[k]["session"] is not None}
    
    # สร้าง embeddings
    emb1 = ensemble_face_recognition(img1, weights)
    emb2 = ensemble_face_recognition(img2, weights)
    
    if emb1 is None or emb2 is None:
        return jsonify({'error': 'Failed to generate embeddings'}), 500
    
    # คำนวณความเหมือน
    similarity = float(np.sum(emb1 * emb2))
    
    # ค่า threshold เริ่มต้น
    threshold = 0.20
    is_match = similarity >= threshold
    
    # รวบรวมผลลัพธ์จากแต่ละโมเดล
    model_details = {}
    for model_name in MODELS:
        if MODELS[model_name]["session"] is not None:
            emb1_single = get_embedding(model_name, img1)
            emb2_single = get_embedding(model_name, img2)
            if emb1_single is not None and emb2_single is not None:
                model_details[model_name] = float(np.sum(emb1_single * emb2_single))
    
    result = {
        "is_match": bool(is_match),
        "similarity": similarity,
        "confidence": similarity * 100,
        "model_details": model_details
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
