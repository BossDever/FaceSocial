from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Path โมเดล
DOMAIN_ADAPT_MODEL_PATH = os.path.join('models', 'domain_adapt_models', 'domain_adaptation_model.pth')
ELA_MODELS_DIR = os.path.join('models', 'ela_models')

# ตรวจสอบว่าใช้ GPU ได้หรือไม่
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# คลาสสำหรับโมเดล Domain Adaptation
class DomainAdaptationModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4'):
        super().__init__()
        # โหลดโมเดลพื้นฐาน
        self.base_model = timm.create_model(model_name, pretrained=True)
        
        # หาขนาดของ feature
        n_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Identity()
        
        # สร้าง classifier สำหรับจำแนก real/fake
        self.classifier = nn.Linear(n_features, 1)
        
        # สร้าง domain classifier (ไม่จำเป็นสำหรับการใช้งานจริง)
        self.domain_classifier = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
    
    def forward(self, x, alpha=1.0):
        # สกัด features
        features = self.base_model(x)
        
        # ส่งต่อไปยัง classifier สำหรับจำแนก real/fake
        class_output = self.classifier(features)
        
        # เฉพาะการทำนาย ไม่จำเป็นต้องใช้ domain classifier
        return class_output

# คลาสสำหรับโมเดล Multi-Task ELA
class MultiTaskModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4', pretrained=False):
        super().__init__()
        # โหลดโมเดลพื้นฐาน
        self.base_model = timm.create_model(model_name, pretrained=pretrained)
        
        # หาขนาดของ feature
        n_features = self.base_model.classifier.in_features
        
        # สร้าง classifier สำหรับจำแนก real/fake
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_features, 1)
        )
        
        # สร้าง regressor สำหรับประมาณค่าระดับความผิดปกติ
        self.error_regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # สกัด feature
        features = self.base_model.forward_features(x)
        
        # global pooling ถ้าจำเป็น
        if len(features.shape) > 2:
            features = self.base_model.global_pool(features)
            
        # flatten ถ้าจำเป็น
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # ส่งต่อไปยัง classifier
        class_output = self.classifier(features)
        
        # ส่งต่อไปยัง regressor
        error_level = self.error_regressor(features)
        
        return class_output, error_level

# คลาสสำหรับ ELA Stacking Ensemble
class StackingEnsemble(nn.Module):
    def __init__(self, base_models, model_name):
        super().__init__()
        self.base_models = base_models
        
        # meta model (stacking layer)
        self.meta_model = nn.Sequential(
            nn.Linear(len(base_models), 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        # เก็บผลการทำนายจากทุกโมเดล
        base_preds = []
        for model in self.base_models:
            with torch.no_grad():
                class_output, _ = model(x)
                probs = torch.sigmoid(class_output)
                base_preds.append(probs)
        
        # รวม predictions เป็น features สำหรับ meta model
        meta_features = torch.cat(base_preds, dim=1)
        
        # ส่งต่อไปยัง meta model
        final_output = self.meta_model(meta_features)
        
        return final_output

# ฟังก์ชันสำหรับโหลดโมเดล Domain Adaptation
def load_domain_adaptation_model():
    # สร้างโมเดล
    model = DomainAdaptationModel()
    
    # ตรวจสอบว่ามีไฟล์โมเดลหรือไม่
    if os.path.exists(DOMAIN_ADAPT_MODEL_PATH):
        # โหลด weights
        state_dict = torch.load(DOMAIN_ADAPT_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("โหลดโมเดล Domain Adaptation สำเร็จ")
    else:
        print(f"ไม่พบไฟล์โมเดล Domain Adaptation ที่ {DOMAIN_ADAPT_MODEL_PATH}")
    
    model = model.to(device)
    model.eval()
    return model

# ฟังก์ชันสำหรับโหลดโมเดล ELA
def load_ela_models():
    # ตรวจสอบโฟลเดอร์ ELA models
    if not os.path.exists(ELA_MODELS_DIR):
        print(f"ไม่พบโฟลเดอร์ ELA models ที่ {ELA_MODELS_DIR}")
        return None
    
    # โหลดโมเดล fold
    base_models = []
    for i in range(5):
        model_path = os.path.join(ELA_MODELS_DIR, f"ela_model_fold{i}.pth")
        if os.path.exists(model_path):
            model = MultiTaskModel('tf_efficientnet_b4', pretrained=False).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            base_models.append(model)
            print(f"โหลดโมเดล ELA fold {i} สำเร็จ")
        else:
            print(f"ไม่พบไฟล์โมเดล ELA fold {i} ที่ {model_path}")
    
    if not base_models:
        print("ไม่สามารถโหลดโมเดล ELA fold ได้")
        return None
    
    # โหลด stacking ensemble
    ensemble = StackingEnsemble(base_models, 'tf_efficientnet_b4').to(device)
    stacking_path = os.path.join(ELA_MODELS_DIR, "ela_stacking_ensemble_model.pth")
    
    if os.path.exists(stacking_path):
        ensemble.load_state_dict(torch.load(stacking_path, map_location=device))
        print("โหลด ELA stacking ensemble สำเร็จ")
        return ensemble
    else:
        print(f"ไม่พบไฟล์โมเดล ELA stacking ensemble ที่ {stacking_path}")
        return None

# โหลดโมเดล
try:
    domain_model = load_domain_adaptation_model()
except Exception as e:
    print(f"ไม่สามารถโหลดโมเดล Domain Adaptation: {str(e)}")
    domain_model = None

try:
    ela_model = load_ela_models()
except Exception as e:
    print(f"ไม่สามารถโหลดโมเดล ELA: {str(e)}")
    ela_model = None

def preprocess_image(img, target_size=(380, 380)):
    # ปรับขนาดภาพ
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # เปลี่ยนรูปร่างเป็น NCHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img

def generate_ela_image(img, quality=90):
    """สร้างภาพ Error Level Analysis (ELA)"""
    # บันทึกเป็นไฟล์ JPEG ชั่วคราว
    temp_filename = 'temp_image.jpg'
    cv2.imwrite(temp_filename, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    # อ่านกลับมาเป็น array
    compressed_img = cv2.imread(temp_filename)
    
    # คำนวณความแตกต่าง (ELA)
    ela = cv2.absdiff(img, compressed_img) * 10
    
    # ลบไฟล์ชั่วคราว
    os.remove(temp_filename)
    
    return ela

def decode_base64_image(base64_str):
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    data = request.json
    
    # แปลงรูปภาพจาก base64
    try:
        img = decode_base64_image(data['image'])
    except Exception as e:
        return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400
    
    # ตรวจสอบว่ามีโมเดลใดโหลดสำเร็จหรือไม่
    if domain_model is None and ela_model is None:
        return jsonify({'error': 'No deepfake detection models loaded'}), 500
    
    # ตรวจจับ Deepfake
    try:
        # ประมวลผลด้วยโมเดล Domain Adaptation
        domain_prediction = None
        if domain_model is not None:
            # เตรียมรูปภาพ
            input_tensor = preprocess_image(img)
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(device)
            
            # ทำนาย
            with torch.no_grad():
                prediction = domain_model(input_tensor)
                domain_prediction = torch.sigmoid(prediction).item()
        
        # ประมวลผลด้วยโมเดล ELA
        ela_prediction = None
        if ela_model is not None:
            # สร้างภาพ ELA
            ela_img = generate_ela_image(img)
            
            # เตรียมรูปภาพ
            input_tensor = preprocess_image(ela_img)
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(device)
            
            # ทำนาย
            with torch.no_grad():
                prediction = ela_model(input_tensor)
                ela_prediction = torch.sigmoid(prediction).item()
        
        # รวมผลการทำนาย
        if domain_prediction is not None and ela_prediction is not None:
            # ถ้ามีทั้งสองโมเดล ถ่วงน้ำหนัก
            final_prediction = domain_prediction * 0.7 + ela_prediction * 0.3
        elif domain_prediction is not None:
            final_prediction = domain_prediction
        elif ela_prediction is not None:
            final_prediction = ela_prediction
        else:
            return jsonify({'error': 'Failed to generate predictions'}), 500
        
        # แปลผล
        threshold = 0.5
        is_fake = final_prediction > threshold
        
        result = {
            "score": float(final_prediction),
            "is_fake": bool(is_fake),
            "threshold": float(threshold),
            "domain_score": float(domain_prediction) if domain_prediction is not None else None,
            "ela_score": float(ela_prediction) if ela_prediction is not None else None
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Deepfake detection failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
