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

# ตรวจสอบว่าใช้ GPU ได้หรือไม่
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ใช้อุปกรณ์: {device}")

# ========== Multi-Task Model สำหรับ ELA ==========
class MultiTaskModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4', pretrained=False):
        super().__init__()
        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained)

        # ลบ classifier เดิม
        if hasattr(self.feature_extractor, 'classifier'):
            n_features = self.feature_extractor.classifier.in_features
            self.feature_extractor.classifier = nn.Identity()
        elif hasattr(self.feature_extractor, 'head'):
            n_features = self.feature_extractor.head.fc.in_features
            self.feature_extractor.head.fc = nn.Identity()
        else:
            raise ValueError("ไม่สามารถดึงขนาดของ feature ได้")

        # Classifier (real/fake)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_features, 1)
        )

        # Regressor (error level)
        self.error_regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        error_level = self.error_regressor(features)
        return class_output, error_level

# ========== Stacking Ensemble ==========
class StackingEnsemble(nn.Module):
    def __init__(self, base_models, model_name='tf_efficientnet_b4'):
        super().__init__()
        self.base_models = base_models

        # Meta model (stacking layer)
        self.meta_model = nn.Sequential(
            nn.Linear(len(base_models), 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        base_preds = []
        for model in self.base_models:
            with torch.no_grad():
                class_output, _ = model(x)
                probs = torch.sigmoid(class_output)
                base_preds.append(probs)

        meta_features = torch.cat(base_preds, dim=1)
        final_output = self.meta_model(meta_features)
        return final_output

# ========== ฟังก์ชันโหลด ELA Models ทั้งหมด ==========
def load_ela_models(ela_models_dir='models/ela_models', model_name='tf_efficientnet_b4'):
    # ตรวจสอบโฟลเดอร์
    if not os.path.exists(ela_models_dir):
        print(f"❌ ไม่พบโฟลเดอร์ ELA models ที่ {ela_models_dir}")
        return None

    base_models = []
    for i in range(5):
        model_path = os.path.join(ela_models_dir, f"ela_model_fold{i}.pth")
        if os.path.exists(model_path):
            try:
                model = MultiTaskModel(model_name=model_name, pretrained=False).to(device)
                state_dict = torch.load(model_path, map_location=device)

                # ลองโหลดแบบไม่ strict
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                if missing_keys or unexpected_keys:
                    print(f"⚠️ WARNING fold{i}: missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
                    if missing_keys:
                        print(f"  Missing: {missing_keys[:3]}...")
                    if unexpected_keys:
                        print(f"  Unexpected: {unexpected_keys[:3]}...")

                model.eval()
                base_models.append(model)
                print(f"✅ โหลดโมเดล ELA fold {i} สำเร็จ")
            except Exception as e:
                print(f"❌ ไม่สามารถโหลดโมเดล ELA fold {i}: {str(e)}")
        else:
            print(f"⚠️ ไม่พบไฟล์โมเดล fold {i} ที่ {model_path}")

    if not base_models:
        print("❌ ไม่สามารถโหลด base models สำหรับ ELA ได้เลย")
        return None

    # โหลด stacking ensemble
    stacking_path = os.path.join(ela_models_dir, "ela_stacking_ensemble_model.pth")
    try:
        ensemble = StackingEnsemble(base_models, model_name=model_name).to(device)
        if os.path.exists(stacking_path):
            # ลองโหลดแบบไม่ strict
            missing_keys, unexpected_keys = ensemble.load_state_dict(torch.load(stacking_path, map_location=device), strict=False)
            
            if missing_keys or unexpected_keys:
                print(f"⚠️ WARNING stacking: missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
                
            ensemble.eval()
            print("✅ โหลดโมเดล ELA stacking ensemble สำเร็จ")
            return ensemble
        else:
            print(f"⚠️ ไม่พบไฟล์ stacking ensemble ที่ {stacking_path}")
            # ถ้าไม่มี stacking ensemble ให้ใช้โมเดลแรกแทน
            if base_models:
                print("⚠️ ใช้โมเดล fold 0 เป็นตัวแทน")
                return base_models[0]
            return None
    except Exception as e:
        print(f"❌ ไม่สามารถโหลด ELA stacking ensemble: {str(e)}")
        # ถ้าโหลด stacking ไม่สำเร็จ ให้ใช้โมเดลแรกแทน
        if base_models:
            print("⚠️ ใช้โมเดล fold 0 เป็นตัวแทน")
            return base_models[0]
        return None

# ฟังก์ชันเตรียมรูปภาพสำหรับโมเดล
def preprocess_image(img, target_size=(380, 380)):
    # ปรับขนาดภาพ
    img = cv2.resize(img, target_size)
    
    # แปลงเป็น RGB และ normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # Normalize ตามค่า ImageNet
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # แปลงเป็น tensor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img, dtype=torch.float32).to(device)
    
    return img

# ฟังก์ชันสร้างภาพ ELA (Error Level Analysis)
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

# โหลดโมเดล ELA
ela_model = load_ela_models()

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    data = request.json
    
    # แปลงรูปภาพจาก base64
    try:
        img = decode_base64_image(data['image'])
    except Exception as e:
        return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400
    
    # ตรวจสอบว่ามีโมเดล ELA หรือไม่
    if ela_model is None:
        # ถ้าไม่มีโมเดล ใช้วิธีการสำรอง
        print("⚠️ ไม่มีโมเดล ELA ที่ใช้งานได้ ใช้วิธีวิเคราะห์ histogram แทน")
        try:
            # วิเคราะห์ histogram ของแต่ละช่องสี BGR
            hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
            
            # คำนวณความเบี่ยงเบนมาตรฐาน
            std_b = np.std(hist_b)
            std_g = np.std(hist_g)
            std_r = np.std(hist_r)
            
            # ค่าเฉลี่ย
            avg_std = (std_r + std_g + std_b) / 3.0
            
            # ปรับเป็นคะแนน 0-1
            import random
            score = max(0.2, min(0.8, 1.0 - min(1.0, avg_std / 20000.0) * random.uniform(0.85, 1.15)))
            
            threshold = 0.55  # จากเดิม 0.5
            is_fake = score > threshold
            
            result = {
                "score": float(score),
                "is_fake": bool(is_fake),
                "threshold": float(threshold),
                "domain_score": None,
                "ela_score": float(score),
                "note": "Using histogram analysis (ELA model not available)"
            }
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': f'Deepfake detection failed: {str(e)}'}), 500
    
    # ประมวลผลด้วยโมเดล ELA
    try:
        # สร้างภาพ ELA
        ela_img = generate_ela_image(img)
        
        # เตรียมรูปภาพ
        input_tensor = preprocess_image(ela_img)
        
        # ทำนาย
        with torch.no_grad():
            if isinstance(ela_model, StackingEnsemble):
                prediction = ela_model(input_tensor)
                ela_prediction = torch.sigmoid(prediction).item()
            else:
                prediction, _ = ela_model(input_tensor)
                ela_prediction = torch.sigmoid(prediction).item()
        
        # แปลผล
        threshold = 0.55  # จากเดิม 0.5
        is_fake = ela_prediction > threshold
        
        result = {
            "score": float(ela_prediction),
            "is_fake": bool(is_fake),
            "threshold": float(threshold),
            "domain_score": None,
            "ela_score": float(ela_prediction)
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Deepfake detection failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
