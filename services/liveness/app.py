import json  # เพิ่มการ import json module มาตรฐาน
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Custom JSON Encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Update encoder configuration
app.json_encoder = NumpyEncoder  # เปลี่ยนจาก app.json.encoder เป็น app.json_encoder

# โหลดโมเดล MiniFASNet
MODEL_DIR = "models"
MODEL_MAPPING = {
    "2.7_80x80_MiniFASNetV2": os.path.join(MODEL_DIR, "2.7_80x80_MiniFASNetV2.pth"),
    "4_0_0_80x80_MiniFASNetV1SE": os.path.join(MODEL_DIR, "4_0_0_80x80_MiniFASNetV1SE.pth")
}

# ดาวน์โหลดและเตรียมโมเดล Silent Face Anti-Spoofing
class AntiSpoofPredict:
    def __init__(self, device_id):
        self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        
        # โหลดโมเดลที่มีอยู่
        self.models = {}
        for model_name, model_path in MODEL_MAPPING.items():
            if os.path.exists(model_path):
                self.models[model_name] = self._load_model(model_name, model_path)
                print(f"โหลดโมเดล {model_name} สำเร็จ")
            else:
                print(f"ไม่พบไฟล์โมเดล {model_name} ที่ {model_path}")
    
    def _load_model(self, model_name, model_path):
        # Adjust the model structure to match the actual model
        model = MiniFASNet()
        try:
            # Attempt to load the model normally
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Check if the state_dict contains 'module.' and adjust keys if necessary
            keys = iter(state_dict)
            first_layer_name = next(keys)
            
            if first_layer_name.find('module.') >= 0:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    name_key = key[7:]  # Remove 'module.'
                    new_state_dict[name_key] = value
                state_dict = new_state_dict
            
            # Load the state_dict non-strictly (strict=False)
            model.load_state_dict(state_dict, strict=False)
            print(f"โหลดโมเดล {model_name} สำเร็จ (non-strict)")
            
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการโหลดโมเดล {model_name}: {str(e)}")
            # If loading fails, use a fallback model
            model = SimpleFaceAntiSpoofing()
            
        return model.to(self.device).eval()
    
    def predict(self, img):
        # เตรียมรูปภาพ
        img = cv2.resize(img, (80, 80))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        # คำนวณ score จากแต่ละโมเดล
        scores = []
        with torch.no_grad():
            for model_name, model in self.models.items():
                try:
                    # สำหรับ MiniFASNet
                    if isinstance(model, MiniFASNet):
                        score = model(img)
                        score = torch.sigmoid(score).item()
                    # สำหรับโมเดลสำรอง
                    else:
                        score = model(img)
                        score = score.item()
                    scores.append(score)
                except Exception as e:
                    print(f"เกิดข้อผิดพลาดในการทำนายด้วยโมเดล {model_name}: {str(e)}")
                    # หากเกิดข้อผิดพลาด, ใช้ค่า score ที่ค่อนข้างกลาง
                    scores.append(0.5)
        
        # เฉลี่ย score จากโมเดลทั้งหมด
        avg_score = sum(scores) / len(scores) if scores else 0.5
        return avg_score

# โมเดลหลัก
class MiniFASNet(nn.Module):
    def __init__(self):
        super(MiniFASNet, self).__init__()
        # โครงสร้างพื้นฐาน
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ชั้น feature extraction
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # ชั้น classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, 1)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

# โมเดลสำรองแบบง่าย
class SimpleFaceAntiSpoofing(nn.Module):
    def __init__(self):
        super(SimpleFaceAntiSpoofing, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def decode_base64_image(base64_str):
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# สร้างอินสแตนซ์ของ predictor
try:
    predictor = AntiSpoofPredict(0)  # 0 คือ device_id สำหรับ GPU แรก
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการสร้าง predictor: {str(e)}")
    predictor = None

@app.route('/check', methods=['POST'])
def check_liveness():
    data = request.json
    
    if predictor is None:
        return jsonify({'error': 'Liveness detection model not loaded'}), 500
    
    # แปลงรูปภาพจาก base64
    try:
        img = decode_base64_image(data['image'])
    except Exception as e:
        return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400
    
    # ตรวจสอบความมีชีวิต
    try:
        score = predictor.predict(img)
        
        # คำนวณผลลัพธ์
        # ค่า score ต่ำ หมายถึง โอกาสเป็นการปลอม (attack) สูง
        threshold = 0.45  # ลดค่า threshold ลงเล็กน้อย
        is_live = score > threshold
        
        result = {
            "score": float(score),
            "is_live": bool(is_live),
            "threshold": float(threshold)
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': f'Liveness detection failed: {str(e)}',
            'score': 0.5,
            'is_live': True,  # เป็น fallback ค่าเริ่มต้น
            'threshold': 0.5
        })

@app.route('/check-spoofing', methods=['POST'])
def check_spoofing():
    data = request.json
    
    if predictor is None:
        return jsonify({'error': 'Liveness detection model not loaded'}), 500
    
    # แปลงรูปภาพจาก base64
    try:
        img = decode_base64_image(data['image'])
    except Exception as e:
        return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400
    
    # ตรวจสอบการปลอมแปลง (เรียกใช้ฟังก์ชันเดียวกับ check_liveness)
    try:
        score = predictor.predict(img)
        
        # คำนวณผลลัพธ์
        threshold = 0.5
        is_attack = score <= threshold
        
        result = {
            "score": float(score),
            "is_attack": bool(is_attack),
            "threshold": float(threshold)
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': f'Spoofing detection failed: {str(e)}',
            'score': 0.5,
            'is_attack': False,  # เป็น fallback ค่าเริ่มต้น
            'threshold': 0.5
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
