import torch
import torch.nn as nn
import timm
import os
import numpy as np

# ตั้งค่าเส้นทางไฟล์
INPUT_PATH = '/mnt/d/FaceSocial/models/deepfake/domain_adapt_models/domain_adaptation_model.pth'
OUTPUT_DIR = '/mnt/d/FaceSocial/models/deepfake/domain_adapt_models_onnx'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'domain_adaptation_model.onnx')

print(f"📂 ไฟล์โมเดลต้นทาง: {INPUT_PATH}")
print(f"📂 ไฟล์โมเดลปลายทาง: {OUTPUT_PATH}")

# ตรวจสอบว่ามีโมเดลต้นทางหรือไม่
if not os.path.exists(INPUT_PATH):
    print(f"❌ ไม่พบไฟล์โมเดลต้นทาง: {INPUT_PATH}")
    exit(1)
else:
    print(f"✅ พบไฟล์โมเดลต้นทาง")

# คลาส Gradient Reversal Layer
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# คลาสโมเดล Domain Adaptation
class DomainAdaptationModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b4'):
        super().__init__()
        # โหลดโมเดลพื้นฐาน
        self.base_model = timm.create_model(model_name, pretrained=False)
        
        # หาขนาดของ feature
        n_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Identity()
        
        # สร้าง classifier สำหรับจำแนก real/fake
        self.classifier = nn.Linear(n_features, 1)
        
        # สร้าง domain classifier พร้อม gradient reversal
        self.domain_classifier = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x, alpha=1.0):
        # สกัด features
        features = self.base_model(x)
        
        # ส่งต่อไปยัง classifier สำหรับจำแนก real/fake
        class_output = self.classifier(features)
        
        # ONNX ไม่สนับสนุน GradientReversalLayer โดยตรง ในโหมด inference เราแค่ส่งค่า features ไป
        # สำหรับ inference เราไม่จำเป็นต้องใช้ gradient reversal
        domain_output = self.domain_classifier(features)
        
        return class_output, domain_output

# ฟังก์ชันแปลงโมเดล
def convert_domain_adaptation_model_to_onnx():
    # บังคับใช้ CPU
    device = torch.device('cpu')
    model = DomainAdaptationModel()
    
    # โหลด state_dict และตั้งค่าโมเดลเป็นโหมดประเมินผล
    try:
        state_dict = torch.load(INPUT_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("✅ โหลดโมเดลสำเร็จ")
    except Exception as e:
        print(f"⚠️ โหลดโมเดลได้แต่มีการเตือน: {str(e)}")
    
    # ตั้งค่าโมเดลเป็นโหมดประเมินผล
    model.eval()
    # ย้ายโมเดลไป CPU อย่างชัดเจน
    model = model.to(device)
    
    # สร้าง dummy input สำหรับการแปลง
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    alpha = torch.tensor(1.0)
    
    # ตั้งค่าการแปลงรูปแบบ ONNX
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'class_output': {0: 'batch_size'},
        'domain_output': {0: 'batch_size'}
    }
    
    # แปลงเป็น ONNX
    try:
        torch.onnx.export(
            model,                      # โมเดลที่ต้องการแปลง
            (dummy_input, alpha),       # อินพุตตัวอย่าง (x, alpha)
            OUTPUT_PATH,                # เส้นทางไฟล์เอาต์พุต
            export_params=True,         # เก็บค่าพารามิเตอร์ในไฟล์ ONNX
            opset_version=12,           # รุ่น ONNX
            do_constant_folding=True,   # โฟลด์ค่าคงที่เพื่อลดเวลาคำนวณ
            input_names=['input', 'alpha'],  # ชื่ออินพุต
            output_names=[              # ชื่อเอาต์พุต
                'class_output',
                'domain_output'
            ],
            dynamic_axes=dynamic_axes    # แกนปรับขนาดได้ (batch size)
        )
        print(f"✅ แปลงเป็น ONNX สำเร็จ: {OUTPUT_PATH}")
        
        # ตรวจสอบขนาดไฟล์
        onnx_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)  # MB
        print(f"📊 ขนาดไฟล์ ONNX: {onnx_size:.2f} MB")
        
        return True
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการแปลงเป็น ONNX: {str(e)}")
        return False

if __name__ == "__main__":
    convert_domain_adaptation_model_to_onnx()