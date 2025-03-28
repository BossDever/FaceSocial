#!/bin/bash

# สร้างโฟลเดอร์ที่จำเป็น
mkdir -p models/face-detection
mkdir -p models/face-recognition
mkdir -p models/liveness
mkdir -p models/deepfake/domain_adapt_models
mkdir -p models/deepfake/ela_models

# ดาวน์โหลดโมเดล Face Detection (SCRFD)
echo "กำลังดาวน์โหลดโมเดล Face Detection..."
wget -q -O models/face-detection/scrfd_10g_bnkps.onnx https://github.com/deepinsight/insightface/raw/master/detection/scrfd/onnx/scrfd_10g_bnkps.onnx

# ดาวน์โหลดโมเดล Liveness Detection
echo "กำลังดาวน์โหลดโมเดล Liveness Detection..."
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git tmp-liveness
mkdir -p models/liveness
if [ -d "tmp-liveness/resources/anti_spoof_models" ]; then
  cp -r tmp-liveness/resources/anti_spoof_models/* models/liveness/
  echo "คัดลอกโมเดล Liveness Detection สำเร็จ"
else
  echo "ไม่พบโฟลเดอร์โมเดล Liveness Detection"
fi
rm -rf tmp-liveness

# สร้างโมเดล FaceNet
echo "กำลังสร้างโมเดล FaceNet..."
python3 download_facenet.py

echo "ดาวน์โหลดและสร้างโมเดลเสร็จสิ้น"
echo "โปรดดาวน์โหลดหรือคัดลอกโมเดลเหล่านี้ด้วยตนเอง:"
echo "1. models/face-recognition/arcface_r100.onnx (ArcFace)"
echo "2. models/face-recognition/adaface_ir101_webface12m.onnx (AdaFace)"
echo "3. models/face-recognition/elasticface.onnx (ElasticFace)"
echo "4. models/deepfake/domain_adapt_models/domain_adaptation_model.pth"
echo "5. models/deepfake/ela_models/*.pth (ELA models)"
