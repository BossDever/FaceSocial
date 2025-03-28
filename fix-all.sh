#!/bin/bash

echo "🔄 เริ่มการแก้ไขปัญหาและรันระบบ FaceSocial..."

# 1. แก้ไขปัญหาโมเดล SCRFD
echo "🔧 กำลังแก้ไขปัญหาโมเดล Face Detection..."
rm -f models/face-detection/scrfd_10g_bnkps.onnx
mkdir -p models/face-detection

# ลองดาวน์โหลดโมเดล RetinaFace แทน
echo "⬇️ กำลังดาวน์โหลดโมเดล RetinaFace..."
wget -q -O models/face-detection/retinaface_r50_v1.onnx https://github.com/deepinsight/insightface/raw/master/detection/retinaface/model/det_onnx/retinaface_r50_v1.onnx

# แก้ไขไฟล์ app.py ให้ใช้โมเดล RetinaFace แทน
sed -i 's/scrfd_10g_bnkps.onnx/retinaface_r50_v1.onnx/g' services/face-detection/app.py

# 2. หยุดและลบ containers เดิม
echo "🛑 กำลังหยุดและลบ containers เดิม..."
docker-compose down -v

# ลบ images เก่า
echo "🧹 กำลังลบ images เก่า..."
docker system prune -af --volumes

# 3. รันระบบใหม่
echo "🚀 กำลังรันระบบ FaceSocial..."
docker-compose up -d

# 4. รอให้ระบบเริ่มต้น
echo "⏳ กำลังรอให้ระบบเริ่มต้น..."
sleep 10

# 5. ตรวจสอบสถานะ
echo "🔍 สถานะของระบบ:"
docker-compose ps

echo "
✅ การแก้ไขเสร็จสิ้น!

คุณสามารถเข้าถึงระบบได้ที่:
- Frontend: http://localhost:3000
- API Gateway: http://localhost:8000

สามารถดู logs ได้ด้วยคำสั่ง:
- docker-compose logs -f <service_name>
  ตัวอย่าง: docker-compose logs -f face-detection

ดู logs เพื่อตรวจสอบปัญหา:
- docker-compose logs face-detection
- docker-compose logs face-recognition
- docker-compose logs liveness
- docker-compose logs deepfake
- docker-compose logs frontend
- docker-compose logs api-gateway

สามารถซ่อมแซมแต่ละ service โดยการ:
- docker-compose up -d --build <service_name>
"
