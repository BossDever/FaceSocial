#!/bin/bash

echo "เริ่มการตั้งค่าโปรเจค FaceSocial..."

# ตรวจสอบว่า Docker มีการเปิดใช้งาน GPU หรือไม่
if docker info | grep -q "Runtimes:.*nvidia"; then
  echo "✅ พบการตั้งค่า NVIDIA Runtime ใน Docker"
else
  echo "⚠️ ไม่พบการตั้งค่า NVIDIA Runtime ใน Docker ระบบจะใช้ CPU แทน"
fi

# สร้าง Docker Images และรัน Docker Compose
echo "กำลังสร้างและรัน Docker containers..."
docker-compose up -d

echo "กำลังตรวจสอบสถานะ containers..."
sleep 5
docker-compose ps

echo "
🎉 การตั้งค่าเสร็จสิ้น!

คุณสามารถเข้าถึงระบบได้ที่:
- Frontend: http://localhost:3000
- API Gateway: http://localhost:8000

สามารถดู logs ได้ด้วยคำสั่ง:
- docker-compose logs -f

หยุดการทำงานของระบบด้วยคำสั่ง:
- docker-compose down
"
