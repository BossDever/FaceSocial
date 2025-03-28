#!/bin/bash

echo "กำลังทำความสะอาดระบบ Docker..."

# หยุดและลบ containers
echo "หยุดการทำงานของ containers..."
docker-compose down -v

# ลบ images ที่ไม่ได้ใช้งาน
echo "ลบ images ที่ไม่ได้ใช้งาน..."
docker system prune -af --volumes

echo "เริ่มการสร้าง containers ใหม่..."
docker-compose up -d

echo "กำลังตรวจสอบสถานะ containers..."
sleep 10
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
