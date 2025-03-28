#!/bin/bash

echo "🔄 เริ่มการแก้ไขปัญหาขั้นสุดท้ายสำหรับระบบ FaceSocial..."

# 1. หยุดและลบ containers เดิม
echo "🛑 กำลังหยุดและลบ containers เดิม..."
docker-compose down -v

# 2. ลบ images เก่า
echo "🧹 กำลังลบ images เก่า..."
docker system prune -af --volumes

# 3. แก้ไขปัญหาต่างๆ
echo "🔧 กำลังแก้ไขปัญหาที่พบ..."

# 3.1 แก้ไข docker-compose.yml ให้มีการรัน restart ได้น้อยลง
cat > docker-compose.yml << 'EOF2'
services:
  # API Gateway
  api-gateway:
    build: ./services/api-gateway
    ports:
      - "8000:8000"
    volumes:
      - ./services/api-gateway:/app
    depends_on:
      - face-detection
      - face-recognition
      - liveness
      - deepfake
    networks:
      - facesocial-network
    restart: "no"

  # Face Detection Service
  face-detection:
    build: ./services/face-detection
    volumes:
      - ./models/face-detection:/app/models
      - ./services/face-detection:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - facesocial-network
    restart: "no"

  # Face Recognition Service
  face-recognition:
    build: ./services/face-recognition
    volumes:
      - ./models/face-recognition:/app/models
      - ./services/face-recognition:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - facesocial-network
    restart: "no"

  # Liveness Detection Service
  liveness:
    build: ./services/liveness
    volumes:
      - ./models/liveness:/app/models
      - ./services/liveness:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - facesocial-network
    restart: "no"

  # Deepfake Detection Service
  deepfake:
    build: ./services/deepfake
    volumes:
      - ./models/deepfake:/app/models
      - ./services/deepfake:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - facesocial-network
    restart: "no"

  # Frontend
  frontend:
    build: ./services/frontend
    ports:
      - "3000:3000"
    volumes:
      - ./services/frontend:/usr/share/nginx/html
    depends_on:
      - api-gateway
    networks:
      - facesocial-network
    restart: "no"

networks:
  facesocial-network:
    driver: bridge
EOF2

# 4. รันระบบใหม่
echo "🚀 กำลังรันระบบ FaceSocial..."
docker-compose up -d

# 5. รอให้ระบบเริ่มต้น
echo "⏳ กำลังรอให้ระบบเริ่มต้น..."
sleep 10

# 6. ตรวจสอบสถานะ
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

ข้อแนะนำ:
- ถ้ามีปัญหากับ service ใด ให้ลองรันเฉพาะ service นั้นโดยใช้:
  docker-compose up -d --build <service_name>

- หากบางฟีเจอร์ไม่ทำงาน เนื่องจากนี่เป็นเวอร์ชันสาธิต ที่ไม่ได้ใช้โมเดล AI จริง
"
