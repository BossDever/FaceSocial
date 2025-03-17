FROM facesocial-base:11.8

LABEL maintainer="FaceSocial AI Team"
LABEL version="1.0"
LABEL description="Face Detection service with GPU acceleration for FaceSocial"

# ติดตั้ง dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.3 \
    pydantic-settings==2.1.0 \
    pillow==10.1.0 \
    opencv-python-headless==4.8.1.78 \
    numpy==1.24.4 \
    gdown==4.7.3 \
    loguru==0.7.2 \
    python-jose==3.3.0 \
    passlib==1.7.4

# สร้างโครงสร้างโฟลเดอร์
RUN mkdir -p /app/models/face-detection/scrfd

# ตั้งค่า unbuffered output เพื่อให้เห็น logs ได้ชัดเจน
ENV PYTHONUNBUFFERED=1

# คัดลอกโค้ด
COPY ./ai-services /app

# เปลี่ยนเป็น non-root user
USER facesocial

# รัน Service
WORKDIR /app/face-detection
CMD ["python", "-u", "main.py"]