FROM facesocial-base:12.8

LABEL maintainer="FaceSocial AI Team"
LABEL version="1.0"
LABEL description="Face Detection and Deepfake Detection service with GPU acceleration for FaceSocial"

# ติดตั้ง dependencies สำหรับ face detection และ deepfake detection
RUN pip install --no-cache-dir \
    onnx==1.15.0 \
    onnxruntime-gpu==1.15.1 \
    insightface==0.7.1 \
    Pillow==10.1.0 \
    numpy==1.24.4 \
    scikit-image==0.21.0 \
    opencv-python-headless==4.8.1.78 \
    albumentations==1.3.1 \
    timm==0.9.5 \
    einops==0.7.0 \
    PyYAML==6.0.1 \
    tqdm==4.66.1

# สร้างโครงสร้างโฟลเดอร์สำหรับโมเดลต่างๆ
RUN mkdir -p /app/models/face-detection/scrfd \
    /app/models/face-detection/retinaface \
    /app/models/deepfake-detection/xception/onnx \
    /app/models/deepfake-detection/xception/tensorrt \
    /app/models/deepfake-detection/xception/original \
    /app/models/deepfake-detection/efficientnet/onnx \
    /app/models/deepfake-detection/efficientnet/tensorrt \
    /app/models/deepfake-detection/efficientnet/original

# คัดลอก application code
COPY . /app/

# เปลี่ยนเป็น non-root user
USER facesocial

# ตั้งค่า environment variables สำหรับ TensorRT
ENV TRT_PRECISION=fp16
ENV TRT_MAX_WORKSPACE_SIZE=4294967296

# ตั้งค่า entrypoint และคำสั่งเริ่มต้น
CMD ["python /app/face-detection/main.py"]