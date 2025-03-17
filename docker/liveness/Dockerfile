FROM facesocial-base:12.8

LABEL maintainer="FaceSocial AI Team"
LABEL version="1.0"
LABEL description="Liveness Detection service with GPU acceleration for FaceSocial"

# ติดตั้ง dependencies สำหรับ liveness detection
RUN pip install --no-cache-dir \
    onnx==1.15.0 \
    onnxruntime-gpu==1.15.1 \
    numpy==1.24.4 \
    scipy==1.11.3 \
    scikit-image==0.21.0 \
    opencv-python-headless==4.8.1.78 \
    Pillow==10.1.0 \
    PyYAML==6.0.1 \
    tqdm==4.66.1

# สร้างโครงสร้างโฟลเดอร์สำหรับโมเดลต่างๆ
RUN mkdir -p /app/models/liveness-detection/3ddfa-v2/onnx \
    /app/models/liveness-detection/3ddfa-v2/tensorrt \
    /app/models/liveness-detection/3ddfa-v2/original \
    /app/models/liveness-detection/reflection-analysis

# คัดลอก application code
COPY . /app/

# เปลี่ยนเป็น non-root user
USER facesocial

# ตั้งค่า environment variables สำหรับ TensorRT
ENV TRT_PRECISION=fp16
ENV TRT_MAX_WORKSPACE_SIZE=4294967296

# ตั้งค่า entrypoint และคำสั่งเริ่มต้น
CMD ["python /app/liveness-detection/main.py"]