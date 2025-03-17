# ใช้ base image ใหม่ที่มี CUDA 11.8
FROM facesocial-base:11.8

LABEL maintainer="FaceSocial AI Team"
LABEL version="1.0"
LABEL description="Face Recognition service with GPU acceleration for FaceSocial"

# ติดตั้ง dependencies สำหรับ face recognition
RUN pip install --no-cache-dir \
    pymilvus==2.3.4 \
    redis==5.0.1 \
    insightface==0.7.1 \
    facenet-pytorch \
    Pillow==10.1.0 \
    numpy==1.24.4 \
    scikit-image==0.21.0 \
    scikit-learn==1.3.0 \
    opencv-python-headless==4.8.1.78 \
    PyYAML==6.0.1 \
    tqdm==4.66.1

# สร้างโครงสร้างโฟลเดอร์สำหรับโมเดลต่างๆ
RUN mkdir -p /app/models/facenet/onnx /app/models/facenet/tensorrt /app/models/facenet/original \
    /app/models/arcface/onnx /app/models/arcface/tensorrt /app/models/arcface/original \
    /app/models/elasticface/onnx /app/models/elasticface/tensorrt /app/models/elasticface/original \
    /app/models/adaface/onnx /app/models/adaface/tensorrt /app/models/adaface/original \
    /app/models/gender-detection/onnx /app/models/gender-detection/tensorrt /app/models/gender-detection/original \
    /app/models/age-detection/onnx /app/models/age-detection/tensorrt /app/models/age-detection/original

# คัดลอก application code
COPY . /app/

# เปลี่ยนเป็น non-root user
USER facesocial

# ตั้งค่า environment variables สำหรับ TensorRT
ENV TRT_PRECISION=fp16
ENV TRT_MAX_WORKSPACE_SIZE=4294967296

# ตั้งค่า entrypoint และคำสั่งเริ่มต้น
CMD ["python /app/face-recognition/main.py"]