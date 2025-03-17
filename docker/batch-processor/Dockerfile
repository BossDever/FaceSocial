FROM facesocial-base:12.8

LABEL maintainer="FaceSocial AI Team"
LABEL version="1.0"
LABEL description="Batch Processing service with multi-GPU acceleration for FaceSocial"

# ติดตั้ง dependencies สำหรับ batch processing
RUN pip install --no-cache-dir \
    onnx==1.15.0 \
    onnxruntime-gpu==1.15.1 \
    ray==2.9.0 \
    pymilvus==2.3.4 \
    redis==5.0.1 \
    insightface==0.7.1 \
    Pillow==10.1.0 \
    numpy==1.24.4 \
    pyarrow==14.0.1 \
    pandas==2.0.3 \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    python-multipart==0.0.6 \
    PyYAML==6.0.1 \
    tqdm==4.66.1

# สร้างโครงสร้างโฟลเดอร์ที่จำเป็น
RUN mkdir -p /app/batch-processor/logs /app/batch-processor/temp

# คัดลอก application code
COPY . /app/

# เปลี่ยนเป็น non-root user
USER facesocial

# ตั้งค่า environment variables สำหรับ Ray
ENV RAY_memory_monitor_refresh_ms=200
ENV RAY_object_store_memory=4000000000

# ตั้งค่า entrypoint และคำสั่งเริ่มต้น
CMD ["python /app/batch-processor/main.py"]