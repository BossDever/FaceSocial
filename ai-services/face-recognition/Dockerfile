FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

WORKDIR /app

# ติดตั้งแพคเกจที่จำเป็น
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ติดตั้ง Python packages
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    opencv-python==4.8.1.78 \
    mtcnn==0.1.1 \
    scikit-learn==1.3.0 \
    pymilvus==2.3.4 \
    redis==4.6.0 \
    python-multipart \
    onnxruntime-gpu==1.16.0

# สร้างโฟลเดอร์สำหรับเก็บโมเดล
RUN mkdir -p /app/models/facenet

# Copy application code
COPY . /app/

# Expose port
EXPOSE 8001

# กำหนด working directory
WORKDIR /app

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]