FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

WORKDIR /app

# ติดตั้งแพคเกจที่จำเป็น
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # More specific CUDA dependencies for ONNX Runtime
    cuda-cudart-11-8 \
    libcublas-11-8 \
    libcublaslt-11-8 \
    libcudnn8=8.6.0* \
    && rm -rf /var/lib/apt/lists/*

# Add library paths to find CUDA libraries
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu

# Create symbolic links if needed
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 /usr/lib/x86_64-linux-gnu/libcublasLt.so \
    && ln -sf /usr/lib/x86_64-linux-gnu/libcublas.so.11 /usr/lib/x86_64-linux-gnu/libcublas.so \
    && ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so

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
    onnxruntime-gpu==1.16.0 \
    requests

# สร้างโฟลเดอร์สำหรับเก็บโมเดล
RUN mkdir -p /app/models/facenet

# Copy application code
COPY . /app/

# Expose port
EXPOSE 8001

# กำหนด working directory
WORKDIR /app

# Environment variables for configuration
ENV MODEL_PATH=/app/models/facenet/20180402-114759.pb \
    MILVUS_HOST=milvus \
    MILVUS_PORT=19530 \
    REDIS_HOST=redis \
    REDIS_PORT=6379 \
    ENVIRONMENT=production

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health').raise_for_status()"

# Run the application
CMD if [ "$ENVIRONMENT" = "development" ]; then \
        uvicorn main:app --host 0.0.0.0 --port 8001 --reload; \
    else \
        uvicorn main:app --host 0.0.0.0 --port 8001; \
    fi