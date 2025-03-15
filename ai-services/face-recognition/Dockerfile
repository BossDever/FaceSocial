FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

WORKDIR /app

# Install basic dependencies only (NVIDIA container already has CUDA)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Add library paths to find CUDA libraries
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu

# Setup directories for CUDA tools
RUN mkdir -p /app/tools

# Create tools directory and copy setup scripts
COPY app/tools/cuda_libraries_fix.sh /app/tools/
RUN chmod +x /app/tools/*.sh

# Install Python packages
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    opencv-python==4.8.1.78 \
    mtcnn==0.1.1 \
    scikit-learn==1.3.0 \
    pymilvus==2.3.4 \
    redis==4.6.0 \
    python-multipart \
    onnxruntime-gpu==1.17.0 \
    requests

# Create directories for models
RUN mkdir -p /app/models/facenet /app/models/arcface /app/models/cosface /app/models/gender

# Copy application code
COPY . /app/

# Expose port
EXPOSE 8001

# Working directory
WORKDIR /app

# Environment variables
ENV MODEL_PATH=/app/models/facenet/20180402-114759.pb \
    MILVUS_HOST=milvus \
    MILVUS_PORT=19530 \
    REDIS_HOST=redis \
    REDIS_PORT=6379 \
    ENVIRONMENT=production

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health').raise_for_status()"

# Run the application using our start script
CMD ["/app/start.sh"]