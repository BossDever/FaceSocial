FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

WORKDIR /app

# ติดตั้งแพคเกจที่จำเป็น
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Use apt-file to help locate CUDA packages
    apt-file \
    && apt-file update \
    # Install specific CUDA and cuDNN packages required by ONNX Runtime
    && apt-get install -y --no-install-recommends \
    cuda-cudart-11-8 \
    cuda-nvrtc-11-8 \
    cuda-nvtx-11-8 \
    libcufft-11-8 \
    libcurand-11-8 \
    libcusolver-11-8 \
    libcusparse-11-8 \
    libcublas-11-8 \
    libcublaslt-11-8 \
    libcudnn8=8.6.0* \
    && rm -rf /var/lib/apt/lists/*

# Add library paths to find CUDA libraries
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda-11/lib64:/usr/lib/x86_64-linux-gnu

# Create a setup script to verify and fix library paths
RUN echo '#!/bin/bash \n\
# Check if CUDA libraries exist and create links if needed \n\
echo "Checking CUDA libraries for ONNX Runtime..." \n\
for lib in libcublas.so.11 libcublasLt.so.11 libcudnn.so.8; do \n\
  if [ -f "/usr/lib/x86_64-linux-gnu/$lib" ]; then \n\
    echo "Found $lib" \n\
  else \n\
    # Try to find the library elsewhere in the system \n\
    LIB_PATH=$(find /usr -name "$lib" | head -n 1) \n\
    if [ -n "$LIB_PATH" ]; then \n\
      echo "Creating symlink for $lib from $LIB_PATH" \n\
      ln -sf "$LIB_PATH" "/usr/lib/x86_64-linux-gnu/$lib" \n\
    else \n\
      echo "WARNING: $lib not found" \n\
    fi \n\
  fi \n\
done \n\
echo "Library setup complete." \n\
# Start the actual application \n\
if [ "$ENVIRONMENT" = "development" ]; then \n\
  uvicorn main:app --host 0.0.0.0 --port 8001 --reload; \n\
else \n\
  uvicorn main:app --host 0.0.0.0 --port 8001; \n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Make sure the start script is executable
RUN chmod +x /app/start.sh

# Create start.sh script with CUDA library setup
COPY app/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Explicitly install CUDA libraries
RUN apt-get update && apt-get install -y \
    aptitude \
    && aptitude search -F "%p" "libcublas|libcudnn" \
    && apt-get install -y --no-install-recommends \
    libcublas-12-* \
    libcublaslt-12-* \
    libcudnn8* \
    && rm -rf /var/lib/apt/lists/*

# Add our CUDA setup tools
RUN mkdir -p /app/tools
COPY app/tools/cuda_setup.py /app/tools/
RUN chmod +x /app/tools/cuda_setup.py

# Find CUDA library locations for debugging
RUN echo "==== CUDA Library Locations ====" && \
    find /usr/ -name "libcublas.so*" || true && \
    find /usr/ -name "libcublasLt.so*" || true && \
    find /usr/ -name "libcudnn.so*" || true && \
    echo "================================"

# Run CUDA setup during image build to ensure proper symbolic links
RUN python /app/tools/cuda_setup.py

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

# Run the application using our updated start script
CMD ["/app/start.sh"]