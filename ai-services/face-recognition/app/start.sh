#!/bin/bash

# Enhanced CUDA library setup for ONNX Runtime GPU acceleration
echo "Setting up CUDA libraries for ONNX Runtime..."

# Create critical symlinks directly here first
mkdir -p /usr/lib/x86_64-linux-gnu

# Find source libraries
echo "Finding source libraries..."
CUBLAS_PATH=$(find /usr -name "libcublas.so*" | grep -v cudnn | head -n 1)
CUBLASLT_PATH=$(find /usr -name "libcublasLt.so*" | head -n 1)
CUDNN_PATH=$(find /usr -name "libcudnn.so*" | head -n 1)

# Log what we found
echo "Found libraries:"
echo "  CUBLAS: $CUBLAS_PATH"
echo "  CUBLASLT: $CUBLASLT_PATH"
echo "  CUDNN: $CUDNN_PATH"

# Create symlinks directly
if [ -n "$CUBLAS_PATH" ]; then
    ln -sf $CUBLAS_PATH /usr/lib/x86_64-linux-gnu/libcublas.so.11
    echo "Created cublas symlink to $CUBLAS_PATH"
fi

if [ -n "$CUBLASLT_PATH" ]; then
    ln -sf $CUBLASLT_PATH /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
    echo "Created cublaslt symlink to $CUBLASLT_PATH"
fi

if [ -n "$CUDNN_PATH" ]; then
    ln -sf $CUDNN_PATH /usr/lib/x86_64-linux-gnu/libcudnn.so.8
    echo "Created cudnn symlink to $CUDNN_PATH"
fi

# Now run the emergency CUDA fix script
echo "Running Emergency CUDA Fix..."
python /app/tools/emergency_cuda_fix.py

# Update LD_LIBRARY_PATH to include all possible CUDA library locations
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Verify critical libraries exist
echo "Verifying critical CUDA libraries..."
ls -la /usr/lib/x86_64-linux-gnu/libcublas.so.11 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 || echo "WARNING: Libraries not found"

# Create direct symlinks as an absolute last resort
echo "Creating direct symlinks as last resort..."
if [ -f "/usr/local/cuda/lib64/libcublas.so" ]; then
    ln -sf /usr/local/cuda/lib64/libcublas.so /usr/lib/x86_64-linux-gnu/libcublas.so.11
    echo "Created cublas direct symlink"
fi

if [ -f "/usr/local/cuda/lib64/libcublasLt.so" ]; then
    ln -sf /usr/local/cuda/lib64/libcublasLt.so /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
    echo "Created cublaslt direct symlink"
fi

# Verify ONNX Runtime can see CUDA
echo "Verifying ONNX Runtime CUDA support..."
python -c "import onnxruntime as ort; print('ONNX Runtime providers:', ort.get_available_providers())"

# Start the application
echo "Starting application..."
if [ "$ENVIRONMENT" = "development" ]; then
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload
else
    uvicorn main:app --host 0.0.0.0 --port 8001
fi
