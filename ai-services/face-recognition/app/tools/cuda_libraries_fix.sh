#!/bin/bash
# Simple script to create necessary CUDA symbolic links for ONNX Runtime

# Create target directory
mkdir -p /usr/lib/x86_64-linux-gnu

# Find CUDA 12 libraries
CUBLAS_SOURCE=$(find /usr -name "libcublas.so.12*" 2>/dev/null | head -1)
CUBLASLT_SOURCE=$(find /usr -name "libcublasLt.so.12*" 2>/dev/null | head -1)

# Fallback to generic libraries if version 12 not found
if [ -z "$CUBLAS_SOURCE" ]; then
    CUBLAS_SOURCE=$(find /usr -name "libcublas.so" 2>/dev/null | head -1)
fi

if [ -z "$CUBLASLT_SOURCE" ]; then
    CUBLASLT_SOURCE=$(find /usr -name "libcublasLt.so" 2>/dev/null | head -1)
fi

# Create symbolic links
if [ -n "$CUBLAS_SOURCE" ]; then
    ln -sf $CUBLAS_SOURCE /usr/lib/x86_64-linux-gnu/libcublas.so.11
    echo "Created libcublas.so.11 -> $CUBLAS_SOURCE"
else
    echo "ERROR: Could not find libcublas source library"
fi

if [ -n "$CUBLASLT_SOURCE" ]; then
    ln -sf $CUBLASLT_SOURCE /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
    echo "Created libcublasLt.so.11 -> $CUBLASLT_SOURCE"
else
    echo "ERROR: Could not find libcublasLt source library"
fi

# Additional symbolic links that might be needed
CUDNN_SOURCE=$(find /usr -name "libcudnn.so*" -o -name "libcudnn.so.*" 2>/dev/null | head -1)
if [ -n "$CUDNN_SOURCE" ]; then
    ln -sf $CUDNN_SOURCE /usr/lib/x86_64-linux-gnu/libcudnn.so.8
    echo "Created libcudnn.so.8 -> $CUDNN_SOURCE"
fi

# Check library locations
ls -la /usr/lib/x86_64-linux-gnu/libcublas.so.11 || echo "libcublas.so.11 not found"
ls -la /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 || echo "libcublasLt.so.11 not found"

# Display ONNX Runtime providers
python -c "import onnxruntime as ort; print('ONNX Runtime providers:', ort.get_available_providers())" || echo "Failed to import onnxruntime"
