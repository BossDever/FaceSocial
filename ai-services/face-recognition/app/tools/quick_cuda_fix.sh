#!/bin/bash

# Quick CUDA fix script for ONNX Runtime
# This script creates the necessary symbolic links based on your successful manual test

echo "Creating CUDA library symlinks for ONNX Runtime GPU acceleration..."

# Create target directory
mkdir -p /usr/lib/x86_64-linux-gnu

# CUDA 12.4 specific fix (based on the paths found in your container)
if [ -f "/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublas.so.12" ]; then
    ln -sf /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.11
    echo "Created libcublas.so.11 -> /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublas.so.12"
fi

if [ -f "/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublasLt.so.12" ]; then
    ln -sf /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
    echo "Created libcublasLt.so.11 -> /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublasLt.so.12"
fi

# Verify
echo "Verifying created symlinks:"
ls -la /usr/lib/x86_64-linux-gnu/libcublas.so.11 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 || echo "Symlinks not created"

# Check ONNX providers
echo "Available ONNX Runtime providers:"
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())" || echo "Failed to get ONNX providers"

echo "Quick fix complete. To verify GPU usage, run: python /app/tools/verify_onnx_gpu.py"
