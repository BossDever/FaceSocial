#!/bin/bash

# This script fixes symbolic links for CUDA libraries
echo "Fixing CUDA symbolic links..."

# Check if libcublasLt.so.11 is a directory
if [ -d "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11" ]; then
    echo "libcublasLt.so.11 is a directory - fixing..."
    rm -rf /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
    ln -sf /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
    echo "Fixed libcublasLt.so.11"
fi

# Check if libcublas.so.11 is a directory
if [ -d "/usr/lib/x86_64-linux-gnu/libcublas.so.11" ]; then
    echo "libcublas.so.11 is a directory - fixing..."
    rm -rf /usr/lib/x86_64-linux-gnu/libcublas.so.11
    ln -sf /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.11
    echo "Fixed libcublas.so.11"
fi

# Display results
echo "Current status of symlinks:"
ls -la /usr/lib/x86_64-linux-gnu/libcublas.so.11 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11

# Check ONNX Runtime providers
echo "ONNX Runtime providers:"
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
