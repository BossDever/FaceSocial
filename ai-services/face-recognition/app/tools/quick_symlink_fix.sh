#!/bin/bash

# Quick fix for symlinks pointing to directories
echo "Checking for symlinks pointing to directories..."

# Fix libcublasLt.so.11
if [ -d "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11" ]; then
    rm -rf /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
    ln -sf /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
    echo "Fixed libcublasLt.so.11"
else
    echo "libcublasLt.so.11 is not a directory, checking if it exists..."
    ls -la /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 || echo "Not found"
fi

# Fix libcublas.so.11
if [ -d "/usr/lib/x86_64-linux-gnu/libcublas.so.11" ]; then
    rm -rf /usr/lib/x86_64-linux-gnu/libcublas.so.11
    ln -sf /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.11
    echo "Fixed libcublas.so.11"
else
    echo "libcublas.so.11 is not a directory, checking if it exists..."
    ls -la /usr/lib/x86_64-linux-gnu/libcublas.so.11 || echo "Not found"
fi

# Verify ONNX Runtime providers
echo "ONNX Runtime providers:"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
