#!/bin/bash

# Enhanced CUDA library setup for ONNX Runtime GPU acceleration
echo "Setting up CUDA libraries for ONNX Runtime..."

# Create critical symlinks directly here first
mkdir -p /usr/lib/x86_64-linux-gnu

# SPECIFIC PATHS FOR CUDA 12.4 - Using the exact paths found in your container
echo "Creating symlinks for CUDA 12.4 libraries..."
# First try target-specific paths (these are the exact paths you found)
if [ -f "/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublas.so.12" ]; then
    ln -sf /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.11
    echo "Created symlink: libcublas.so.11 -> /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublas.so.12"
fi

if [ -f "/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublasLt.so.12" ]; then
    ln -sf /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
    echo "Created symlink: libcublasLt.so.11 -> /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublasLt.so.12"
fi

# Fallbacks to standard paths if specific versions not found
if [ ! -f "/usr/lib/x86_64-linux-gnu/libcublas.so.11" ]; then
    if [ -f "/usr/lib/x86_64-linux-gnu/libcublas.so.12" ]; then
        ln -sf /usr/lib/x86_64-linux-gnu/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.11
        echo "Created fallback symlink: libcublas.so.11 -> /usr/lib/x86_64-linux-gnu/libcublas.so.12"
    elif [ -f "/usr/local/cuda/lib64/libcublas.so" ]; then
        ln -sf /usr/local/cuda/lib64/libcublas.so /usr/lib/x86_64-linux-gnu/libcublas.so.11
        echo "Created generic symlink: libcublas.so.11 -> /usr/local/cuda/lib64/libcublas.so"
    fi
fi

if [ ! -f "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11" ]; then
    if [ -f "/usr/lib/x86_64-linux-gnu/libcublasLt.so.12" ]; then
        ln -sf /usr/lib/x86_64-linux-gnu/libcublasLt.so.12 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
        echo "Created fallback symlink: libcublasLt.so.11 -> /usr/lib/x86_64-linux-gnu/libcublasLt.so.12"
    elif [ -f "/usr/local/cuda/lib64/libcublasLt.so" ]; then
        ln -sf /usr/local/cuda/lib64/libcublasLt.so /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
        echo "Created generic symlink: libcublasLt.so.11 -> /usr/local/cuda/lib64/libcublasLt.so"
    fi
fi

# Verify critical libraries exist
echo "Verifying critical CUDA libraries..."
ls -la /usr/lib/x86_64-linux-gnu/libcublas.so.11 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11

# Verify ONNX Runtime can see CUDA
echo "Verifying ONNX Runtime CUDA support..."
python -c "import onnxruntime as ort; print('ONNX Runtime providers:', ort.get_available_providers())"

# Update LD_LIBRARY_PATH to include all possible CUDA library locations
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.4/targets/x86_64-linux/lib
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Start the application
echo "Starting application..."
if [ "$ENVIRONMENT" = "development" ]; then
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload
else
    uvicorn main:app --host 0.0.0.0 --port 8001
fi
