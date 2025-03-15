#!/bin/bash

# Check CUDA version and create appropriate symlinks
echo "Checking CUDA libraries for ONNX Runtime..."

# Find existing CUDA libraries
CUDA_LIB_DIR=$(find /usr -path "*/cuda*/lib*" -type d | head -n 1)
echo "Found CUDA libraries at: $CUDA_LIB_DIR"

# Create symbolic links from CUDA 12.x to CUDA 11.x libraries
if [ -f "$CUDA_LIB_DIR/libcublas.so.12" ]; then
  echo "Creating symlink from libcublas.so.12 to libcublas.so.11"
  ln -sf $CUDA_LIB_DIR/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.11
fi

if [ -f "$CUDA_LIB_DIR/libcublasLt.so.12" ]; then
  echo "Creating symlink from libcublasLt.so.12 to libcublasLt.so.11"
  ln -sf $CUDA_LIB_DIR/libcublasLt.so.12 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
fi

# Find latest cudnn version
CUDNN_LIB=$(find /usr/lib -name "libcudnn.so*" | sort -V | tail -n 1)
if [ -n "$CUDNN_LIB" ]; then
  echo "Creating symlink from $CUDNN_LIB to libcudnn.so.8"
  ln -sf $CUDNN_LIB /usr/lib/x86_64-linux-gnu/libcudnn.so.8
fi

# Check if symlinks were created successfully
if [ -L "/usr/lib/x86_64-linux-gnu/libcublas.so.11" ] && \
   [ -L "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11" ] && \
   [ -L "/usr/lib/x86_64-linux-gnu/libcudnn.so.8" ]; then
  echo "All CUDA symbolic links created successfully."
else
  echo "Warning: Some CUDA symbolic links could not be created."
fi

# Start the application
if [ "$ENVIRONMENT" = "development" ]; then
  uvicorn main:app --host 0.0.0.0 --port 8001 --reload;
else
  uvicorn main:app --host 0.0.0.0 --port 8001;
fi
