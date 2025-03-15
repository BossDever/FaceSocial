#!/bin/bash

echo "Creating CUDA library symlinks for ONNX Runtime GPU acceleration"

# Create target directory
mkdir -p /usr/lib/x86_64-linux-gnu

# Find all potential source libraries
echo "Looking for CUDA libraries..."
CUDA_LIBS=$(find /usr -name "libcublas.so*" -o -name "libcublasLt.so*" -o -name "libcudnn.so*" 2>/dev/null | sort)
echo "Found libraries:"
echo "$CUDA_LIBS"

# Create direct symlinks for critical libraries
echo "Creating symlinks:"
if [ -f "/usr/local/cuda/lib64/libcublas.so" ]; then
    ln -sf /usr/local/cuda/lib64/libcublas.so /usr/lib/x86_64-linux-gnu/libcublas.so.11
    echo "Created libcublas.so.11 -> /usr/local/cuda/lib64/libcublas.so"
fi

if [ -f "/usr/local/cuda/lib64/libcublasLt.so" ]; then
    ln -sf /usr/local/cuda/lib64/libcublasLt.so /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
    echo "Created libcublasLt.so.11 -> /usr/local/cuda/lib64/libcublasLt.so"
fi

# Create symlinks from CUDA 12 to CUDA 11 (more aggressive approach)
for lib in $(find /usr/lib/x86_64-linux-gnu -name "libcublas.so.12*" 2>/dev/null); do
    ln -sf $lib /usr/lib/x86_64-linux-gnu/libcublas.so.11
    echo "Created libcublas.so.11 -> $lib"
    break
done

for lib in $(find /usr/lib/x86_64-linux-gnu -name "libcublasLt.so.12*" 2>/dev/null); do
    ln -sf $lib /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
    echo "Created libcublasLt.so.11 -> $lib"
    break
done

# Try alternative methods if the above didn't work
if [ ! -f "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11" ]; then
    echo "Direct symlinks failed, trying alternative methods..."
    
    # Use the NVIDIA library directory structure
    for dir in /usr/local/cuda*/lib64; do
        if [ -d "$dir" ]; then
            echo "Checking $dir for CUDA libraries"
            for lib in $(find $dir -name "libcublas*.so*" -o -name "libcublasLt*.so*" 2>/dev/null); do
                filename=$(basename $lib)
                if [[ $filename == libcublas.so* ]]; then
                    ln -sf $lib /usr/lib/x86_64-linux-gnu/libcublas.so.11
                    echo "Created libcublas.so.11 -> $lib"
                fi
                if [[ $filename == libcublasLt.so* ]]; then
                    ln -sf $lib /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
                    echo "Created libcublasLt.so.11 -> $lib"
                fi
            done
        fi
    done
fi

# Verify
echo "Verifying created symlinks:"
ls -la /usr/lib/x86_64-linux-gnu/libcublas.so.11 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 || echo "Symlinks not created"

# Check ONNX providers
echo "Available ONNX Runtime providers:"
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())" || echo "Failed to get ONNX providers"
