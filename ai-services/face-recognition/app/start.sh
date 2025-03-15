#!/bin/bash

# Enhanced CUDA library setup for ONNX Runtime GPU acceleration
echo "Setting up CUDA libraries for ONNX Runtime..."

# Function to find and link CUDA libraries
setup_cuda_library() {
    local target_lib=$1
    local source_pattern=$2
    local fallback_pattern=$3
    
    echo "Setting up $target_lib..."
    
    # Check if target already exists and is valid
    if [ -f "/usr/lib/x86_64-linux-gnu/$target_lib" ] && [ ! -L "/usr/lib/x86_64-linux-gnu/$target_lib" ]; then
        echo "✓ $target_lib already exists as real file"
        return 0
    fi
    
    # Find the source library using the provided pattern
    source_lib=$(find /usr -path "*$source_pattern*" -type f 2>/dev/null | head -n 1)
    
    if [ -n "$source_lib" ]; then
        echo "Found source library at: $source_lib"
        ln -sf "$source_lib" "/usr/lib/x86_64-linux-gnu/$target_lib"
        echo "✓ Created symbolic link: $source_lib -> /usr/lib/x86_64-linux-gnu/$target_lib"
        return 0
    fi
    
    # Try fallback pattern
    if [ -n "$fallback_pattern" ]; then
        fallback_lib=$(find /usr -path "*$fallback_pattern*" -type f 2>/dev/null | head -n 1)
        
        if [ -n "$fallback_lib" ]; then
            echo "Found fallback library at: $fallback_lib"
            ln -sf "$fallback_lib" "/usr/lib/x86_64-linux-gnu/$target_lib"
            echo "✓ Created symbolic link (fallback): $fallback_lib -> /usr/lib/x86_64-linux-gnu/$target_lib"
            return 0
        fi
    fi
    
    echo "✗ Failed to find a suitable library for $target_lib"
    return 1
}

# Find CUDA installation directory
CUDA_DIRS=$(find /usr -path "*/cuda*" -type d 2>/dev/null)
echo "Found CUDA directories:"
echo "$CUDA_DIRS" | sed 's/^/  /'

# Create symbolic links for required CUDA libraries
setup_cuda_library "libcublas.so.11" "libcublas.so.12" "libcublas.so"
setup_cuda_library "libcublasLt.so.11" "libcublasLt.so.12" "libcublasLt.so"
setup_cuda_library "libcudnn.so.8" "libcudnn.so.8" "libcudnn.so"

# Verify links
echo "Verifying symbolic links:"
for lib in libcublas.so.11 libcublasLt.so.11 libcudnn.so.8; do
    if [ -L "/usr/lib/x86_64-linux-gnu/$lib" ]; then
        target=$(readlink "/usr/lib/x86_64-linux-gnu/$lib")
        if [ -f "$target" ]; then
            echo "✓ $lib -> $target (VALID)"
        else
            echo "✗ $lib -> $target (INVALID TARGET)"
        fi
    elif [ -f "/usr/lib/x86_64-linux-gnu/$lib" ]; then
        echo "✓ $lib exists (REAL FILE)"
    else
        echo "✗ $lib not found"
    fi
done

# Update LD_LIBRARY_PATH to include all possible CUDA library locations
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda-11/lib64:/usr/lib/x86_64-linux-gnu

# Display LD_LIBRARY_PATH for debugging
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Start the application
echo "Starting application..."
if [ "$ENVIRONMENT" = "development" ]; then
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload;
else
    uvicorn main:app --host 0.0.0.0 --port 8001;
fi
