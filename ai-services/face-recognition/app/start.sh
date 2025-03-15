#!/bin/bash

# Enhanced CUDA library setup for ONNX Runtime GPU acceleration
echo "Setting up CUDA libraries for ONNX Runtime..."

# Function to find CUDA libraries with more specific patterns
find_library() {
    local lib_name=$1
    local patterns=("${@:2}")
    
    echo "Looking for $lib_name..."
    
    for pattern in "${patterns[@]}"; do
        echo "  Searching pattern: $pattern"
        # Use find with case-insensitive name matching
        found_libs=$(find /usr -name "$pattern" -type f 2>/dev/null)
        
        if [ -n "$found_libs" ]; then
            # Take the first result
            result=$(echo "$found_libs" | head -n 1)
            echo "  ✓ Found at: $result"
            echo "$result"
            return 0
        fi
    done
    
    # Additional specific search in NVIDIA container common locations
    for dir in "/usr/local/cuda/lib64" "/usr/local/cuda/targets/x86_64-linux/lib" "/usr/lib/x86_64-linux-gnu"; do
        if [ -d "$dir" ]; then
            echo "  Searching in $dir..."
            # List all files in directory to see what's available
            ls -la $dir | grep -i $(echo $lib_name | cut -d. -f1)
        fi
    done
    
    echo "  ✗ Not found"
    return 1
}

# Create symlink with verification
create_symlink() {
    local source=$1
    local target=$2
    
    if [ -z "$source" ]; then
        echo "✗ No source provided for symlink"
        return 1
    fi
    
    # Create parent directory if needed
    mkdir -p $(dirname "$target")
    
    # Remove existing symlink or file if it exists
    if [ -e "$target" ]; then
        echo "Removing existing $target..."
        rm -f "$target"
    fi
    
    # Create symlink
    ln -sf "$source" "$target"
    
    # Verify symlink
    if [ -L "$target" ]; then
        echo "✓ Created symlink: $source -> $target"
        return 0
    else
        echo "✗ Failed to create symlink"
        return 1
    fi
}

# Step 1: Find existing CUDA libraries (search with multiple patterns)
echo "=== Finding CUDA Libraries ==="
CUBLAS_LIB=$(find_library "libcublas.so.11" "libcublas.so.12*" "libcublas.so.*" "libcublas.so")
CUBLASLT_LIB=$(find_library "libcublasLt.so.11" "libcublasLt.so.12*" "libcublasLt.so.*" "libcublasLt.so")
CUDNN_LIB=$(find_library "libcudnn.so.8" "libcudnn.so.8*" "libcudnn.so.*" "libcudnn.so")

# Step 2: Try specific NVIDIA container paths if not found
if [ -z "$CUBLAS_LIB" ]; then
    echo "Trying specific NVIDIA container paths..."
    CUDA_VERSION=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n 1)
    echo "Latest CUDA directory: $CUDA_VERSION"
    
    # Print all relevant libraries in standard locations for debugging
    echo "Available libraries in standard locations:"
    find /usr/local/cuda*/lib64 /usr/lib/x86_64-linux-gnu -name "*cublas*" | sort
    
    # Try again with exact paths common in NVIDIA containers
    if [ -f "/usr/local/cuda/lib64/libcublas.so" ]; then
        CUBLAS_LIB="/usr/local/cuda/lib64/libcublas.so"
    elif [ -f "/usr/lib/x86_64-linux-gnu/libcublas.so" ]; then
        CUBLAS_LIB="/usr/lib/x86_64-linux-gnu/libcublas.so"
    fi
    
    if [ -f "/usr/local/cuda/lib64/libcublasLt.so" ]; then
        CUBLASLT_LIB="/usr/local/cuda/lib64/libcublasLt.so"
    elif [ -f "/usr/lib/x86_64-linux-gnu/libcublasLt.so" ]; then
        CUBLASLT_LIB="/usr/lib/x86_64-linux-gnu/libcublasLt.so"
    fi
fi

# Step 3: Create symbolic links
echo "=== Creating Symbolic Links ==="
TARGET_DIR="/usr/lib/x86_64-linux-gnu"

if [ -n "$CUBLAS_LIB" ]; then
    create_symlink "$CUBLAS_LIB" "$TARGET_DIR/libcublas.so.11"
else
    echo "✗ Could not find libcublas.so source"
    # Try to create directly with the CUDA major version
    echo "Attempting to create direct symlink to version 11"
    if [ -f "/usr/local/cuda/lib64/libcublas.so" ]; then
        ln -sf /usr/local/cuda/lib64/libcublas.so "$TARGET_DIR/libcublas.so.11"
        echo "Created direct symlink: /usr/local/cuda/lib64/libcublas.so -> $TARGET_DIR/libcublas.so.11"
    fi
fi

if [ -n "$CUBLASLT_LIB" ]; then
    create_symlink "$CUBLASLT_LIB" "$TARGET_DIR/libcublasLt.so.11"
else
    echo "✗ Could not find libcublasLt.so source"
    # Try to create directly with the CUDA major version
    echo "Attempting to create direct symlink to version 11"
    if [ -f "/usr/local/cuda/lib64/libcublasLt.so" ]; then
        ln -sf /usr/local/cuda/lib64/libcublasLt.so "$TARGET_DIR/libcublasLt.so.11"
        echo "Created direct symlink: /usr/local/cuda/lib64/libcublasLt.so -> $TARGET_DIR/libcublasLt.so.11"
    fi
fi

if [ -n "$CUDNN_LIB" ]; then
    create_symlink "$CUDNN_LIB" "$TARGET_DIR/libcudnn.so.8"
fi

# Step 4: Verify symbolic links and library loading
echo "=== Verifying Libraries ==="
for lib in libcublas.so.11 libcublasLt.so.11 libcudnn.so.8; do
    if [ -L "$TARGET_DIR/$lib" ]; then
        target=$(readlink -f "$TARGET_DIR/$lib")
        echo "• $lib -> $target"
        if [ -f "$target" ]; then
            echo "  ✓ Valid target"
        else
            echo "  ✗ Invalid target!"
        fi
    elif [ -f "$TARGET_DIR/$lib" ]; then
        echo "• $lib exists as real file ✓"
    else
        echo "• $lib not found ✗"
    fi
done

# Update LD_LIBRARY_PATH to include all possible CUDA library locations
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Start the application
echo "Starting application..."
if [ "$ENVIRONMENT" = "development" ]; then
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload;
else
    uvicorn main:app --host 0.0.0.0 --port 8001;
fi
