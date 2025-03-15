#!/bin/bash

echo "Starting application with ONNX Runtime 1.17.0+ (native CUDA 12.x support)"

# รันสคริปต์แก้ปัญหา unmount ถ้ามีอยู่
if [ -f "/app/tools/unmount_fix.py" ]; then
    echo "Running unmount fix script..."
    python /app/tools/unmount_fix.py
fi

# แก้ปัญหา directory symlink
if [ -d "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11" ] || [ -d "/usr/lib/x86_64-linux-gnu/libcublas.so.11" ]; then
    echo "WARNING: พบ symlink ที่ชี้ไปยัง directory กำลังแก้ไข..."
    
    # แก้ไข libcublasLt.so.11
    if [ -d "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11" ]; then
        # ใช้ force unmount ถ้าจำเป็น
        mountpoint -q /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 && umount -f /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
        rm -rf /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
        ln -sf /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
        echo "Fixed: libcublasLt.so.11"
    fi
    
    # แก้ไข libcublas.so.11
    if [ -d "/usr/lib/x86_64-linux-gnu/libcublas.so.11" ]; then
        # ใช้ force unmount ถ้าจำเป็น
        mountpoint -q /usr/lib/x86_64-linux-gnu/libcublas.so.11 && umount -f /usr/lib/x86_64-linux-gnu/libcublas.so.11
        rm -rf /usr/lib/x86_64-linux-gnu/libcublas.so.11
        ln -sf /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.11
        echo "Fixed: libcublas.so.11"
    fi
fi

# Run the Python symlink fix script if available
if [ -f "/app/tools/fix_symlink_directory.py" ]; then
    echo "Running symlink fix script..."
    python /app/tools/fix_symlink_directory.py
fi

# ตรวจสอบว่า ONNX Runtime สามารถเห็น CUDA
echo "Verifying ONNX Runtime CUDA support..."
python -c "import onnxruntime as ort; print('ONNX Runtime providers:', ort.get_available_providers())"

# เริ่มแอปพลิเคชัน
echo "Starting application..."
if [ "$ENVIRONMENT" = "development" ]; then
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload
else
    uvicorn main:app --host 0.0.0.0 --port 8001
fi
