#!/bin/bash

echo "Starting application with CUDA libraries setup"

# รันสคริปต์แก้ไข CUDA libraries ก่อน
if [ -f "/app/tools/cuda_libraries_fix.sh" ]; then
    echo "Running CUDA libraries fix script..."
    bash /app/tools/cuda_libraries_fix.sh
else
    echo "CUDA fix script not found, will attempt manual fixes"
    
    # สร้าง symbolic links ที่จำเป็นตรงนี้
    mkdir -p /usr/lib/x86_64-linux-gnu
    
    # ค้นหา libraries
    for CUDA_DIR in "/usr/local/cuda-12.4" "/usr/local/cuda" "/usr"; do
        if [ -f "${CUDA_DIR}/targets/x86_64-linux/lib/libcublas.so.12" ]; then
            ln -sf ${CUDA_DIR}/targets/x86_64-linux/lib/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.11
            echo "Created symlink: libcublas.so.11 -> ${CUDA_DIR}/targets/x86_64-linux/lib/libcublas.so.12"
            break
        elif [ -f "${CUDA_DIR}/lib64/libcublas.so" ]; then
            ln -sf ${CUDA_DIR}/lib64/libcublas.so /usr/lib/x86_64-linux-gnu/libcublas.so.11
            echo "Created symlink: libcublas.so.11 -> ${CUDA_DIR}/lib64/libcublas.so"
            break
        fi
    done
    
    # ทำเช่นเดียวกับ libcublasLt
    for CUDA_DIR in "/usr/local/cuda-12.4" "/usr/local/cuda" "/usr"; do
        if [ -f "${CUDA_DIR}/targets/x86_64-linux/lib/libcublasLt.so.12" ]; then
            ln -sf ${CUDA_DIR}/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
            echo "Created symlink: libcublasLt.so.11 -> ${CUDA_DIR}/targets/x86_64-linux/lib/libcublasLt.so.12"
            break
        elif [ -f "${CUDA_DIR}/lib64/libcublasLt.so" ]; then
            ln -sf ${CUDA_DIR}/lib64/libcublasLt.so /usr/lib/x86_64-linux-gnu/libcublasLt.so.11
            echo "Created symlink: libcublasLt.so.11 -> ${CUDA_DIR}/lib64/libcublasLt.so"
            break
        fi
    done
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
