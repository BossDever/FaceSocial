#!/bin/bash

# เอาส่วนการสร้าง symlinks ออกเพราะไม่จำเป็นแล้วกับ ONNX Runtime 1.17.0+
echo "Starting application with ONNX Runtime 1.17.0+ (native CUDA 12.x support)"

# Verify ONNX Runtime can see CUDA
echo "Verifying ONNX Runtime CUDA support..."
python -c "import onnxruntime as ort; print('ONNX Runtime providers:', ort.get_available_providers())"

# Start the application
echo "Starting application..."
if [ "$ENVIRONMENT" = "development" ]; then
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload
else
    uvicorn main:app --host 0.0.0.0 --port 8001
fi
