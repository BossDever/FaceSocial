#!/bin/bash

echo "====== ทดสอบการติดตั้ง Docker และสภาพแวดล้อม ======"

# ตรวจสอบว่ามี Docker หรือไม่
if ! command -v docker &> /dev/null; then
    echo "❌ ไม่พบ Docker - กรุณาติดตั้ง Docker ก่อน"
    exit 1
else
    echo "✅ พบ Docker - $(docker --version)"
fi

# ตรวจสอบว่ามี Docker Compose หรือไม่
if ! command -v docker-compose &> /dev/null; then
    echo "❌ ไม่พบ Docker Compose - กรุณาติดตั้ง Docker Compose ก่อน"
    exit 1
else
    echo "✅ พบ Docker Compose - $(docker-compose --version)"
fi

# ตรวจสอบ NVIDIA Container Toolkit
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ ไม่พบคำสั่ง nvidia-smi - อาจจะไม่มี GPU หรือ driver ไม่ได้ติดตั้ง"
else
    echo "✅ พบ NVIDIA GPU - $(nvidia-smi --query-gpu=name --format=csv,noheader)"
fi

# ทดสอบ NVIDIA Docker
echo "ทดสอบ NVIDIA Docker..."
docker run --rm --gpus all nvcr.io/nvidia/tensorflow:24.05-tf2-py3 nvidia-smi || {
    echo "❌ ไม่สามารถใช้งาน GPU ใน Docker ได้"
    echo "กรุณาตรวจสอบการติดตั้ง NVIDIA Container Toolkit"
    exit 1
}
echo "✅ สามารถใช้งาน GPU ใน Docker ได้"

# ทดสอบ Docker image ที่ใช้งาน
echo "ทดสอบ Docker image ที่ใช้งาน..."
docker run --rm nvcr.io/nvidia/tensorflow:24.05-tf2-py3 python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))" || {
    echo "❌ ไม่สามารถใช้งาน TensorFlow ใน Docker ได้"
    exit 1
}
echo "✅ Docker image พร้อมใช้งาน"

echo "====== การทดสอบเสร็จสมบูรณ์ ======"
echo "พร้อมสำหรับการติดตั้งและใช้งานโปรเจค"