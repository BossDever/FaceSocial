#!/bin/bash

echo "🔄 กำลังดาวน์โหลดโมเดล RetinaFace ใหม่..."

# สร้างโฟลเดอร์ (ถ้ายังไม่มี)
mkdir -p models/face-detection

# ดาวน์โหลดโมเดล RetinaFace
wget -O models/face-detection/retinaface_r50_v1.onnx https://github.com/deepinsight/insightface/raw/master/detection/retinaface/model/det_onnx/retinaface_r50_v1.onnx

# ตรวจสอบว่าดาวน์โหลดสำเร็จหรือไม่
if [ -s models/face-detection/retinaface_r50_v1.onnx ]; then
    echo "✅ ดาวน์โหลดโมเดล RetinaFace สำเร็จ"
    ls -la models/face-detection/retinaface_r50_v1.onnx
else
    echo "❌ ดาวน์โหลดโมเดล RetinaFace ไม่สำเร็จ จะลองดาวน์โหลดโมเดลสำรอง"
    
    # ลองดาวน์โหลดโมเดล SCRFD เป็นตัวสำรอง
    wget -O models/face-detection/scrfd_10g_bnkps.onnx https://github.com/deepinsight/insightface/raw/master/detection/scrfd/onnx/scrfd_10g_bnkps.onnx
    
    if [ -s models/face-detection/scrfd_10g_bnkps.onnx ]; then
        echo "✅ ดาวน์โหลดโมเดล SCRFD สำเร็จ"
        ls -la models/face-detection/scrfd_10g_bnkps.onnx
    else
        echo "❌ ดาวน์โหลดโมเดลสำรองไม่สำเร็จเช่นกัน"
    fi
fi

echo "เสร็จสิ้น. รีสตาร์ท service ด้วย: docker-compose restart face-detection"