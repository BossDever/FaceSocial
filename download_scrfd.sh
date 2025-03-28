#!/bin/bash

# สร้างโฟลเดอร์ที่จำเป็น
mkdir -p models/face-detection

# ดาวน์โหลดโมเดล SCRFD - ลองใช้ link สำรอง
wget -q -O models/face-detection/scrfd_10g_bnkps.onnx https://github.com/deepinsight/insightface/raw/master/detection/scrfd/onnx/scrfd_10g_bnkps.onnx

# ตรวจสอบขนาดไฟล์
file_size=$(stat -c %s "models/face-detection/scrfd_10g_bnkps.onnx" 2>/dev/null || echo 0)
echo "ขนาดไฟล์โมเดล SCRFD: $file_size bytes"

# หากขนาดไฟล์น้อยกว่า 1MB, ให้ใช้โมเดลสำรอง RetinaFace แทน
if [ "$file_size" -lt 1000000 ]; then
    echo "โมเดล SCRFD ดาวน์โหลดไม่สมบูรณ์ กำลังดาวน์โหลด RetinaFace แทน..."
    wget -q -O models/face-detection/retinaface_r50_v1.onnx https://github.com/deepinsight/insightface/raw/master/detection/retinaface/model/det_onnx/retinaface_r50_v1.onnx
    
    # แก้ไขโค้ดให้ใช้ RetinaFace แทน
    sed -i 's/scrfd_10g_bnkps.onnx/retinaface_r50_v1.onnx/g' services/face-detection/app.py
else
    echo "โมเดล SCRFD ดาวน์โหลดสำเร็จ"
fi
