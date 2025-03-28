#!/bin/bash

# สร้างโฟลเดอร์สำหรับ ELA models
mkdir -p models/deepfake/ela_models

# ถ้าคุณมีไฟล์ ELA models จากเอกสาร deepfake แล้ว
# ให้คัดลอกไฟล์เหล่านั้นไปยังโฟลเดอร์ที่สร้างไว้
echo "กรุณาคัดลอกไฟล์ ELA models ไปยังโฟลเดอร์ models/deepfake/ela_models"
echo "ตัวอย่างคำสั่ง:"
echo "  cp -r /path/to/your/ela_models/*.pth models/deepfake/ela_models/"

# ไฟล์ที่ต้องมีตามเอกสาร
echo "
ไฟล์โมเดลที่ต้องมี:
1. ela_model_fold0.pth
2. ela_model_fold1.pth  
3. ela_model_fold2.pth
4. ela_model_fold3.pth
5. ela_model_fold4.pth
6. ela_stacking_ensemble_model.pth
"
