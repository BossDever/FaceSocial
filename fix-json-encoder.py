#!/usr/bin/env python3
import os
import re

# โฟลเดอร์ของ services
services_dirs = [
    'services/face-detection',
    'services/face-recognition',
    'services/liveness',
    'services/deepfake'
]

# รูปแบบที่ต้องการแก้ไข
fixes = [
    # 1. แก้ import: เพิ่ม import json (ถ้ายังไม่มี)
    (r'^from flask import (.*?)(?:json,|json)?(.*?)$', 
     r'from flask import \1\2\nimport json  # เพิ่มการ import json module มาตรฐาน'),
    
    # 2. แก้ไขการใช้ flask.json.JSONEncoder เป็น json.JSONEncoder 
    (r'class NumpyEncoder\(flask\.json\.JSONEncoder\):', 
     r'class NumpyEncoder(json.JSONEncoder):'),
    
    # 3. แก้ไขการกำหนด encoder
    (r'app\.json\.encoder = NumpyEncoder', 
     r'app.json_encoder = NumpyEncoder  # เปลี่ยนจาก app.json.encoder เป็น app.json_encoder')
]

def fix_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # ตรวจสอบและแก้ไขตามรูปแบบที่กำหนด
        original_content = content
        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content)
        
        # บันทึกไฟล์ถ้ามีการแก้ไข
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ แก้ไขไฟล์ {file_path} สำเร็จ")
            return True
        else:
            print(f"ไม่จำเป็นต้องแก้ไขไฟล์ {file_path}")
            return False
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการแก้ไขไฟล์ {file_path}: {str(e)}")
        return False

# แก้ไขทุก service
for service_dir in services_dirs:
    app_py_path = os.path.join(service_dir, 'app.py')
    if os.path.exists(app_py_path):
        fix_file(app_py_path)