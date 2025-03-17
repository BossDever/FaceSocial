#!/bin/bash

while true; do
    # ตรวจสอบว่ามีไฟล์ที่ถูกแก้ไขหรือไม่
    if [[ -n $(git status --porcelain) ]]; then
        echo "🔄 พบการเปลี่ยนแปลง กำลัง commit..."
        git add .
        git commit -m "Auto commit: $(date +'%Y-%m-%d %H:%M:%S')"
        git push origin main
        echo "✅ Commit และ Push สำเร็จ"
    else
        echo "⏳ ไม่มีการเปลี่ยนแปลง"
    fi
    
    # รอ 30 วินาทีก่อนทำซ้ำ
    sleep 30
done

