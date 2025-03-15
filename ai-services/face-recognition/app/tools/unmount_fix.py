#!/usr/bin/env python3
"""
เครื่องมือแก้ปัญหา unmount ไฟล์ที่เป็น mountpoint และสร้าง symlink ใหม่
"""

import os
import subprocess
import logging
import time

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("unmount_fix")

def is_mountpoint(path):
    """ตรวจสอบว่า path เป็น mountpoint หรือไม่"""
    try:
        result = subprocess.run(['mountpoint', '-q', path], check=False)
        return result.returncode == 0
    except:
        return False

def unmount_if_mounted(path):
    """Unmount path ถ้าเป็น mountpoint"""
    if not os.path.exists(path):
        return True
        
    if is_mountpoint(path):
        logger.info(f"พบ mountpoint: {path} - กำลังพยายาม unmount...")
        try:
            subprocess.run(['umount', '-f', path], check=False)
            time.sleep(1)  # รอการ unmount เสร็จสมบูรณ์
            
            if not is_mountpoint(path):
                logger.info(f"Unmount สำเร็จ: {path}")
                # ลบ directory หลังจาก unmount
                if os.path.isdir(path):
                    os.rmdir(path)
                return True
            else:
                logger.error(f"ไม่สามารถ unmount ได้: {path}")
                return False
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดระหว่าง unmount {path}: {e}")
            return False
    else:
        # เป็น directory ธรรมดา ไม่ใช่ mountpoint
        if os.path.isdir(path):
            try:
                import shutil
                shutil.rmtree(path)
                logger.info(f"ลบ directory สำเร็จ: {path}")
                return True
            except Exception as e:
                logger.error(f"ไม่สามารถลบ directory ได้ {path}: {e}")
                return False
        return True

def create_symlink(source, target):
    """สร้าง symbolic link"""
    if not os.path.exists(source):
        logger.error(f"Source file ไม่พบ: {source}")
        return False
        
    try:
        if os.path.exists(target):
            os.remove(target)
        os.symlink(source, target)
        logger.info(f"สร้าง symlink สำเร็จ: {target} -> {source}")
        return True
    except Exception as e:
        logger.error(f"ไม่สามารถสร้าง symlink ได้: {e}")
        return False

def main():
    """ฟังก์ชั่นหลัก"""
    logger.info("เริ่มต้นการแก้ไขปัญหา CUDA symlinks")
    
    # Path ที่มีปัญหา
    problem_paths = [
        "/usr/lib/x86_64-linux-gnu/libcublas.so.11",
        "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11"
    ]
    
    # แหล่งที่มาของไฟล์ CUDA 12
    source_files = {
        "/usr/lib/x86_64-linux-gnu/libcublas.so.11": "/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublas.so.12",
        "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11": "/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcublasLt.so.12"
    }
    
    for path in problem_paths:
        # Unmount และลบถ้าจำเป็น
        if unmount_if_mounted(path):
            # สร้าง symlink ใหม่
            create_symlink(source_files[path], path)
    
    # ตรวจสอบ ONNX Runtime providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        cuda_available = "CUDAExecutionProvider" in providers
        
        if cuda_available:
            logger.info(f"✅ ONNX Runtime สามารถใช้งาน CUDA ได้: {providers}")
        else:
            logger.error(f"❌ ONNX Runtime ไม่สามารถใช้งาน CUDA ได้: {providers}")
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการตรวจสอบ ONNX Runtime: {e}")

if __name__ == "__main__":
    main()
