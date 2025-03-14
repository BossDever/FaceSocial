import tensorflow as tf
import numpy as np
import time

def test_gpu():
    """
    ทดสอบการทำงานของ TensorFlow กับ GPU
    
    Returns:
        dict: ผลการทดสอบรวมถึงข้อมูล GPU และความเร็วในการคำนวณ
    """
    print("TensorFlow version:", tf.__version__)
    
    # ตรวจสอบว่ามี GPU หรือไม่
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available:", len(gpus))
    
    if len(gpus) > 0:
        for gpu in gpus:
            print("GPU Name:", gpu.name)
            try:
                print("GPU Details:", tf.config.experimental.get_device_details(gpu))
            except:
                print("ไม่สามารถดึงรายละเอียด GPU ได้")
        
        # ทดสอบการคำนวณด้วย GPU
        print("\nทดสอบประสิทธิภาพการคำนวณ matrix multiplication...")
        
        # สร้าง matrices ขนาดใหญ่
        matrix_size = 5000
        a = tf.random.normal([matrix_size, matrix_size])
        b = tf.random.normal([matrix_size, matrix_size])
        
        # วัดเวลาในการคำนวณด้วย GPU
        start_time = time.time()
        c = tf.matmul(a, b)
        # Force execution
        result = c.numpy()
        gpu_time = time.time() - start_time
        
        print(f"คำนวณ matrix multiplication ขนาด {matrix_size}x{matrix_size}")
        print(f"เวลาที่ใช้: {gpu_time:.4f} วินาที")
        
        return {
            "tf_version": tf.__version__,
            "gpu_available": True,
            "gpu_count": len(gpus),
            "matrix_size": matrix_size,
            "computation_time": gpu_time
        }
    else:
        print("ไม่พบ GPU สำหรับใช้งาน")
        return {
            "tf_version": tf.__version__,
            "gpu_available": False
        }

if __name__ == "__main__":
    results = test_gpu()
    print("\nผลการทดสอบโดยสรุป:", results)
