#!/usr/bin/env python3
"""
สคริปต์สำหรับทดสอบความเข้ากันของ PyTorch, ONNX Runtime และ TensorRT
"""

import torch
import numpy as np
import sys
import os
import time

def test_pytorch_to_onnx_conversion():
    """ทดสอบการแปลงโมเดล PyTorch เป็น ONNX"""
    print("\n===== PyTorch to ONNX Conversion Test =====")
    
    try:
        import torch.onnx
        
        # สร้างโมเดล PyTorch ง่ายๆ
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = torch.nn.Linear(100, 50)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(50, 10)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
        
        # สร้าง instance ของโมเดล
        model = SimpleModel()
        model.eval()
        
        # สร้าง input dummy
        dummy_input = torch.randn(1, 100)
        
        # แปลงเป็น ONNX
        output_path = "test_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )
        
        # ตรวจสอบว่าไฟล์ ONNX ถูกสร้างขึ้นหรือไม่
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"ONNX model created successfully: {output_path} ({file_size} bytes)")
            
            # ลบไฟล์หลังจากทดสอบเสร็จ
            os.remove(output_path)
            print("ONNX model file removed")
            
            print("PyTorch to ONNX conversion test passed!")
            return True
        else:
            print(f"Failed to create ONNX model: {output_path}")
            return False
    
    except Exception as e:
        print(f"Error during PyTorch to ONNX conversion test: {e}")
        return False

def test_onnx_runtime():
    """ทดสอบการทำงานของ ONNX Runtime"""
    print("\n===== ONNX Runtime Test =====")
    
    try:
        import onnxruntime as ort
        
        print(f"ONNX Runtime version: {ort.__version__}")
        
        # ตรวจสอบ available providers
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        # ตรวจสอบว่ามี CUDA provider หรือไม่
        has_cuda = 'CUDAExecutionProvider' in providers
        print(f"CUDA Execution Provider available: {has_cuda}")
        
        if has_cuda:
            # ทดสอบสร้าง ONNX Runtime session ด้วย CUDA provider
            print("Creating test ONNX Runtime session with CUDA provider...")
            
            # สร้างโมเดล PyTorch ง่ายๆ และแปลงเป็น ONNX
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super(SimpleModel, self).__init__()
                    self.fc = torch.nn.Linear(100, 10)
                
                def forward(self, x):
                    return self.fc(x)
            
            model = SimpleModel()
            model.eval()
            
            dummy_input = torch.randn(1, 100)
            output_path = "test_model_ort.onnx"
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=12,
                input_names=['input'],
                output_names=['output']
            )
            
            # สร้าง ONNX Runtime session
            session = ort.InferenceSession(
                output_path,
                providers=['CUDAExecutionProvider']
            )
            
            # ตรวจสอบว่าใช้ CUDA provider หรือไม่
            used_provider = session.get_providers()[0]
            print(f"Active provider: {used_provider}")
            
            # รันการ inference
            print("Running inference with ONNX Runtime...")
            ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outputs = session.run(None, ort_inputs)
            
            print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")
            
            # ลบไฟล์หลังจากทดสอบเสร็จ
            os.remove(output_path)
            print("ONNX model file removed")
            
            print("ONNX Runtime with CUDA test passed!")
            return True
        else:
            print("CUDA Execution Provider is not available for ONNX Runtime")
            return False
    
    except Exception as e:
        print(f"Error during ONNX Runtime test: {e}")
        return False

def test_tensorrt():
    """ทดสอบการทำงานของ TensorRT ผ่าน ONNX Runtime"""
    print("\n===== TensorRT Test =====")
    
    try:
        import onnxruntime as ort
        
        # ตรวจสอบว่ามี TensorRT provider หรือไม่
        providers = ort.get_available_providers()
        has_tensorrt = 'TensorrtExecutionProvider' in providers
        
        print(f"TensorRT Execution Provider available: {has_tensorrt}")
        
        if has_tensorrt:
            print("TensorRT is available through ONNX Runtime")
            
            # เพิ่มการทดสอบเพิ่มเติมสำหรับ TensorRT ได้ที่นี่
            # (การทดสอบแบบสมบูรณ์ต้องการการแปลงโมเดล ONNX เป็น TensorRT engine
            # ซึ่งอาจใช้เวลานานและซับซ้อน)
            
            print("TensorRT basic availability test passed!")
            return True
        else:
            print("TensorRT Execution Provider is not available for ONNX Runtime")
            return False
    
    except Exception as e:
        print(f"Error during TensorRT test: {e}")
        return False

def main():
    """ฟังก์ชันหลักสำหรับรัน test ทั้งหมด"""
    print("Starting GPU compatibility tests...")
    
    results = {
        "PyTorch to ONNX": test_pytorch_to_onnx_conversion(),
        "ONNX Runtime": test_onnx_runtime(),
        "TensorRT": test_tensorrt()
    }
    
    print("\n===== Summary =====")
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name} test: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\nAll compatibility tests passed!")
    else:
        print("\nSome compatibility tests failed. Please check the logs above.")

if __name__ == "__main__":
    main()