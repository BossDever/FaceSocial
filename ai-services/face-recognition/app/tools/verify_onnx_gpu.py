#!/usr/bin/env python3
"""
ONNX Runtime GPU Verification Script

This script creates a simple ONNX model and tests whether it runs on GPU.
"""

import os
import time
import numpy as np

def verify_gpu():
    """Verify if ONNX Runtime can use GPU acceleration"""
    try:
        import onnxruntime as ort
        
        # Check available providers
        providers = ort.get_available_providers()
        print(f"Available ONNX Runtime providers: {providers}")
        print(f"CUDA available: {'CUDAExecutionProvider' in providers}")
        
        if 'CUDAExecutionProvider' not in providers:
            print("❌ CUDA provider not available")
            return False
        
        # Create a simple test model
        try:
            from onnx import helper, TensorProto
            import onnx
            
            # Create model
            input_name = "input"
            output_name = "output"
            
            # Create a simple identity model
            node_def = helper.make_node(
                "Identity",
                inputs=[input_name],
                outputs=[output_name],
            )
            
            graph_def = helper.make_graph(
                [node_def],
                "test-model",
                [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 3, 10, 10])],
                [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 3, 10, 10])],
            )
            
            model_def = helper.make_model(graph_def, producer_name="gpu-test")
            
            # Save model to temporary file
            model_path = "/tmp/gpu_test.onnx"
            with open(model_path, "wb") as f:
                f.write(model_def.SerializeToString())
            
            print(f"Created test model at {model_path}")
            
            # Create session with CUDA provider
            session_options = ort.SessionOptions()
            cuda_provider_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            
            providers = [
                ('CUDAExecutionProvider', cuda_provider_options),
                'CPUExecutionProvider'
            ]
            
            print("Creating ONNX session with CUDA provider...")
            session = ort.InferenceSession(
                model_path, 
                providers=providers,
                sess_options=session_options
            )
            
            # Verify the provider actually being used
            used_providers = session.get_providers()
            print(f"Actual providers being used: {used_providers}")
            
            # Run inference
            test_input = np.random.rand(1, 3, 10, 10).astype(np.float32)
            
            # Warm up
            print("Warming up...")
            session.run([output_name], {input_name: test_input})
            
            # Benchmark
            print("Benchmarking...")
            iterations = 100
            start_time = time.time()
            for _ in range(iterations):
                outputs = session.run([output_name], {input_name: test_input})
            end_time = time.time()
            
            avg_time = (end_time - start_time) * 1000 / iterations  # ms
            print(f"Average inference time: {avg_time:.2f} ms")
            
            # Clean up
            os.remove(model_path)
            
            if 'CUDAExecutionProvider' == used_providers[0]:
                print("✅ ONNX Runtime is using GPU acceleration successfully!")
                return True
            else:
                print("❌ ONNX Runtime fell back to CPU!")
                return False
                
        except Exception as e:
            print(f"Error creating/running test model: {e}")
            return False
    except Exception as e:
        print(f"Error verifying GPU: {e}")
        return False

if __name__ == "__main__":
    print("Verifying ONNX Runtime GPU acceleration:")
    if verify_gpu():
        print("✅ SUCCESS: ONNX Runtime is using GPU acceleration!")
    else:
        print("❌ ERROR: ONNX Runtime is NOT using GPU acceleration!")
