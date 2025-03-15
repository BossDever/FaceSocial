#!/usr/bin/env python3
"""
Manual CUDA Library Fix for ONNX Runtime

This script creates direct symbolic links from CUDA 12 to CUDA 11 libraries.
Run with elevated privileges in the container:
docker exec -it ai-services-face-recognition-1 python /app/tools/manual_cuda_fix.py
"""

import os
import subprocess
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("manual_cuda_fix")

def find_latest_library(pattern):
    """Find the latest version of a library matching the pattern"""
    candidates = []
    
    # Common paths to search
    search_paths = [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/targets/x86_64-linux/lib"
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            full_pattern = os.path.join(path, pattern)
            candidates.extend(glob.glob(full_pattern))
    
    if not candidates:
        return None
    
    # Sort by version, with highest last
    candidates.sort()
    return candidates[-1]

def create_symlink(source, target):
    """Create a symbolic link, removing target if it exists"""
    if not os.path.exists(source):
        logger.error(f"Source file not found: {source}")
        return False
    
    target_dir = os.path.dirname(target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    if os.path.exists(target):
        os.remove(target)
    
    os.symlink(source, target)
    logger.info(f"Created symlink: {target} → {source}")
    return True

def main():
    logger.info("Starting manual CUDA fix...")
    
    # Make sure the target directory exists
    target_dir = "/usr/lib/x86_64-linux-gnu"
    os.makedirs(target_dir, exist_ok=True)
    
    # Required target libraries for ONNX Runtime
    required_links = {
        "libcublas.so.11": "libcublas.so*",
        "libcublasLt.so.11": "libcublasLt.so*",
        "libcudnn.so.8": "libcudnn.so*",
        "libcufft.so.10": "libcufft.so*",
        "libcurand.so.10": "libcurand.so*"
    }
    
    # Create direct symbolic links for each required library
    for target_name, source_pattern in required_links.items():
        target_path = os.path.join(target_dir, target_name)
        
        # Find the source library
        source_path = find_latest_library(source_pattern)
        if source_path:
            create_symlink(source_path, target_path)
        else:
            logger.error(f"Could not find source for {source_pattern}")
    
    # Verify if ONNX Runtime can now use CUDA
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        cuda_available = "CUDAExecutionProvider" in providers
        
        if cuda_available:
            logger.info(f"✅ ONNX Runtime CUDA support is available: {providers}")
            
            # Create a simple ONNX model to test
            import numpy as np
            from onnx import helper, TensorProto, save_model
            
            # Create a simple model
            input_name = "input"
            output_name = "output"
            
            # Create a simple model
            input_info = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 3, 10, 10])
            output_info = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 3, 10, 10])
            
            node = helper.make_node(
                'Identity',
                inputs=[input_name],
                outputs=[output_name]
            )
            
            graph = helper.make_graph([node], 'test_graph', [input_info], [output_info])
            model = helper.make_model(graph)
            
            # Save the model to a temporary file
            model_path = "/tmp/test.onnx"
            with open(model_path, 'wb') as f:
                f.write(model.SerializeToString())
            
            # Try to run with CUDA
            session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
            
            # Check which provider was selected
            used_provider = session.get_providers()[0]
            
            if "CUDA" in used_provider:
                logger.info(f"✅ CUDA TEST PASSED! Using: {used_provider}")
            else:
                logger.warning(f"⚠️ Test model using: {used_provider} (not CUDA)")
            
            # Clean up
            os.remove(model_path)
        else:
            logger.error(f"❌ ONNX Runtime CUDA support is NOT available: {providers}")
    except Exception as e:
        logger.error(f"Error testing ONNX Runtime: {e}")

if __name__ == "__main__":
    main()
