import os
import sys
import onnxruntime as ort
from typing import List, Optional

def get_best_available_provider() -> List[str]:
    """
    Returns the best available ONNX runtime execution provider, with fallbacks.
    
    Returns:
        List of provider names in order of preference
    """
    available = ort.get_available_providers()
    print(f"Available ONNX Runtime providers: {available}")
    
    # Check for CUDA-related libraries
    cuda_available = _check_cuda_libraries()
    if not cuda_available:
        print("CUDA libraries missing, forcing CPU provider only")
        return ["CPUExecutionProvider"]
    
    # Preferred provider order
    if 'CUDAExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        return ['CPUExecutionProvider']

def _check_cuda_libraries() -> bool:
    """
    Check if necessary CUDA libraries are available.
    
    Returns:
        bool: True if required libraries are found
    """
    required_libs = ['libcublas.so.11', 'libcublasLt.so.11', 'libcudnn.so.8']
    missing_libs = []
    
    # Library search paths
    search_paths = [
        '/usr/lib/x86_64-linux-gnu/',
        '/usr/local/cuda/lib64/',
        '/usr/local/cuda-11/lib64/'
    ]
    
    # Check environment variable paths
    if 'LD_LIBRARY_PATH' in os.environ:
        lib_paths = os.environ['LD_LIBRARY_PATH'].split(':')
        search_paths.extend(lib_paths)
    
    # Check if libraries exist
    for lib in required_libs:
        found = False
        for path in search_paths:
            full_path = os.path.join(path, lib)
            if os.path.exists(full_path):
                print(f"Found {lib} at {full_path}")
                found = True
                break
        
        if not found:
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"Missing CUDA libraries: {missing_libs}")
        return False
    else:
        print("All required CUDA libraries found")
        return True

def create_onnx_session(model_path: str) -> ort.InferenceSession:
    """
    Create an ONNX runtime session with the best available provider
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        ort.InferenceSession: ONNX runtime session
    """
    try:
        # First try with all providers
        providers = get_best_available_provider()
        print(f"Creating ONNX session for {model_path} with providers: {providers}")
        session = ort.InferenceSession(model_path, providers=providers)
        return session
    except Exception as e:
        print(f"Failed to create ONNX session: {e}")
        # Fall back to CPU only
        print("Falling back to CPU provider only")
        try:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            return session
        except Exception as e2:
            print(f"Failed to create CPU session: {e2}")
            raise RuntimeError(f"Could not load ONNX model: {e2}")
