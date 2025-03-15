#!/usr/bin/env python3
"""
CUDA Library Setup Script for ONNX Runtime GPU Acceleration

This script creates the necessary symbolic links for CUDA libraries required by ONNX Runtime.
It's designed to run at container startup to ensure proper GPU acceleration.
"""

import os
import glob
import shutil
import subprocess
from typing import List, Dict, Tuple, Optional

def find_cuda_lib(lib_name: str, search_dirs: List[str] = None) -> Optional[str]:
    """
    Find a CUDA library file in common directories.
    
    Args:
        lib_name: Base name of the library (e.g., 'libcublas.so')
        search_dirs: List of directories to search in
    
    Returns:
        Path to the found library or None if not found
    """
    if search_dirs is None:
        search_dirs = [
            '/usr/lib/x86_64-linux-gnu',
            '/usr/local/cuda/lib64',
            '/usr/local/cuda/targets/x86_64-linux/lib'
        ]
        
        # Add CUDA version-specific directories
        for cuda_dir in glob.glob('/usr/local/cuda-*'):
            search_dirs.append(f"{cuda_dir}/lib64")
            search_dirs.append(f"{cuda_dir}/targets/x86_64-linux/lib")
    
    # First try exact name
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        
        full_path = os.path.join(directory, lib_name)
        if os.path.exists(full_path):
            return full_path
    
    # Try with wildcard for version numbers
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        
        pattern = os.path.join(directory, lib_name.split('.')[0] + '.so.*')
        matches = glob.glob(pattern)
        if matches:
            # Return the highest version
            return sorted(matches)[-1]
    
    # Try base lib without version
    base_lib = lib_name.split('.')[0] + '.so'
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        
        full_path = os.path.join(directory, base_lib)
        if os.path.exists(full_path):
            return full_path
    
    return None

def create_symlink(source: str, target: str) -> bool:
    """
    Create a symbolic link from source to target.
    
    Args:
        source: Source file path
        target: Target link path
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(source):
        print(f"Source file does not exist: {source}")
        return False
    
    try:
        # Create parent directory if it doesn't exist
        target_dir = os.path.dirname(target)
        os.makedirs(target_dir, exist_ok=True)
        
        # Remove existing link/file if it exists
        if os.path.exists(target):
            os.remove(target)
        
        # Create symlink
        os.symlink(source, target)
        print(f"Created symlink: {source} -> {target}")
        return True
    except Exception as e:
        print(f"Error creating symlink: {str(e)}")
        return False

def check_onnxruntime_cuda() -> bool:
    """
    Check if ONNX Runtime can use CUDA
    
    Returns:
        True if ONNX Runtime can use CUDA, False otherwise
    """
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        return 'CUDAExecutionProvider' in providers
    except Exception as e:
        print(f"Error checking ONNX Runtime providers: {str(e)}")
        return False

def main():
    """Main function to set up CUDA libraries"""
    print("CUDA Library Setup for ONNX Runtime")
    
    # Target directory for symlinks
    target_dir = "/usr/lib/x86_64-linux-gnu"
    
    # ONNX Runtime CUDA libraries needed for version 11
    cuda_libs = {
        "libcublas.so.11": ["libcublas.so", "libcublas.so.*"],
        "libcublasLt.so.11": ["libcublasLt.so", "libcublasLt.so.*"],
        "libcudnn.so.8": ["libcudnn.so", "libcudnn.so.*"]
    }
    
    # Find library files and create symlinks
    for target_lib, alternative_names in cuda_libs.items():
        print(f"\nSetting up {target_lib}:")
        
        # First check if target already exists
        target_path = os.path.join(target_dir, target_lib)
        if os.path.exists(target_path):
            print(f"Target already exists: {target_path}")
            continue
        
        # Look for the source library
        source_path = None
        
        # Try the exact name first
        source_path = find_cuda_lib(target_lib)
        
        # Try alternative names if exact match not found
        if not source_path:
            for alt_name in alternative_names:
                source_path = find_cuda_lib(alt_name)
                if source_path:
                    break
        
        if source_path:
            create_symlink(source_path, target_path)
        else:
            print(f"Could not find source for {target_lib}")
            
            # Try to locate any relevant libraries for debugging
            print("Looking for relevant libraries:")
            lib_prefix = target_lib.split('.')[0]
            result = subprocess.run(f"find /usr -name '{lib_prefix}*' | sort", 
                                    shell=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            else:
                print("No relevant libraries found")
    
    # Check if ONNX Runtime can use CUDA
    print("\nChecking ONNX Runtime CUDA support:")
    if check_onnxruntime_cuda():
        print("✅ ONNX Runtime CUDA support is available")
    else:
        print("❌ ONNX Runtime CUDA support is NOT available")
    
    print("\nCUDA library setup complete")

if __name__ == "__main__":
    main()
