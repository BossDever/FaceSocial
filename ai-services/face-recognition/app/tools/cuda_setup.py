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

# Define global constants
TARGET_DIR = "/usr/lib/x86_64-linux-gnu"
REQUIRED_LIBS = {
    "libcublas.so.11": ["libcublas.so.12*", "libcublas.so"],
    "libcublasLt.so.11": ["libcublasLt.so.12*", "libcublasLt.so"],
    "libcudnn.so.8": ["libcudnn.so.8.*", "libcudnn.so"]
}

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

def fix_cuda_libraries() -> Dict[str, bool]:
    """
    Fix CUDA libraries by creating necessary symbolic links.
    """
    results = {}
    for target_lib, source_alternatives in REQUIRED_LIBS.items():
        target_path = os.path.join(TARGET_DIR, target_lib)
        if os.path.exists(target_path):
            results[target_lib] = True
            continue
        for pattern in source_alternatives:
            matches = find_library_files(pattern)
            if matches:
                create_symlink(matches[0], target_path)
                results[target_lib] = True
                break
        else:
            results[target_lib] = False
    return results

def find_library_files(pattern: str) -> List[str]:
    """Find library files matching the pattern in standard directories."""
    search_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    
    # Add specific CUDA version directories
    for cuda_dir in glob.glob("/usr/local/cuda-*"):
        search_paths.append(f"{cuda_dir}/lib64")
        search_paths.append(f"{cuda_dir}/targets/x86_64-linux/lib")
    
    # Add any LD_LIBRARY_PATH directories
    if "LD_LIBRARY_PATH" in os.environ:
        for path in os.environ["LD_LIBRARY_PATH"].split(":"):
            if path and path not in search_paths:
                search_paths.append(path)
    
    # Find libraries matching pattern in search paths
    results = []
    for search_path in search_paths:
        if not os.path.isdir(search_path):
            continue
        
        if "*" in pattern:
            # Use glob for wildcards
            full_pattern = os.path.join(search_path, pattern)
            matches = glob.glob(full_pattern)
            results.extend(matches)
        else:
            # Direct file check
            full_path = os.path.join(search_path, pattern)
            if os.path.exists(full_path):
                results.append(full_path)
    
    # Special handling for version 12 libraries when looking for version 11
    if "11" in pattern and "*" not in pattern:
        v12_pattern = pattern.replace("11", "12")
        for search_path in search_paths:
            if not os.path.isdir(search_path):
                continue
            
            # Try direct version 12 file
            full_path = os.path.join(search_path, v12_pattern)
            if os.path.exists(full_path):
                print(f"Found version 12 library for {pattern}: {full_path}")
                results.append(full_path)
    
    return results

def main():
    """Main function to set up CUDA libraries"""
    print("CUDA Library Setup for ONNX Runtime")
    
    # Find library files and create symlinks with special handling for
    # critical libraries needed by ONNX Runtime GPU
    
    # Direct library checks for critical libraries
    critical_libs = {
        "libcublas.so.11": True,
        "libcublasLt.so.11": True
    }
    
    # Special direct search for specific versions in standard locations
    for lib in critical_libs:
        target_path = os.path.join(TARGET_DIR, lib)
        if os.path.exists(target_path):
            print(f"Target already exists: {target_path}")
            continue
            
        # Try direct mapping from CUDA 12 to CUDA 11
        v12_lib = lib.replace("11", "12")
        potential_sources = [
            f"/usr/local/cuda/lib64/{v12_lib}",
            f"/usr/local/cuda/lib64/{lib.split('.')[0]}.so",
            f"/usr/lib/x86_64-linux-gnu/{v12_lib}",
            f"/usr/lib/x86_64-linux-gnu/{lib.split('.')[0]}.so"
        ]
        
        for source in potential_sources:
            if os.path.exists(source):
                print(f"Creating critical symlink: {source} -> {target_path}")
                create_symlink(source, target_path)
                break
        else:
            # Find any matching library as last resort
            lib_prefix = lib.split('.')[0]
            cmd_find = f"find /usr -name '{lib_prefix}*' | grep -v '/cuda-*/' | sort"
            result = subprocess.run(cmd_find, shell=True, capture_output=True, text=True)
            if result.stdout:
                sources = result.stdout.strip().split('\n')
                if sources:
                    print(f"Creating last-resort symlink: {sources[0]} -> {target_path}")
                    create_symlink(sources[0], target_path)
    
    # Process the remaining libraries
    for target_lib, alternative_names in REQUIRED_LIBS.items():
        if target_lib in critical_libs:
            continue  # Skip already handled libraries
            
        print(f"\nSetting up {target_lib}:")
        
        # First check if target already exists
        target_path = os.path.join(TARGET_DIR, target_lib)
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
        print("\nAttempting to fix CUDA libraries with brute force approach...")
        # Create links from any available version to ensure ONNX Runtime works
        for suffix in ["so", "so.12", "so.11"]:
            src_cublas = f"/usr/local/cuda/lib64/libcublas.{suffix}"
            src_cublaslt = f"/usr/local/cuda/lib64/libcublasLt.{suffix}"
            
            if os.path.exists(src_cublas):
                create_symlink(src_cublas, f"{TARGET_DIR}/libcublas.so.11")
            
            if os.path.exists(src_cublaslt):
                create_symlink(src_cublaslt, f"{TARGET_DIR}/libcublasLt.so.11")
    
    print("\nCUDA library setup complete")

if __name__ == "__main__":
    main()
