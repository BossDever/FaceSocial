#!/usr/bin/env python3
"""
CUDA Library Manager for FaceSocial AI Services

This script provides comprehensive CUDA library management for ONNX Runtime GPU acceleration.
It detects available libraries, creates necessary symbolic links, and verifies the setup.

Run at container startup to ensure proper GPU acceleration for all models.
"""

import os
import glob
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Set
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("cuda_manager")

# CUDA library requirements (ONNX Runtime 1.16.0 needs CUDA 11.x)
REQUIRED_LIBS = {
    # Format: target_name: [source_alternatives]
    "libcublas.so.11": ["libcublas.so.12", "libcublas.so"],
    "libcublasLt.so.11": ["libcublasLt.so.12", "libcublasLt.so"],
    "libcudnn.so.8": ["libcudnn.so.8.*", "libcudnn.so"],
    "libcufft.so.10": ["libcufft.so", "libcufft.so.11", "libcufft.so.*"],
    "libcurand.so.10": ["libcurand.so", "libcurand.so.*"],
    "libcusolver.so.11": ["libcusolver.so", "libcusolver.so.*"],
    "libcusparse.so.11": ["libcusparse.so", "libcusparse.so.*"],
}

TARGET_DIR = "/usr/lib/x86_64-linux-gnu"

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
    
    # Add direct search for version 12 libraries when looking for version 11
    if "11" in pattern and "*" not in pattern:
        v12_pattern = pattern.replace("11", "12")
        for search_path in search_paths:
            if not os.path.isdir(search_path):
                continue
            
            # Try direct version 12 file
            full_path = os.path.join(search_path, v12_pattern)
            if os.path.exists(full_path):
                logger.info(f"Found version 12 library for {pattern}: {full_path}")
                results.append(full_path)
    
    return results

def create_symlink(source: str, target: str, force: bool = True) -> bool:
    """Create a symbolic link from source to target."""
    try:
        # Check if target already exists
        if os.path.exists(target):
            if os.path.islink(target):
                existing_link_target = os.readlink(target)
                if existing_link_target == source:
                    logger.info(f"Symlink already exists: {target} -> {source}")
                    return True
                else:
                    logger.info(f"Replacing existing symlink: {target} -> {existing_link_target} with {source}")
                    if force:
                        os.remove(target)
                    else:
                        logger.warning(f"Not replacing existing symlink (force=False)")
                        return False
            else:
                if force:
                    logger.warning(f"Removing existing file: {target}")
                    os.remove(target)
                else:
                    logger.warning(f"Not replacing existing file (force=False)")
                    return False
        
        # Create symlink
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.symlink(source, target)
        logger.info(f"Created symlink: {target} -> {source}")
        return True
    except Exception as e:
        logger.error(f"Error creating symlink {target} -> {source}: {e}")
        return False

def fix_cuda_libraries() -> Dict[str, bool]:
    """
    Fix CUDA libraries by creating necessary symbolic links.
    
    Returns:
        Dict mapping library names to success status
    """
    results = {}
    
    # Use more aggressive search for specific libraries needed by ONNX Runtime
    critical_libs = {
        "libcublas.so.11": ["/usr/local/cuda/lib64/libcublas.so", "/usr/lib/x86_64-linux-gnu/libcublas.so*"],
        "libcublasLt.so.11": ["/usr/local/cuda/lib64/libcublasLt.so", "/usr/lib/x86_64-linux-gnu/libcublasLt.so*"]
    }
    
    # Process critical libraries first with special handling
    for target_lib, search_patterns in critical_libs.items():
        target_path = os.path.join(TARGET_DIR, target_lib)
        
        # Skip if target already exists
        if os.path.exists(target_path):
            logger.info(f"✅ {target_lib} already exists")
            results[target_lib] = True
            continue
        
        # Try to find source library using specific patterns
        source_lib = None
        for pattern in search_patterns:
            # Use glob for wildcard patterns
            if "*" in pattern:
                matches = glob.glob(pattern)
                if matches:
                    source_lib = matches[0]
                    break
            # Direct file check
            elif os.path.exists(pattern):
                source_lib = pattern
                break
        
        if source_lib:
            # Create symbolic link
            success = create_symlink(source_lib, target_path)
            results[target_lib] = success
        else:
            # Special fallback for CUDA 12 libraries
            v12_lib = target_lib.replace("11", "12")
            v12_path = os.path.join(TARGET_DIR, v12_lib)
            if os.path.exists(v12_path):
                logger.info(f"Creating link from version 12 library: {v12_lib} -> {target_lib}")
                success = create_symlink(v12_path, target_path)
                results[target_lib] = success
            else:
                # Last resort - find any libcublas/libcublasLt library
                lib_prefix = target_lib.split('.')[0]
                cmd = f"find /usr -name '{lib_prefix}*' | sort"
                process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if process.stdout:
                    potential_libs = process.stdout.strip().split('\n')
                    if potential_libs:
                        logger.info(f"Last resort - linking: {potential_libs[0]} -> {target_lib}")
                        success = create_symlink(potential_libs[0], target_path)
                        results[target_lib] = success
                    else:
                        results[target_lib] = False
                else:
                    results[target_lib] = False
    
    # Process remaining standard libraries
    for target_lib, source_alternatives in REQUIRED_LIBS.items():
        # Skip already processed libraries
        if target_lib in results:
            continue
        
        target_path = os.path.join(TARGET_DIR, target_lib)
        
        # Skip if target already exists and is not a symlink
        if os.path.exists(target_path) and not os.path.islink(target_path):
            logger.info(f"✅ {target_lib} already exists as real file")
            results[target_lib] = True
            continue
        
        # Try to find a suitable source library
        source_lib = None
        for pattern in source_alternatives:
            matches = find_library_files(pattern)
            if matches:
                # Use the first match
                source_lib = matches[0]
                break
        
        if source_lib:
            # Create symbolic link
            success = create_symlink(source_lib, target_path)
            results[target_lib] = success
            if success:
                logger.info(f"✅ Created link for {target_lib} -> {source_lib}")
            else:
                logger.error(f"❌ Failed to create link for {target_lib}")
        else:
            # If no source found, look for existing link that might work
            logger.warning(f"❌ Could not find source file for {target_lib}")
            results[target_lib] = False
    
    return results

def download_cuda_libraries() -> bool:
    """
    Attempt to download and install missing CUDA libraries.
    This is a more aggressive approach if creating symlinks fails.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Determine which CUDA libraries are missing
        missing_libs = []
        for lib in REQUIRED_LIBS.keys():
            target_path = os.path.join(TARGET_DIR, lib)
            if not os.path.exists(target_path):
                missing_libs.append(lib)
        
        if not missing_libs:
            logger.info("No missing CUDA libraries to download")
            return True
        
        logger.info(f"Attempting to install missing CUDA libraries: {missing_libs}")
        
        # First try to install using apt
        try:
            apt_packages = [
                "libcublas-11-*", 
                "libcublaslt-11-*", 
                "libcudnn8"
            ]
            
            cmd = ["apt-get", "update", "-y"]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            cmd = ["apt-get", "install", "-y"] + apt_packages
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Verify if installation helped
            fixed = fix_cuda_libraries()
            if all(fixed.values()):
                logger.info("All CUDA libraries now available after apt install")
                return True
        except Exception as e:
            logger.warning(f"apt-get installation failed: {e}")
        
        # If apt installation didn't work, try direct download
        tmp_dir = "/tmp/cuda_libs"
        os.makedirs(tmp_dir, exist_ok=True)
        
        # This is a simplified example - in production, you should verify downloads
        # and use more secure download methods
        cuda_deb_url = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
        cuda_debs = [
            "libcublas-11-8_11.11.3.6-1_amd64.deb",
            "libcublas-dev-11-8_11.11.3.6-1_amd64.deb",
            "libcudnn8_8.9.2.26-1+cuda11.8_amd64.deb"
        ]
        
        for deb in cuda_debs:
            try:
                target_file = os.path.join(tmp_dir, deb)
                url = f"{cuda_deb_url}{deb}"
                
                logger.info(f"Downloading {url}")
                cmd = ["wget", "-O", target_file, url]
                subprocess.run(cmd, check=True)
                
                logger.info(f"Installing {target_file}")
                cmd = ["dpkg", "-i", target_file]
                subprocess.run(cmd, check=True)
            except Exception as e:
                logger.warning(f"Failed to download/install {deb}: {e}")
        
        # Fix library links again after download attempt
        fixed = fix_cuda_libraries()
        return all(fixed.values())
    
    except Exception as e:
        logger.error(f"Failed to download CUDA libraries: {e}")
        return False

def add_ld_library_paths():
    """Add all potential CUDA library paths to LD_LIBRARY_PATH."""
    cuda_lib_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu"
    ]
    
    # Add specific CUDA version directories
    for cuda_dir in glob.glob("/usr/local/cuda-*"):
        cuda_lib_paths.append(f"{cuda_dir}/lib64")
        cuda_lib_paths.append(f"{cuda_dir}/targets/x86_64-linux/lib")
    
    # Get current LD_LIBRARY_PATH
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    current_paths = set(ld_library_path.split(":")) if ld_library_path else set()
    
    # Add new paths
    for path in cuda_lib_paths:
        if os.path.isdir(path) and path not in current_paths:
            current_paths.add(path)
    
    # Update environment variable
    new_ld_library_path = ":".join(filter(None, current_paths))
    os.environ["LD_LIBRARY_PATH"] = new_ld_library_path
    logger.info(f"Updated LD_LIBRARY_PATH: {new_ld_library_path}")

def verify_onnx_cuda_support() -> bool:
    """Verify if ONNX Runtime can use CUDA."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        cuda_available = "CUDAExecutionProvider" in providers
        
        if cuda_available:
            logger.info("✅ ONNX Runtime CUDA support is available")
            logger.info(f"Available providers: {providers}")
            return True
        else:
            logger.error("❌ ONNX Runtime CUDA support is NOT available")
            logger.info(f"Available providers: {providers}")
            return False
    except Exception as e:
        logger.error(f"Error checking ONNX Runtime CUDA support: {e}")
        return False

def display_cuda_libraries():
    """Display all CUDA libraries in the system."""
    logger.info("=== CUDA Libraries on System ===")
    
    # Find all CUDA libraries
    cmd = 'find /usr -name "libcuda*.so*" -o -name "libcublas*.so*" -o -name "libcudnn*.so*" | sort'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        for line in result.stdout.splitlines():
            # Check if file is a symlink
            if os.path.islink(line):
                target = os.readlink(line)
                logger.info(f"{line} -> {target}")
            else:
                logger.info(line)
    else:
        logger.warning("No CUDA libraries found")
    
    logger.info("===============================")

def main():
    """Main function to fix CUDA libraries for ONNX Runtime."""
    logger.info("Starting CUDA Library Manager for ONNX Runtime GPU Acceleration")
    
    # Step 1: Display current environment
    logger.info("=== Current Environment ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    
    # Step 2: Display CUDA libraries before fixes
    display_cuda_libraries()
    
    # Step 3: Add standard library paths to LD_LIBRARY_PATH
    add_ld_library_paths()
    
    # Step 4: Fix CUDA libraries by creating symlinks
    results = fix_cuda_libraries()
    
    # Step 5: If any libraries missing, try to download them
    if not all(results.values()):
        logger.warning("Some CUDA libraries still missing, attempting to download")
        download_cuda_libraries()
    
    # Step 6: Display CUDA libraries after fixes
    display_cuda_libraries()
    
    # Step 7: Verify ONNX Runtime CUDA support
    verify_onnx_cuda_support()
    
    logger.info("CUDA Library Manager completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
