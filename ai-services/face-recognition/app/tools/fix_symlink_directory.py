#!/usr/bin/env python3
"""
Script to fix symbolic link issues causing "cannot read file data: Is a directory" errors
"""

import os
import glob
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("symlink_fix")

def find_real_library(name_pattern):
    """Find a real library file matching the pattern"""
    # Search in common CUDA locations
    search_paths = [
        "/usr/local/cuda-12.4/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu"
    ]
    
    for path in search_paths:
        if not os.path.exists(path):
            continue
            
        # Look for exact file with version number
        for item in os.listdir(path):
            if name_pattern in item and os.path.isfile(os.path.join(path, item)):
                return os.path.join(path, item)
    
    return None

def fix_broken_symlinks():
    """Fix incorrect symlinks that point to directories instead of files"""
    target_dir = "/usr/lib/x86_64-linux-gnu"
    problem_links = ["libcublas.so.11", "libcublasLt.so.11"]
    
    for link_name in problem_links:
        link_path = os.path.join(target_dir, link_name)
        
        # Check if the link exists
        if os.path.exists(link_path):
            # Check if it's a symlink pointing to a directory
            if os.path.islink(link_path):
                target = os.readlink(link_path)
                target_path = os.path.join(os.path.dirname(link_path), target) if not os.path.isabs(target) else target
                
                if os.path.isdir(target_path):
                    logger.error(f"Found incorrect symlink pointing to a directory: {link_path} -> {target}")
                    
                    # Remove the incorrect link
                    os.remove(link_path)
                    logger.info(f"Removed incorrect symlink: {link_path}")
                    
                    # Find the correct library file
                    lib_prefix = link_name.split('.')[0]  # Get base name
                    real_lib = find_real_library(lib_prefix)
                    
                    if real_lib:
                        # Create new symlink to the real file
                        os.symlink(real_lib, link_path)
                        logger.info(f"Created new symlink: {link_path} -> {real_lib}")
                    else:
                        logger.error(f"Could not find a real {lib_prefix} library file")
                else:
                    logger.info(f"Symlink appears to be correct: {link_path} -> {target}")
            elif os.path.isdir(link_path):
                # It's a directory, not a symlink - this is definitely wrong
                logger.error(f"Found directory instead of symlink: {link_path}")
                
                # Try to find the correct library file
                lib_prefix = link_name.split('.')[0]
                real_lib = find_real_library(lib_prefix)
                
                if real_lib:
                    # Remove directory and create symlink
                    import shutil
                    shutil.rmtree(link_path)
                    os.symlink(real_lib, link_path)
                    logger.info(f"Removed directory and created symlink: {link_path} -> {real_lib}")
                else:
                    logger.error(f"Could not find a real {lib_prefix} library file")
            else:
                logger.info(f"Regular file found (not a symlink or directory): {link_path}")
        else:
            logger.warning(f"Link does not exist: {link_path}")

def main():
    """Main function"""
    logger.info("Fixing symlinks pointing to directories...")
    
    # Fix broken symlinks
    fix_broken_symlinks()
    
    # Check if ONNX Runtime can use CUDA
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        cuda_available = "CUDAExecutionProvider" in providers
        
        if cuda_available:
            logger.info(f"✅ ONNX Runtime CUDA support is available. Providers: {providers}")
        else:
            logger.error(f"❌ ONNX Runtime CUDA support is NOT available. Providers: {providers}")
    except Exception as e:
        logger.error(f"Error checking ONNX Runtime: {str(e)}")

if __name__ == "__main__":
    main()
