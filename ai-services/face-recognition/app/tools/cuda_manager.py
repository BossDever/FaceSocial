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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("cuda_manager")

# ONNX Runtime 1.16.0 requires CUDA 11.x libraries
REQUIRED_LIBS = {
    # Format: target_name: [source_alternatives]
    "libcublas.so.11": ["libcublas.so.12.*", "libcublas.so.12", "libcublas.so"],
    "libcublasLt.so.11": ["libcublasLt.so.12.*", "libcublasLt.so.12", "libcublasLt.so"],
    "libcudnn.so.8": ["libcudnn.so.8.*", "libcudnn.so"]
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
    
    # Process each required library
    for target_lib, source_alternatives in REQUIRED_LIBS.items():
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
            # If no source library found, try creating a direct symlink to non-versioned lib
            alt_source = None
            base_lib = target_lib.split('.')[0] + '.so'
            alt_path = os.path.join("/usr/local/cuda/lib64", base_lib)
            
            if os.path.exists(alt_path):
                alt_source = alt_path
                logger.info(f"Found alternative source: {alt_source}")
                success = create_symlink(alt_source, target_path)
                results[target_lib] = success
            else:
                logger.warning(f"❌ Could not find source file for {target_lib}")
                results[target_lib] = False
    
    return results

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

def create_hard_copy_if_needed(target_lib: str) -> bool:
    """
    Create a hard copy of the library if symlinking fails.
    This is a more aggressive approach that might work in restricted environments.
    """
    try:
        # Map to alternative libraries
        alternatives = {
            "libcublas.so.11": ["libcublas.so.12", "libcublas.so"],
            "libcublasLt.so.11": ["libcublasLt.so.12", "libcublasLt.so"],
        }
        
        # If the target library isn't in our mapping, return False
        if target_lib not in alternatives:
            return False
            
        source_options = alternatives[target_lib]
        target_path = os.path.join(TARGET_DIR, target_lib)
        
        # Try each source option
        for source_name in source_options:
            # Find all instances of this library
            source_files = find_library_files(source_name)
            
            if source_files:
                # Use the first match
                source_path = source_files[0]
                
                # Create a hard copy
                logger.info(f"Creating hard copy: {source_path} -> {target_path}")
                shutil.copy2(source_path, target_path)
                
                # Verify the copy
                if os.path.exists(target_path):
                    logger.info(f"✅ Successfully created hard copy for {target_lib}")
                    return True
        
        return False
    except Exception as e:
        logger.error(f"Error creating hard copy for {target_lib}: {e}")
        return False

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
    cmd = 'find /usr -name "libcublas*.so*" -o -name "libcudnn*.so*" | sort'
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

def fix_links_with_nvidia_container_toolkit():
    """
    Fix CUDA links using NVIDIA Container Toolkit paths.
    This is a backup approach that works well in containers.
    """
    # These are common paths in NVIDIA containers
    source_dir = "/usr/local/cuda/lib64"
    
    # Only proceed if the directory exists
    if not os.path.isdir(source_dir):
        logger.warning(f"{source_dir} does not exist, skipping toolkit approach")
        return False
        
    fixes_applied = 0
    
    # Create direct symlinks from CUDA libraries
    links = [
        ("libcublas.so", "libcublas.so.11"),
        ("libcublasLt.so", "libcublasLt.so.11"),
    ]
    
    for source_name, target_name in links:
        source_path = os.path.join(source_dir, source_name)
        target_path = os.path.join(TARGET_DIR, target_name)
        
        if os.path.exists(source_path):
            if create_symlink(source_path, target_path):
                fixes_applied += 1
    
    return fixes_applied > 0

def main():
    """Main function to fix CUDA libraries for ONNX Runtime."""
    logger.info("Starting CUDA Library Manager for ONNX Runtime GPU Acceleration")
    
    # Step 1: Display current environment
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    
    # Step 2: Display CUDA libraries before fixes
    display_cuda_libraries()
    
    # Step 3: Add standard library paths to LD_LIBRARY_PATH
    add_ld_library_paths()
    
    # Step 4: Fix CUDA libraries by creating symlinks
    results = fix_cuda_libraries()
    
    # Step 5: If any libraries are still missing, try the NVIDIA container toolkit approach
    if not all(results.values()):
        logger.info("Some libraries still missing, trying NVIDIA container toolkit approach")
        fix_links_with_nvidia_container_toolkit()
    
    # Step 6: If libraries are still missing, try creating hard copies
    for lib_name, success in results.items():
        if not success:
            logger.info(f"Creating hard copy for {lib_name}")
            create_hard_copy_if_needed(lib_name)
    
    # Step 7: Display CUDA libraries after fixes
    display_cuda_libraries()
    
    # Step 8: Verify ONNX Runtime CUDA support
    verify_onnx_cuda_support()
    
    logger.info("CUDA Library Manager completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
