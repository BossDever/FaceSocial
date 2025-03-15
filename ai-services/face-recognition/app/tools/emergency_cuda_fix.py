#!/usr/bin/env python3
"""
Emergency CUDA Library Fix for ONNX Runtime

This script creates the necessary symbolic links for CUDA libraries
required by ONNX Runtime, with a focus on fixing the libcublasLt.so.11 issue.
"""

import os
import glob
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("emergency_cuda_fix")

def find_cuda_libraries() -> dict:
    """Find CUDA libraries in the system"""
    result = {}
    libraries = [
        "libcublas.so*", 
        "libcublasLt.so*",
        "libcudnn.so*"
    ]
    
    for lib_pattern in libraries:
        cmd = f"find /usr -name '{lib_pattern}' | sort"
        try:
            output = subprocess.check_output(cmd, shell=True, text=True)
            result[lib_pattern] = output.strip().split("\n") if output.strip() else []
            logger.info(f"Found {len(result[lib_pattern])} matches for {lib_pattern}")
            for lib in result[lib_pattern]:
                logger.info(f"  - {lib}")
        except subprocess.CalledProcessError:
            result[lib_pattern] = []
            logger.warning(f"Error finding {lib_pattern}")
    
    return result

def create_specific_symlinks() -> bool:
    """Create specific symlinks for ONNX Runtime CUDA libraries"""
    target_dir = "/usr/lib/x86_64-linux-gnu"
    success = True
    
    # Critical libraries that ONNX Runtime needs
    critical_links = {
        "libcublas.so.11": ["libcublas.so.12", "libcublas.so"],
        "libcublasLt.so.11": ["libcublasLt.so.12", "libcublasLt.so"]
    }
    
    for target, sources in critical_links.items():
        target_path = os.path.join(target_dir, target)
        
        # Skip if target already exists
        if os.path.exists(target_path):
            logger.info(f"Target already exists: {target_path}")
            
            # Check if it's a valid link
            if os.path.islink(target_path):
                link_target = os.readlink(target_path)
                if not os.path.exists(link_target):
                    logger.warning(f"Found broken symlink: {target_path} -> {link_target}")
                    # Remove broken link
                    os.remove(target_path)
                else:
                    logger.info(f"Valid symlink: {target_path} -> {link_target}")
                    continue
            else:
                logger.info(f"Target is a real file, not a symlink: {target_path}")
                continue
        
        # Try to find source libraries
        source_path = None
        for source in sources:
            # First check in standard locations
            for source_dir in ["/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64"]:
                candidate = os.path.join(source_dir, source)
                if os.path.exists(candidate):
                    source_path = candidate
                    break
            
            if source_path:
                break
            
            # If not found, search more broadly
            cmd = f"find /usr -name '{source}' | head -1"
            try:
                output = subprocess.check_output(cmd, shell=True, text=True).strip()
                if output:
                    source_path = output
                    break
            except:
                pass
        
        # Create symlink if source was found
        if source_path and os.path.exists(source_path):
            try:
                os.symlink(source_path, target_path)
                logger.info(f"Created symlink: {target_path} -> {source_path}")
            except Exception as e:
                logger.error(f"Failed to create symlink: {str(e)}")
                success = False
        else:
            logger.error(f"Could not find any source for {target}")
            success = False
    
    return success

def check_onnx_cuda() -> bool:
    """Check if ONNX Runtime can use CUDA"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        cuda_available = "CUDAExecutionProvider" in providers
        
        if cuda_available:
            logger.info(f"✅ ONNX Runtime CUDA support is available. Providers: {providers}")
            return True
        else:
            logger.error(f"❌ ONNX Runtime CUDA support is NOT available. Providers: {providers}")
            return False
    except Exception as e:
        logger.error(f"Error checking ONNX Runtime: {str(e)}")
        return False

def create_direct_symlinks() -> None:
    """Create direct symlinks for critical libraries as a last resort"""
    try:
        # Direct creation of symlinks without checking
        os.system("ln -sf /usr/lib/x86_64-linux-gnu/libcublas.so.12 /usr/lib/x86_64-linux-gnu/libcublas.so.11")
        os.system("ln -sf /usr/lib/x86_64-linux-gnu/libcublasLt.so.12 /usr/lib/x86_64-linux-gnu/libcublasLt.so.11")
        logger.info("Created direct symlinks from CUDA 12 to CUDA 11 libraries")
        
        # Alternative method
        if not os.path.exists("/usr/lib/x86_64-linux-gnu/libcublasLt.so.11"):
            source_cmd = "find /usr -name 'libcublasLt.so*' | sort | head -1"
            source = subprocess.check_output(source_cmd, shell=True, text=True).strip()
            if source:
                os.system(f"ln -sf {source} /usr/lib/x86_64-linux-gnu/libcublasLt.so.11")
                logger.info(f"Created emergency symlink from {source} to libcublasLt.so.11")
    except Exception as e:
        logger.error(f"Error creating direct symlinks: {str(e)}")

def main() -> int:
    """Main function"""
    logger.info("Starting Emergency CUDA Library Fix")
    
    # Find CUDA libraries
    libraries = find_cuda_libraries()
    
    # Create specific symlinks
    success = create_specific_symlinks()
    
    # Last resort - create direct symlinks
    create_direct_symlinks()
    
    # Check if ONNX Runtime can use CUDA
    cuda_available = check_onnx_cuda()
    
    if success and cuda_available:
        logger.info("✅ CUDA library fix successful")
        return 0
    else:
        logger.warning("⚠️ CUDA library fix may not be complete")
        return 1

if __name__ == "__main__":
    main()
