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
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("emergency_cuda_fix")

def find_cuda_libraries() -> dict:
    """Find CUDA libraries in the system"""
    result = {}
    libraries = [
        "libcublas.so*", 
        "libcublasLt.so*",
        "libcudnn.so*",
        "libcufft.so*",
        "libcurand.so*",
        "libcusolver.so*",
        "libcusparse.so*"
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

def map_cuda12_to_cuda11():
    """Aggressively map CUDA 12 libraries to CUDA 11 libraries"""
    target_dir = "/usr/lib/x86_64-linux-gnu"
    os.makedirs(target_dir, exist_ok=True)
    
    # Get a list of all CUDA 12 libraries in the system
    cmd = "find /usr -name 'libcublas*.so.12*' -o -name 'libcublasLt*.so.12*'"
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
        cuda12_libs = output.strip().split('\n') if output.strip() else []
        
        for lib_path in cuda12_libs:
            if not os.path.exists(lib_path):
                continue
                
            # Create the CUDA 11 equivalent name
            lib_name = os.path.basename(lib_path)
            lib11_name = lib_name.replace('.12', '.11')
            target_path = os.path.join(target_dir, lib11_name)
            
            # Create symlink
            try:
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.symlink(lib_path, target_path)
                logger.info(f"Created symlink: {target_path} -> {lib_path}")
            except Exception as e:
                logger.error(f"Failed to create symlink: {str(e)}")
    except Exception as e:
        logger.error(f"Error mapping CUDA 12 to CUDA 11: {str(e)}")

def create_specific_symlinks() -> bool:
    """Create specific symlinks for ONNX Runtime CUDA libraries"""
    target_dir = "/usr/lib/x86_64-linux-gnu"
    success = True
    
    # Critical libraries that ONNX Runtime needs
    critical_links = {
        "libcublas.so.11": ["libcublas.so.12", "libcublas.so"],
        "libcublasLt.so.11": ["libcublasLt.so.12", "libcublasLt.so"],
        "libcudnn.so.8": ["libcudnn.so.*", "libcudnn.so"],
        "libcufft.so.10": ["libcufft.so.12", "libcufft.so"],
        "libcurand.so.10": ["libcurand.so.12", "libcurand.so"],
        "libcusolver.so.11": ["libcusolver.so.12", "libcusolver.so"],
        "libcusparse.so.11": ["libcusparse.so.12", "libcusparse.so"]
    }
    
    for target, sources in critical_links.items():
        target_path = os.path.join(target_dir, target)
        
        # Skip if target already exists and is not a broken symlink
        if os.path.exists(target_path):
            if os.path.islink(target_path):
                link_target = os.readlink(target_path)
                if not os.path.exists(os.path.join(os.path.dirname(target_path), link_target)) and not os.path.isabs(link_target):
                    logger.warning(f"Found broken symlink: {target_path} -> {link_target}")
                    os.remove(target_path)
                else:
                    logger.info(f"Valid symlink exists: {target_path} -> {link_target}")
                    continue
            else:
                logger.info(f"Target is a real file, not a symlink: {target_path}")
                continue
        
        # Try to find source libraries
        source_path = None
        for source in sources:
            # First check in standard locations
            for source_dir in ["/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64"]:
                if "*" in source:
                    # Use glob for pattern matching
                    matches = glob.glob(os.path.join(source_dir, source))
                    if matches:
                        source_path = matches[0]
                        break
                else:
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
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.symlink(source_path, target_path)
                logger.info(f"Created symlink: {target_path} -> {source_path}")
            except Exception as e:
                logger.error(f"Failed to create symlink: {str(e)}")
                success = False
        else:
            logger.error(f"Could not find any source for {target}")
            success = False
    
    return success

def create_direct_symlinks() -> None:
    """Create direct symlinks for critical libraries as a last resort"""
    try:
        target_dir = "/usr/lib/x86_64-linux-gnu"
        cuda_lib_dir = "/usr/local/cuda/lib64"
        
        # Create directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Manual mapping of critical libraries
        commands = [
            f"ln -sf {cuda_lib_dir}/libcublas.so {target_dir}/libcublas.so.11",
            f"ln -sf {cuda_lib_dir}/libcublasLt.so {target_dir}/libcublasLt.so.11",
            f"ln -sf {target_dir}/libcudnn.so {target_dir}/libcudnn.so.8",
            f"ln -sf {cuda_lib_dir}/libcufft.so {target_dir}/libcufft.so.10",
            f"ln -sf {cuda_lib_dir}/libcurand.so {target_dir}/libcurand.so.10",
            f"ln -sf {cuda_lib_dir}/libcusolver.so {target_dir}/libcusolver.so.11",
            f"ln -sf {cuda_lib_dir}/libcusparse.so {target_dir}/libcusparse.so.11"
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, check=False)
                logger.info(f"Executed: {cmd}")
            except Exception as e:
                logger.error(f"Command failed: {cmd} - {e}")
        
        # Verify the created symlinks
        for lib in ["libcublas.so.11", "libcublasLt.so.11", "libcudnn.so.8"]:
            path = os.path.join(target_dir, lib)
            if os.path.exists(path):
                if os.path.islink(path):
                    logger.info(f"Symlink created: {path} -> {os.readlink(path)}")
                else:
                    logger.info(f"File exists: {path}")
            else:
                logger.error(f"Failed to create: {path}")
                
    except Exception as e:
        logger.error(f"Error creating direct symlinks: {str(e)}")

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

def create_onnx_test_file():
    """Create a simple ONNX model file for testing CUDA"""
    try:
        import numpy as np
        from onnx import helper, TensorProto, save
        
        # Create a simple test model
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
        
        model_def = helper.make_model(graph_def, producer_name="onnx-test")
        
        # Save the model
        model_path = "/tmp/cuda_test.onnx"
        save(model_def, model_path)
        logger.info(f"Created test ONNX model at {model_path}")
        
        # Try to run the model with CUDA
        import onnxruntime as ort
        
        # Create session with CUDA provider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session_options = ort.SessionOptions()
        try:
            session = ort.InferenceSession(
                model_path, 
                providers=providers,
                sess_options=session_options
            )
            
            # Run inference on random data
            input_data = np.random.rand(1, 3, 10, 10).astype(np.float32)
            outputs = session.run([output_name], {input_name: input_data})
            
            # Check which provider was used
            used_provider = session.get_providers()[0]
            logger.info(f"Test model ran successfully using provider: {used_provider}")
            
            return used_provider == "CUDAExecutionProvider"
        except Exception as e:
            logger.error(f"Failed to run test model: {e}")
            return False
    except Exception as e:
        logger.error(f"Error creating test model: {e}")
        return False

def main() -> int:
    """Main function"""
    logger.info("Starting Emergency CUDA Library Fix")
    
    # Step 1: Map CUDA 12 libraries to CUDA 11
    logger.info("Mapping CUDA 12 libraries to CUDA 11...")
    map_cuda12_to_cuda11()
    
    # Step 2: Find all existing CUDA libraries
    logger.info("Finding existing CUDA libraries...")
    libraries = find_cuda_libraries()
    
    # Step 3: Create specific symlinks for required libraries
    logger.info("Creating specific symlinks for required libraries...")
    success = create_specific_symlinks()
    
    # Step 4: Last resort - create direct symlinks
    logger.info("Creating direct symlinks as last resort...")
    create_direct_symlinks()
    
    # Step 5: Run a test to verify CUDA is working
    logger.info("Testing ONNX with CUDA...")
    cuda_test_result = create_onnx_test_file()
    
    # Step 6: Check if ONNX Runtime can use CUDA
    logger.info("Checking ONNX Runtime CUDA support...")
    cuda_available = check_onnx_cuda()
    
    if success and cuda_available:
        logger.info("✅ CUDA library fix successful")
        return 0
    else:
        logger.warning("⚠️ CUDA library fix may not be complete")
        if cuda_test_result:
            logger.info("✅ However, basic CUDA test successful!")
            return 0
        return 1

if __name__ == "__main__":
    sys.exit(main())
