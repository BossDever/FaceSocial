import os
import sys
import onnxruntime as ort
from typing import List, Optional, Dict, Any, Tuple
import glob
import subprocess
import logging

def get_best_available_provider() -> List[str]:
    """
    Returns the best available ONNX runtime execution provider, with fallbacks.
    
    Returns:
        List of provider names in order of preference
    """
    try:
        available = ort.get_available_providers()
        print(f"Available ONNX providers: {available}")
        
        # First check for direct CUDA availability
        if 'CUDAExecutionProvider' in available:
            # Let's try CUDA first but don't check libraries yet
            # ONNX Runtime will fall back to CPU if CUDA fails
            print("CUDAExecutionProvider available, attempting to use it")
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            print("CUDA provider not available in ONNX Runtime")
            return ['CPUExecutionProvider']
    except Exception as e:
        print(f"Error getting providers: {str(e)}")
        return ['CPUExecutionProvider']

def verify_cuda_available() -> bool:
    """
    Verify if CUDA is actually available by testing ONNX with a simple model
    
    Returns:
        bool: True if CUDA is working, False otherwise
    """
    try:
        # Create a very simple ONNX model
        import numpy as np
        
        # Get ONNX providers
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' not in providers:
            print("CUDAExecutionProvider not available in ONNX Runtime")
            return False
        
        print(f"Available ONNX providers: {providers}")
        
        # Try creating a simple model and run inference
        try:
            # Create a simple test tensor
            test_data = np.random.rand(1, 3, 10, 10).astype(np.float32)
            
            # Set up a simple session options
            options = ort.SessionOptions()
            options.log_severity_level = 3  # Reduce verbosity
            
            # Create an in-memory model for testing
            input_name = "input"
            output_name = "output"
            
            from onnx import helper, TensorProto
            import onnx
            
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
            
            model_def = helper.make_model(graph_def, producer_name="onnx-example")
            model_bytes = model_def.SerializeToString()
            
            # Try to create a session with CUDA provider
            cuda_session = ort.InferenceSession(
                model_bytes,
                providers=["CUDAExecutionProvider"],
                sess_options=options
            )
            
            # Run inference
            outputs = cuda_session.run([output_name], {input_name: test_data})
            
            # If we got here, CUDA is working
            print("✅ CUDA verification successful - ran test model on GPU")
            return True
            
        except Exception as e:
            print(f"❌ CUDA verification failed: {str(e)}")
            return False
            
    except Exception as e:
        print(f"❌ Error during CUDA verification: {str(e)}")
        return False

def check_cuda_libraries() -> Dict[str, Any]:
    """
    Check if required CUDA libraries are available.
    
    Returns:
        Dict with status of library checks
    """
    required_libs = ['libcublas.so.11', 'libcublasLt.so.11', 'libcudnn.so.8']
    found_libs = []
    missing_libs = []
    
    # Library search paths - include more possible locations
    search_paths = [
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/cuda/lib64', 
        '/usr/local/cuda-11/lib64',
        '/usr/local/cuda/targets/x86_64-linux/lib'
    ]
    
    # Add paths from LD_LIBRARY_PATH
    if 'LD_LIBRARY_PATH' in os.environ:
        for path in os.environ['LD_LIBRARY_PATH'].split(':'):
            if path and path not in search_paths:
                search_paths.append(path)
    
    # Check each required library
    for lib in required_libs:
        lib_found = False
        
        # First check for direct matches
        for path in search_paths:
            full_path = os.path.join(path, lib)
            if os.path.exists(full_path) or os.path.islink(full_path):
                # Check if link is valid when it's a symbolic link
                if os.path.islink(full_path):
                    link_target = os.readlink(full_path)
                    # Handle relative paths in symlinks
                    if not os.path.isabs(link_target):
                        link_target = os.path.join(os.path.dirname(full_path), link_target)
                    
                    if os.path.exists(link_target):
                        print(f"✓ Found {lib} at {full_path} -> {link_target}")
                        lib_found = True
                        found_libs.append((lib, full_path, link_target))
                        break
                    else:
                        print(f"⚠ Found broken symlink {full_path} -> {link_target}")
                else:
                    print(f"✓ Found {lib} at {full_path}")
                    lib_found = True
                    found_libs.append((lib, full_path, None))
                    break
        
        # If not found, check for version-agnostic libs
        if not lib_found:
            base_lib = lib.split('.')[0] + '.so'  # e.g., libcublas.so
            
            for path in search_paths:
                full_path = os.path.join(path, base_lib)
                if os.path.exists(full_path) or os.path.islink(full_path):
                    print(f"✓ Found {base_lib} at {full_path} (can be used for {lib})")
                    lib_found = True
                    found_libs.append((lib, full_path, None))
                    break
        
        if not lib_found:
            print(f"✗ Missing required library: {lib}")
            missing_libs.append(lib)
    
    return {
        'all_found': len(missing_libs) == 0,
        'found': found_libs,
        'missing': missing_libs
    }

def fix_cuda_libraries(missing_libs: List[str]) -> bool:
    """
    Try to fix missing CUDA libraries by creating symlinks.
    
    Parameters:
        missing_libs: List of missing library names
    
    Returns:
        bool: True if fixed successfully, False otherwise
    """
    if not missing_libs:
        return True
    
    fixed_count = 0
    
    for lib in missing_libs:
        fixed = _create_symlink_for_lib(lib)
        if fixed:
            fixed_count += 1
    
    return fixed_count == len(missing_libs)

def _create_symlink_for_lib(lib_name: str) -> bool:
    """Create a symbolic link for the specified library."""
    try:
        # Map of target libs to possible source libs (in order of preference)
        lib_mapping = {
            'libcublas.so.11': ['libcublas.so.12', 'libcublas.so'],
            'libcublasLt.so.11': ['libcublasLt.so.12', 'libcublasLt.so'],
            'libcudnn.so.8': ['libcudnn.so.8.*', 'libcudnn.so']
        }
        
        # Get potential source libraries
        potential_sources = lib_mapping.get(lib_name, [])
        
        # Find existing libraries that match potential sources
        for pattern in potential_sources:
            source_libs = []
            
            if '*' in pattern:
                # Use glob for pattern matching
                for path in ['/usr/lib/x86_64-linux-gnu', '/usr/local/cuda*/lib64']:
                    source_libs.extend(glob.glob(f"{path}/{pattern}"))
            else:
                # Search in common locations
                for path in ['/usr/lib/x86_64-linux-gnu', '/usr/local/cuda*/lib64']:
                    if os.path.exists(f"{path}/{pattern}"):
                        source_libs.append(f"{path}/{pattern}")
            
            if source_libs:
                # Use the first found library
                source_lib = source_libs[0]
                target_path = f"/usr/lib/x86_64-linux-gnu/{lib_name}"
                
                # Create the symbolic link
                try:
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    os.symlink(source_lib, target_path)
                    print(f"✓ Created symlink: {source_lib} -> {target_path}")
                    return True
                except Exception as e:
                    print(f"✗ Failed to create symlink: {str(e)}")
        
        print(f"✗ No suitable source found for {lib_name}")
        return False
    except Exception as e:
        print(f"✗ Error fixing library {lib_name}: {str(e)}")
        return False

def create_onnx_session(model_path: str) -> ort.InferenceSession:
    """
    Create an ONNX runtime session with the best available provider.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    providers = get_best_available_provider()
    print(f"Creating ONNX session for {model_path} with providers: {providers}")
    
    try:
        # First try with GPU acceleration
        if 'CUDAExecutionProvider' in providers:
            try:
                # Set execution provider options for better performance
                provider_options = [
                    {'device_id': 0, 
                     'arena_extend_strategy': 'kNextPowerOfTwo',
                     'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                     'cudnn_conv_algo_search': 'EXHAUSTIVE',
                     'do_copy_in_default_stream': True,
                    }
                ]
                
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(
                    model_path, 
                    providers=[('CUDAExecutionProvider', provider_options), 'CPUExecutionProvider'],
                    sess_options=session_options
                )
                
                # Verify that CUDA is actually being used
                if 'CUDAExecutionProvider' in session.get_providers():
                    print(f"✅ ONNX Session created with CUDA provider")
                    return session
                else:
                    print("⚠️ CUDA provider registration succeeded but not selected by ONNX Runtime")
                    # Fall through to CPU
            except Exception as e:
                print(f"⚠️ Failed to create CUDA session: {e}")
                # Fall through to CPU
        
        # CPU fallback
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider'],
            sess_options=session_options
        )
        print(f"⚠️ ONNX Session created with CPU provider")
        return session
        
    except Exception as e:
        print(f"❌ Failed to create ONNX session: {e}")
        raise RuntimeError(f"Could not load ONNX model: {e}")
