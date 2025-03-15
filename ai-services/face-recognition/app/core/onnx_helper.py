import os
import sys
import onnxruntime as ort
from typing import List, Optional, Dict, Any, Tuple
import glob
import subprocess

def get_best_available_provider() -> List[str]:
    """
    Returns the best available ONNX runtime execution provider, with fallbacks.
    
    Returns:
        List of provider names in order of preference
    """
    try:
        available = ort.get_available_providers()
        print(f"Available ONNX providers: {available}")
        
        # Check for required CUDA libraries
        libraries_status = check_cuda_libraries()
        
        if 'CUDAExecutionProvider' in available and libraries_status['all_found']:
            print("✓ Using CUDA for ONNX models (GPU acceleration enabled)")
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif 'CUDAExecutionProvider' in available:
            print("⚠ CUDA provider available but missing libraries. Fixing...")
            if fix_cuda_libraries(libraries_status['missing']):
                print("✓ Fixed CUDA libraries, using CUDAExecutionProvider")
                # Reload providers after fixing libraries
                try:
                    available = ort.get_available_providers()
                    if 'CUDAExecutionProvider' in available:
                        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
                except:
                    pass
            
            print("⚠ Falling back to CPU provider")
            return ['CPUExecutionProvider']
        else:
            print("⚠ CUDA provider not available, using CPU only")
            return ['CPUExecutionProvider']
    except Exception as e:
        print(f"Error getting providers: {str(e)}")
        return ['CPUExecutionProvider']

def check_cuda_libraries() -> Dict[str, Any]:
    """
    Check if required CUDA libraries are available.
    
    Returns:
        Dict with status of library checks
    """
    required_libs = ['libcublas.so.11', 'libcublasLt.so.11', 'libcudnn.so.8']
    found_libs = []
    missing_libs = []
    
    # Library search paths
    search_paths = [
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/cuda/lib64',
        '/usr/local/cuda-11/lib64'
    ]
    
    # Add paths from LD_LIBRARY_PATH
    if 'LD_LIBRARY_PATH' in os.environ:
        for path in os.environ['LD_LIBRARY_PATH'].split(':'):
            if path and path not in search_paths:
                search_paths.append(path)
    
    # Check each required library
    for lib in required_libs:
        lib_found = False
        
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
    
    Parameters:
        model_path: Path to the ONNX model file
        
    Returns:
        ort.InferenceSession: ONNX runtime session
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    providers = get_best_available_provider()
    print(f"Creating ONNX session for {model_path} with providers: {providers}")
    
    try:
        # Try with specified providers
        session_options = ort.SessionOptions()
        # Enable optimization for better performance
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            model_path, 
            providers=providers,
            sess_options=session_options
        )
        
        # Verify provider was actually used
        used_provider = session.get_providers()[0]
        print(f"✓ ONNX Session created with provider: {used_provider}")
        
        return session
    except Exception as e:
        print(f"✗ Failed to create ONNX session with {providers}: {e}")
        print("Falling back to CPU provider...")
        
        try:
            # Fall back to CPU only
            session = ort.InferenceSession(
                model_path, 
                providers=["CPUExecutionProvider"]
            )
            print("✓ ONNX Session created with CPU provider")
            return session
        except Exception as e2:
            print(f"✗ Failed to create CPU session: {e2}")
            raise RuntimeError(f"Could not load ONNX model: {e2}")
