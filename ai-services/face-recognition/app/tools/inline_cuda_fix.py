#!/usr/bin/env python3
import os
import glob
import subprocess

# Target directory for symlinks
target_dir = "/usr/lib/x86_64-linux-gnu"
os.makedirs(target_dir, exist_ok=True)

# Print current environment
print("Checking current environment...")
print(f"Current LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

# Find existing CUDA libraries
print("\nFinding CUDA libraries...")
cmd = "find /usr -name 'libcublas*' -o -name 'libcublasLt*' | sort"
output = subprocess.check_output(cmd, shell=True, text=True)
print(output)

# Create direct symlinks for critical libraries
source_paths = {
    "libcublas.so.11": [
        "/usr/local/cuda/lib64/libcublas.so",
        "/usr/local/cuda/lib64/libcublas.so.12",
        "/usr/lib/x86_64-linux-gnu/libcublas.so.12"
    ],
    "libcublasLt.so.11": [
        "/usr/local/cuda/lib64/libcublasLt.so",
        "/usr/local/cuda/lib64/libcublasLt.so.12",
        "/usr/lib/x86_64-linux-gnu/libcublasLt.so.12"
    ]
}

for target, sources in source_paths.items():
    target_path = os.path.join(target_dir, target)
    
    # Skip if target already exists
    if os.path.exists(target_path):
        print(f"Target already exists: {target_path}")
        continue
        
    # Try to find a source file that exists
    for source in sources:
        if os.path.exists(source):
            try:
                os.symlink(source, target_path)
                print(f"Created symlink: {target_path} -> {source}")
                break
            except Exception as e:
                print(f"Failed to create symlink {target_path} -> {source}: {e}")
    else:
        print(f"No valid source found for {target}")

# Verify created symlinks
print("\nVerifying created symlinks:")
for target in ["libcublas.so.11", "libcublasLt.so.11"]:
    target_path = os.path.join(target_dir, target)
    if os.path.exists(target_path):
        if os.path.islink(target_path):
            link_target = os.readlink(target_path)
            print(f"{target_path} -> {link_target} ({'exists' if os.path.exists(link_target) else 'broken link'})")
        else:
            print(f"{target_path} exists as a regular file")
    else:
        print(f"{target_path} does not exist")

# Check ONNX providers
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"\nONNX providers: {providers}")
    cuda_available = "CUDAExecutionProvider" in providers
    print(f"CUDA available: {'Yes' if cuda_available else 'No'}")
except Exception as e:
    print(f"Error checking ONNX providers: {e}")
