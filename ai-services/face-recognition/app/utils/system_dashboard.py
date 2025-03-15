#!/usr/bin/env python3
"""
System Status Dashboard for FaceSocial AI Services

This module provides comprehensive system status reports for the face recognition service,
including model loading status, hardware usage, and CUDA library status.
"""

import os
import sys
import subprocess
import glob
from typing import Dict, List, Any

class SystemDashboard:
    """System dashboard for monitoring the status of face recognition models and GPU acceleration"""
    
    def __init__(self):
        """Initialize the system dashboard"""
        pass
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        status = {
            "facial_recognition": [],
            "gender_detection": [],
            "total_models_loaded": 0
        }
        
        # Scan model directories for files
        model_dirs = ["/app/models", "/app/app/models"]
        for base_dir in model_dirs:
            if not os.path.exists(base_dir):
                continue
                
            # Look for face recognition models
            for model_type in ["facenet", "arcface", "cosface"]:
                models = glob.glob(f"{base_dir}/**/{model_type}/*", recursive=True)
                for model_path in models:
                    model_info = {
                        "name": model_type,
                        "path": model_path,
                        "exists": os.path.exists(model_path),
                        "type": self._get_model_type(model_path),
                    }
                    status["facial_recognition"].append(model_info)
            
            # Look for gender detection models
            gender_models = glob.glob(f"{base_dir}/**/gender/*", recursive=True)
            for model_path in gender_models:
                model_info = {
                    "name": "gender_detector",
                    "path": model_path,
                    "exists": os.path.exists(model_path),
                    "type": self._get_model_type(model_path),
                }
                status["gender_detection"].append(model_info)
        
        # Count total models found
        status["total_models_loaded"] = len(status["facial_recognition"]) + len(status["gender_detection"])
        
        # Check for ONNX GPU status
        onnx_gpu = self._check_onnx_gpu()
        status["onnx_gpu_available"] = onnx_gpu["available"]
        status["onnx_providers"] = onnx_gpu["providers"]
        
        return status
    
    def _get_model_type(self, path: str) -> str:
        """Determine the type of model from its file extension"""
        if path.endswith('.pb'):
            return 'TensorFlow'
        elif path.endswith('.h5'):
            return 'Keras'
        elif path.endswith('.onnx'):
            return 'ONNX'
        elif path.endswith('.caffemodel'):
            return 'Caffe'
        else:
            return 'Unknown'
    
    def _check_onnx_gpu(self) -> Dict[str, Any]:
        """Check if ONNX Runtime can use GPU"""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            return {
                "available": 'CUDAExecutionProvider' in providers,
                "providers": providers
            }
        except Exception as e:
            return {
                "available": False,
                "providers": [],
                "error": str(e)
            }
    
    def get_cuda_status(self) -> Dict[str, Any]:
        """Get status of CUDA libraries"""
        status = {
            "cuda_libraries": [],
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "Not set"),
            "cudnn_version": None,
            "cublas_version": None
        }
        
        # Check for critical CUDA libraries
        critical_libs = ["libcublas.so.11", "libcublasLt.so.11", "libcudnn.so.8"]
        for lib in critical_libs:
            lib_status = {"name": lib, "found": False, "path": None, "is_symlink": False, "target": None}
            
            # Try to find the library in standard locations
            for base_dir in ["/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64"]:
                path = os.path.join(base_dir, lib)
                if os.path.exists(path):
                    lib_status["found"] = True
                    lib_status["path"] = path
                    
                    # Check if it's a symlink
                    if os.path.islink(path):
                        lib_status["is_symlink"] = True
                        lib_status["target"] = os.readlink(path)
                    
                    break
            
            status["cuda_libraries"].append(lib_status)
        
        # Try to get CUDA version
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            status["cuda_version"] = result.stdout.strip() if result.returncode == 0 else "Not found"
        except:
            status["cuda_version"] = "Error retrieving version"
        
        return status

    def generate_html_report(self) -> str:
        """Generate HTML status report"""
        model_status = self.get_model_status()
        cuda_status = self.get_cuda_status()
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>FaceSocial AI System Status</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                h1, h2 { color: #333; }
                .status-card { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
                .status-card h2 { margin-top: 0; }
                .status-ok { color: green; }
                .status-warning { color: orange; }
                .status-error { color: red; }
                table { border-collapse: collapse; width: 100%; }
                th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
                tr:hover { background-color: #f5f5f5; }
            </style>
        </head>
        <body>
            <h1>FaceSocial AI System Status</h1>
            
            <div class="status-card">
                <h2>Model Status</h2>
                <h3>Face Recognition Models</h3>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Type</th>
                        <th>Path</th>
                        <th>Status</th>
                    </tr>
        """
        
        # Add face recognition models
        for model in model_status["facial_recognition"]:
            status_class = "status-ok" if model["exists"] else "status-error"
            status_text = "✓ Loaded" if model["exists"] else "✗ Not found"
            html += f"""
                    <tr>
                        <td>{model["name"]}</td>
                        <td>{model["type"]}</td>
                        <td>{model["path"]}</td>
                        <td class="{status_class}">{status_text}</td>
                    </tr>
            """
        
        html += """
                </table>
                
                <h3>Gender Detection Models</h3>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Type</th>
                        <th>Path</th>
                        <th>Status</th>
                    </tr>
        """
        
        # Add gender detection models
        for model in model_status["gender_detection"]:
            status_class = "status-ok" if model["exists"] else "status-error"
            status_text = "✓ Loaded" if model["exists"] else "✗ Not found"
            html += f"""
                    <tr>
                        <td>{model["name"]}</td>
                        <td>{model["type"]}</td>
                        <td>{model["path"]}</td>
                        <td class="{status_class}">{status_text}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="status-card">
                <h2>ONNX Runtime Status</h2>
        """
        
        # Add ONNX GPU status
        onnx_status_class = "status-ok" if model_status["onnx_gpu_available"] else "status-error"
        onnx_status_text = "✓ Available" if model_status["onnx_gpu_available"] else "✗ Not available"
        html += f"""
                <p>ONNX GPU Support: <span class="{onnx_status_class}">{onnx_status_text}</span></p>
                <p>Available providers: {', '.join(model_status["onnx_providers"])}</p>
            </div>
            
            <div class="status-card">
                <h2>CUDA Libraries</h2>
                <table>
                    <tr>
                        <th>Library</th>
                        <th>Found</th>
                        <th>Path</th>
                        <th>Symlink Target</th>
                    </tr>
        """
        
        # Add CUDA libraries
        for lib in cuda_status["cuda_libraries"]:
            status_class = "status-ok" if lib["found"] else "status-error"
            status_text = "✓ Found" if lib["found"] else "✗ Not found"
            target = lib["target"] if lib["is_symlink"] else "N/A"
            html += f"""
                    <tr>
                        <td>{lib["name"]}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{lib["path"] or "N/A"}</td>
                        <td>{target}</td>
                    </tr>
            """
        
        html += """
                </table>
                <p><strong>LD_LIBRARY_PATH:</strong> """ + cuda_status["ld_library_path"] + """</p>
                <p><strong>CUDA Version:</strong> """ + str(cuda_status.get("cuda_version", "Unknown")) + """</p>
            </div>
        </body>
        </html>
        """
        
        return html

# Add a simple API endpoint function
def get_system_status():
    """Get system status for API endpoint"""
    dashboard = SystemDashboard()
    return {
        "models": dashboard.get_model_status(),
        "cuda": dashboard.get_cuda_status()
    }
