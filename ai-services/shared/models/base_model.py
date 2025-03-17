import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import onnxruntime as ort
import torch
from loguru import logger

from ..config.settings import settings

class BaseModel(ABC):
    """
    Base class สำหรับโมเดล AI ทั้งหมดใน FaceSocial
    """
    def __init__(
        self,
        model_name: str,
        model_type: str,
        input_shape: Tuple[int, ...],
        use_tensorrt: bool = True,
        precision: str = "fp16",
        gpu_id: int = 0
    ):
        """
        Initialize โมเดล AI
        
        Args:
            model_name: ชื่อของโมเดล
            model_type: ประเภทของโมเดล (facenet, arcface, etc.)
            input_shape: รูปร่างของ input (e.g., (1, 3, 112, 112))
            use_tensorrt: ใช้ TensorRT optimization หรือไม่
            precision: precision ที่ใช้ (fp16, fp32, int8)
            gpu_id: ID ของ GPU ที่จะใช้
        """
        self.model_name = model_name
        self.model_type = model_type
        self.input_shape = input_shape
        self.use_tensorrt = use_tensorrt and settings.USE_GPU
        self.precision = precision
        self.gpu_id = gpu_id if settings.USE_GPU else -1
        
        # Path to different model formats
        self.model_dir = os.path.join(settings.MODEL_PATH, model_type, model_name)
        self.onnx_path = os.path.join(self.model_dir, "onnx", f"{model_name}.onnx")
        self.tensorrt_path = os.path.join(self.model_dir, "tensorrt", f"{model_name}_{precision}.engine")
        self.original_path = os.path.join(self.model_dir, "original", f"{model_name}.pth")
        
        # ONNX Runtime session
        self.session = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self) -> None:
        """
        โหลดโมเดลตามลำดับความสำคัญ: TensorRT > ONNX > Original
        """
        if settings.USE_GPU:
            # Setup ONNX Runtime with GPU
            gpu_options = {"device_id": self.gpu_id}
            
            # พยายามโหลด TensorRT engine ถ้าเปิดใช้งาน
            if self.use_tensorrt and os.path.exists(self.tensorrt_path):
                logger.info(f"Loading TensorRT engine from {self.tensorrt_path}")
                providers = [("TensorrtExecutionProvider", gpu_options), ("CUDAExecutionProvider", gpu_options)]
                self.session = ort.InferenceSession(self.tensorrt_path, providers=providers)
                logger.info(f"TensorRT engine loaded successfully")
            
            # ถ้าไม่สามารถโหลด TensorRT ได้ ใช้ ONNX + CUDA แทน
            elif os.path.exists(self.onnx_path):
                logger.info(f"Loading ONNX model from {self.onnx_path} with CUDA")
                providers = [("CUDAExecutionProvider", gpu_options)]
                self.session = ort.InferenceSession(self.onnx_path, providers=providers)
                logger.info(f"ONNX model loaded successfully with CUDA")
            
            # ถ้าไม่มีทั้ง TensorRT และ ONNX ใช้โมเดลดั้งเดิม
            elif os.path.exists(self.original_path):
                logger.info(f"Loading original PyTorch model from {self.original_path}")
                self._load_original_model()
                logger.info(f"Original PyTorch model loaded successfully")
            
            else:
                raise FileNotFoundError(f"No model found for {self.model_name} in {self.model_dir}")
        
        else:
            # ถ้าไม่ใช้ GPU ใช้ ONNX + CPU
            if os.path.exists(self.onnx_path):
                logger.info(f"Loading ONNX model from {self.onnx_path} with CPU")
                self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
                logger.info(f"ONNX model loaded successfully with CPU")
            
            # ถ้าไม่มี ONNX ใช้โมเดลดั้งเดิม
            elif os.path.exists(self.original_path):
                logger.info(f"Loading original PyTorch model from {self.original_path}")
                self._load_original_model()
                logger.info(f"Original PyTorch model loaded successfully")
            
            else:
                raise FileNotFoundError(f"No model found for {self.model_name} in {self.model_dir}")
    
    @abstractmethod
    def _load_original_model(self) -> None:
        """
        โหลดโมเดลดั้งเดิม (PyTorch) - ให้ implement ในคลาสลูก
        """
        pass
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        ทำนายผลลัพธ์จากโมเดล
        
        Args:
            input_data: ข้อมูล input ในรูปแบบ numpy array
            
        Returns:
            np.ndarray: ผลลัพธ์จากโมเดล
        """
        if self.session is not None:
            # ใช้ ONNX Runtime
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_data})
            return outputs[0]
        else:
            # ใช้โมเดลดั้งเดิม
            return self._predict_with_original_model(input_data)
    
    @abstractmethod
    def _predict_with_original_model(self, input_data: np.ndarray) -> np.ndarray:
        """
        ทำนายผลลัพธ์โดยใช้โมเดลดั้งเดิม - ให้ implement ในคลาสลูก
        
        Args:
            input_data: ข้อมูล input ในรูปแบบ numpy array
            
        Returns:
            np.ndarray: ผลลัพธ์จากโมเดล
        """
        pass
    
    def __del__(self):
        """
        Cleanup resources เมื่อ object ถูกทำลาย
        """
        # ปิด ONNX Runtime session
        if self.session is not None:
            self.session = None
        
        # ทำความสะอาด PyTorch resources
        if hasattr(self, 'torch_model'):
            if self.torch_model is not None:
                del self.torch_model
        
        # ทำความสะอาด CUDA cache
        if settings.USE_GPU:
            try:
                torch.cuda.empty_cache()
            except:
                pass