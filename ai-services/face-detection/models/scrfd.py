"""SCRFD model implementation."""
import os
import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Tuple, Dict, Optional, Union, Any
import time
import logging

from shared.config.settings import settings

logger = logging.getLogger(__name__)

class SCRFDDetector:
    """
    Face Detector ที่ใช้โมเดล SCRFD จาก Insightface
    รองรับการเร่งความเร็วด้วย GPU
    """
    
    def __init__(
        self,
        model_file: str,
        use_gpu: bool = True,
        gpu_id: int = 0,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize SCRFD face detector
        
        Args:
            model_file: path to SCRFD ONNX model file
            use_gpu: use GPU acceleration if available
            gpu_id: ID of GPU to use
            confidence_threshold: threshold for detection confidence
            nms_threshold: threshold for non-maximum suppression
            input_size: input size for the model (width, height)
        """
        self.model_file = model_file
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        # ตรวจสอบว่าไฟล์โมเดลมีอยู่หรือไม่
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"SCRFD model file not found: {model_file}")
        
        # ตั้งค่า ONNX runtime session
        self.session = self._create_session()
        
        # ดึงข้อมูลของ input และ output
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # จัดเก็บข้อมูลสำหรับ post-processing
        self.center_points = {}
        self.strides = [8, 16, 32]
        logger.info(f"SCRFD detector initialized with {'GPU' if self.use_gpu else 'CPU'}")
    
    def _create_session(self) -> ort.InferenceSession:
        """
        สร้าง ONNX Runtime session สำหรับโมเดล SCRFD
        """
        try:
            providers = []
            
            if self.use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append(('CUDAExecutionProvider', {'device_id': self.gpu_id}))
                logger.info(f"Using CUDA for SCRFD with GPU ID: {self.gpu_id}")
            
            providers.append('CPUExecutionProvider')
            
            # สร้าง ONNX Runtime session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.log_severity_level = 3  # Warning level
            
            session = ort.InferenceSession(
                self.model_file,
                sess_options=session_options,
                providers=providers
            )
            
            # แสดงข้อมูล provider ที่ใช้งาน
            logger.info(f"ONNX Runtime providers: {session.get_providers()}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {str(e)}")
            raise
    
    def detect_faces(
        self,
        image: np.ndarray,
        min_face_size: int = 20,
        confidence_threshold: Optional[float] = None,
        return_landmarks: bool = True
    ) -> List[Dict[str, Any]]:
        """
        ตรวจจับใบหน้าในรูปภาพ
        
        Args:
            image: รูปภาพในรูปแบบ numpy array (BGR)
            min_face_size: ขนาดใบหน้าขั้นต่ำที่ต้องการตรวจจับ (พิกเซล)
            confidence_threshold: ค่า threshold สำหรับความมั่นใจในการตรวจจับ (ถ้าไม่ระบุจะใช้ค่าเริ่มต้น)
            return_landmarks: ต้องการ landmarks หรือไม่
            
        Returns:
            List of detected faces with bounding boxes, confidence scores, and landmarks (if requested)
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        start_time = time.time()
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Prepare the input image
        input_img, scale_factor = self._preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_img})
        
        # Post-process the results
        faces = self._postprocess(
            outputs,
            scale_factor,
            img_width,
            img_height,
            min_face_size,
            confidence_threshold,
            return_landmarks
        )
        
        logger.debug(f"Detected {len(faces)} faces in {(time.time() - start_time) * 1000:.2f} ms")
        return faces
    
    def batch_detect(
        self,
        images: List[np.ndarray],
        min_face_size: int = 20,
        confidence_threshold: Optional[float] = None,
        return_landmarks: bool = True,
        batch_size: int = 16
    ) -> List[List[Dict[str, Any]]]:
        """
        ตรวจจับใบหน้าในหลายรูปภาพ
        
        Args:
            images: รายการรูปภาพในรูปแบบ numpy array (BGR)
            min_face_size: ขนาดใบหน้าขั้นต่ำที่ต้องการตรวจจับ (พิกเซล)
            confidence_threshold: ค่า threshold สำหรับความมั่นใจในการตรวจจับ (ถ้าไม่ระบุจะใช้ค่าเริ่มต้น)
            return_landmarks: ต้องการ landmarks หรือไม่
            batch_size: ขนาด batch สำหรับการประมวลผล
            
        Returns:
            List of lists, containing detected faces for each image
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        start_time = time.time()
        results = []
        
        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_results = []
            
            # Process each image in batch
            for img in batch_images:
                img_faces = self.detect_faces(
                    img, 
                    min_face_size, 
                    confidence_threshold, 
                    return_landmarks
                )
                batch_results.append(img_faces)
            
            results.extend(batch_results)
        
        logger.debug(f"Batch processed {len(images)} images in {(time.time() - start_time) * 1000:.2f} ms")
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        ประมวลผลรูปภาพก่อนการตรวจจับใบหน้า
        
        Args:
            image: รูปภาพต้นฉบับในรูปแบบ numpy array (BGR)
            
        Returns:
            Tuple containing processed image and scale factor
        """
        # Calculate scale factor
        img_height, img_width = image.shape[:2]
        scale = min(self.input_size[1] / img_height, self.input_size[0] / img_width)
        scale_factor = 1.0 / scale
        
        # Resize image
        resized_img = cv2.resize(
            image, 
            (0, 0), 
            fx=scale, 
            fy=scale, 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Calculate padding
        pad_h = self.input_size[1] - resized_img.shape[0]
        pad_w = self.input_size[0] - resized_img.shape[1]
        
        # Add padding
        padded_img = cv2.copyMakeBorder(
            resized_img, 
            0, pad_h, 0, pad_w, 
            cv2.BORDER_CONSTANT, 
            value=(0, 0, 0)
        )
        
        # Convert to RGB and normalize
        padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        padded_img = padded_img.astype(np.float32) / 255.0
        
        # HWC to NCHW format
        padded_img = padded_img.transpose(2, 0, 1)[np.newaxis, ...]
        
        return padded_img, scale_factor
    
    def _postprocess(
        self,
        outputs: List[np.ndarray],
        scale_factor: float,
        img_width: int,
        img_height: int,
        min_face_size: int,
        confidence_threshold: float,
        return_landmarks: bool
    ) -> List[Dict[str, Any]]:
        """
        ประมวลผลผลลัพธ์จากการตรวจจับใบหน้า
        
        Args:
            outputs: ผลลัพธ์จาก ONNX inference
            scale_factor: ค่า scale factor จากการ preprocess
            img_width: ความกว้างของรูปภาพต้นฉบับ
            img_height: ความสูงของรูปภาพต้นฉบับ
            min_face_size: ขนาดใบหน้าขั้นต่ำที่ต้องการตรวจจับ (พิกเซล)
            confidence_threshold: ค่า threshold สำหรับความมั่นใจ
            return_landmarks: ต้องการ landmarks หรือไม่
            
        Returns:
            List of detected faces with bounding boxes, confidence scores, and landmarks (if requested)
        """
        all_boxes = []
        all_scores = []
        all_landmarks = []
        
        # จำนวน outputs ควรเป็น 9 (3 strides x (boxes, scores, landmarks))
        num_outputs = len(outputs)
        expected_outputs = 9 if return_landmarks else 6
        
        if num_outputs < expected_outputs:
            logger.warning(f"Expected at least {expected_outputs} outputs, got {num_outputs}")
        
        # วนลูปตาม strides
        for stride_idx, stride in enumerate(self.strides):
            # ดึง output สำหรับ stride นี้
            if stride_idx * 3 >= num_outputs:
                continue
                
            # ดึงข้อมูลสำหรับ stride นี้
            boxes = outputs[stride_idx * 3]
            scores = outputs[stride_idx * 3 + 1]
            
            # ถ้าต้องการ landmarks
            if return_landmarks and (stride_idx * 3 + 2) < num_outputs:
                landmarks = outputs[stride_idx * 3 + 2]
            else:
                landmarks = None
            
            # ประมวลผลกล่องสำหรับ stride นี้
            stride_boxes, stride_scores, stride_landmarks = self._process_stride(
                boxes, scores, landmarks, stride, scale_factor, confidence_threshold
            )
            
            all_boxes.append(stride_boxes)
            all_scores.append(stride_scores)
            
            if landmarks is not None:
                all_landmarks.append(stride_landmarks)
        
        # รวม bounding boxes, scores, และ landmarks
        all_boxes = np.vstack(all_boxes) if all_boxes else np.array([])
        all_scores = np.vstack(all_scores) if all_scores else np.array([])
        
        if all_landmarks and return_landmarks:
            all_landmarks = np.vstack(all_landmarks)
        else:
            all_landmarks = None
        
        # Non-maximum suppression
        keep_indices = self._nms(all_boxes, all_scores, self.nms_threshold)
        
        # Create results list
        faces = []
        for idx in keep_indices:
            # ดึงข้อมูลกล่อง
            x1, y1, x2, y2 = all_boxes[idx]
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_width, int(x2))
            y2 = min(img_height, int(y2))
            
            # ตรวจสอบขนาดของใบหน้า
            width = x2 - x1
            height = y2 - y1
            
            if width < min_face_size or height < min_face_size:
                continue
            
            face_data = {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(all_scores[idx][0])
            }
            
            # เพิ่ม landmarks ถ้าต้องการ
            if all_landmarks is not None:
                landmarks = all_landmarks[idx].reshape(5, 2)
                landmarks[:, 0] = np.clip(landmarks[:, 0], 0, img_width)
                landmarks[:, 1] = np.clip(landmarks[:, 1], 0, img_height)
                face_data["landmarks"] = landmarks.tolist()
            
            faces.append(face_data)
        
        return faces
    
    def _process_stride(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        landmarks: Optional[np.ndarray],
        stride: int,
        scale_factor: float,
        confidence_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        ประมวลผลกล่องสำหรับ stride นี้
        
        Args:
            boxes: กล่องจาก output
            scores: scores จาก output
            landmarks: landmarks จาก output (ถ้ามี)
            stride: ค่า stride ปัจจุบัน
            scale_factor: ค่า scale factor
            confidence_threshold: ค่า threshold สำหรับความมั่นใจ
            
        Returns:
            Tuple of processed boxes, scores, and landmarks
        """
        # ดึงความมั่นใจสำหรับใบหน้า
        scores = scores[:, 1:2]
        
        # คัดกรองด้วย confidence threshold
        mask = scores >= confidence_threshold
        boxes = boxes[mask.reshape(-1)]
        scores = scores[mask.reshape(-1)]
        
        if landmarks is not None:
            landmarks = landmarks[mask.reshape(-1)]
        
        # ถ้าไม่มีใบหน้าที่ตรวจพบ
        if boxes.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]) if landmarks is not None else None
        
        # แปลงค่ากลับไปเป็นพิกัดในรูปภาพต้นฉบับ
        boxes = self._distance2box(self._get_center_points(boxes.shape[0], stride), boxes)
        boxes *= scale_factor
        
        if landmarks is not None:
            landmarks = self._distance2landmark(self._get_center_points(landmarks.shape[0], stride), landmarks)
            landmarks *= scale_factor
        
        return boxes, scores, landmarks
    
    def _get_center_points(self, num_boxes: int, stride: int) -> np.ndarray:
        """
        สร้างจุดศูนย์กลางสำหรับแต่ละกล่อง
        
        Args:
            num_boxes: จำนวนกล่อง
            stride: ค่า stride
            
        Returns:
            Array of center points
        """
        if stride not in self.center_points:
            # คำนวณขนาดของ feature map
            feature_map_size = self.input_size[0] // stride
            # สร้าง grid
            shift_x = np.arange(0, feature_map_size) * stride
            shift_y = np.arange(0, feature_map_size) * stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            self.center_points[stride] = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
        
        return self.center_points[stride][:num_boxes]
    
    def _distance2box(self, center_points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """
        แปลงระยะทางเป็นกล่อง
        
        Args:
            center_points: จุดศูนย์กลาง
            distance: ระยะทาง
            
        Returns:
            Array of bounding boxes
        """
        x1 = center_points[:, 0] - distance[:, 0]
        y1 = center_points[:, 1] - distance[:, 1]
        x2 = center_points[:, 0] + distance[:, 2]
        y2 = center_points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=1)
    
    def _distance2landmark(self, center_points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """
        แปลงระยะทางเป็น landmarks
        
        Args:
            center_points: จุดศูนย์กลาง
            distance: ระยะทาง
            
        Returns:
            Array of landmarks
        """
        x = center_points[:, 0:1].repeat(5, axis=1)
        y = center_points[:, 1:2].repeat(5, axis=1)
        
        x += distance[:, 0::2]
        y += distance[:, 1::2]
        
        return np.stack([x, y], axis=2).reshape(-1, 10)
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        ใช้ non-maximum suppression เพื่อลดจำนวนกล่องที่ซ้ำซ้อน
        
        Args:
            boxes: กล่องในรูปแบบ [x1, y1, x2, y2]
            scores: คะแนนสำหรับแต่ละกล่อง
            threshold: ค่า threshold สำหรับ IoU
            
        Returns:
            Indices of boxes to keep
        """
        if boxes.shape[0] == 0:
            return np.array([], dtype=np.int32)
        
        # แปลง scores เป็น 1D array
        if len(scores.shape) > 1:
            scores = scores.squeeze()
        
        # คำนวณพื้นที่ของแต่ละกล่อง
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1) * (y2 - y1)
        
        # เรียงลำดับตาม scores
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # คำนวณ IoU กับกล่องที่เหลือ
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            # คำนวณ IoU
            overlap = intersection / (area[i] + area[order[1:]] - intersection)
            
            # คัดกรองกล่องที่มี IoU เกิน threshold
            inds = np.where(overlap <= threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=np.int32)