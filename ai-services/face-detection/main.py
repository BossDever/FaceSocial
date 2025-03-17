"""Main entry point for the face detection service."""
import os
import sys
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

# เพิ่ม parent directory เข้าไปใน sys.path เพื่อให้สามารถ import shared modules ได้
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ตั้งค่า logger แบบพื้นฐานก่อน
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from shared.config.settings import settings
    from shared.utils.logging import setup_logging, structured_log
    from shared.middleware.auth import APIKeyMiddleware
    from face_detection.api.routes import router as api_router
    from face_detection.models.scrfd import SCRFDDetector
    
    # ตั้งค่า logger
    logger = setup_logging()
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

# สร้าง FastAPI application
app = FastAPI(
    title="FaceSocial AI - Face Detection Service",
    description="API สำหรับตรวจจับใบหน้าด้วย GPU acceleration",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# เพิ่ม middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ในการใช้งานจริง ควรระบุ origins ที่อนุญาตเท่านั้น
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# เพิ่ม API Key middleware
app.add_middleware(APIKeyMiddleware)

# โหลดโมเดล SCRFD เมื่อเริ่มต้น server
face_detector = None

@app.on_event("startup")
async def startup_event():
    global face_detector
    try:
        # โหลดโมเดล SCRFD
        logger.info("Loading SCRFD face detector model...")
        
        # ตรวจสอบ MODEL_PATH และไฟล์โมเดล
        model_path = settings.MODEL_PATH
        logger.info(f"Model path from settings: {model_path}")
        
        model_file = os.path.join(model_path, "face-detection/scrfd/scrfd_10g_bnkps.onnx")
        logger.info(f"Complete model path: {model_file}")
        
        if os.path.exists(model_file):
            logger.info(f"Model file exists at: {model_file}")
        else:
            logger.error(f"Model file NOT found at: {model_file}")
            logger.info("Searching for model file...")
            
            # ค้นหาไฟล์โมเดลในโฟลเดอร์ models
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith('.onnx'):
                        logger.info(f"Found ONNX file: {os.path.join(root, file)}")
            
            logger.error("Cannot proceed without model file")
            return
        
        face_detector = SCRFDDetector(
            model_file=model_file,
            use_gpu=settings.USE_GPU,
            gpu_id=settings.GPU_IDS[0] if settings.GPU_IDS else 0
        )
        logger.info("SCRFD face detector model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load face detector model: {str(e)}", exc_info=e)
        # ในกรณีที่โหลดโมเดลไม่สำเร็จ ให้ server ทำงานต่อไปแต่จะส่ง error เมื่อมีการเรียกใช้งาน

# เพิ่ม middleware สำหรับวัดเวลาการตอบสนอง
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# จัดการ error ทั้งหมด
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)}
    )

# เพิ่ม routes
app.include_router(api_router, prefix="/v1/face")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "face-detection",
        "version": "1.0.0",
        "gpu_available": settings.USE_GPU,
        "model_loaded": face_detector is not None
    }

# เปิดใช้งาน server ในโหมด development
if __name__ == "__main__":
    logger.info(f"Starting Face Detection Service on 0.0.0.0:8000...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development"
    )