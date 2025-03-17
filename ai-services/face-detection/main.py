"""Main entry point for the face detection service."""
import os
import sys
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

# เพิ่ม parent directory เข้าไปใน sys.path เพื่อให้สามารถ import shared modules ได้
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config.settings import settings
from shared.utils.logging import setup_logging, structured_log
from shared.middleware.auth import APIKeyMiddleware

from api.routes import router as api_router
from models.scrfd import SCRFDDetector

# ตั้งค่า logger
logger = setup_logging()

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
        face_detector = SCRFDDetector(
            model_file=os.path.join(settings.MODEL_PATH, "face-detection/scrfd/scrfd_10g_bnkps.onnx"),
            use_gpu=settings.USE_GPU,
            gpu_id=settings.GPU_IDS[0] if settings.GPU_IDS else 0
        )
        logger.info("SCRFD face detector model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load face detector model: {str(e)}")
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
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development"
    )