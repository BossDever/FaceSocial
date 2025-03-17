import logging
import sys
from loguru import logger
import json
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..config.settings import settings

# กำหนดรูปแบบการบันทึก log
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

# สร้าง class สำหรับบันทึก log ที่มีโครงสร้าง
class LogRecord(BaseModel):
    timestamp: str
    level: str
    message: str
    service: str
    method: Optional[str]
    path: Optional[str]
    request_id: Optional[str]
    user_id: Optional[str]
    duration_ms: Optional[float]
    status_code: Optional[int]
    error: Optional[Dict[str, Any]]
    extra: Optional[Dict[str, Any]]

# ฟังก์ชันสำหรับการตั้งค่า logger
def setup_logging():
    # กำหนดระดับ log จาก settings
    log_level = settings.LOG_LEVEL.upper()
    
    # ลบ default handlers
    logger.remove()
    
    # เพิ่ม handler สำหรับแสดงผลในคอนโซล
    logger.add(
        sys.stdout,
        format=LOG_FORMAT,
        level=log_level,
        colorize=True
    )
    
    # เพิ่ม handler สำหรับบันทึกลงไฟล์
    logger.add(
        f"/app/logs/facesocial_{settings.ENVIRONMENT}.log",
        format=LOG_FORMAT,
        level=log_level,
        rotation="10 MB",
        retention="3 days"
    )
    
    # เพิ่ม handler สำหรับบันทึก log แบบ JSON สำหรับการวิเคราะห์
    logger.add(
        f"/app/logs/facesocial_{settings.ENVIRONMENT}_json.log",
        format="{message}",
        level=log_level,
        rotation="10 MB",
        retention="3 days",
        serialize=True
    )
    
    # Override default logging to use loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    
    # แจ้งว่าได้ตั้งค่า logger เรียบร้อยแล้ว
    logger.info(f"Logging setup completed with level: {log_level}")
    return logger

# Handler สำหรับเชื่อมต่อ standard logging เข้ากับ loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

# ฟังก์ชันสำหรับสร้าง log ที่มีโครงสร้าง
def structured_log(
    level: str,
    message: str,
    service: str,
    method: Optional[str] = None,
    path: Optional[str] = None,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
    status_code: Optional[int] = None,
    error: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None
):
    record = LogRecord(
        timestamp=datetime.now().isoformat(),
        level=level,
        message=message,
        service=service,
        method=method,
        path=path,
        request_id=request_id,
        user_id=user_id,
        duration_ms=duration_ms,
        status_code=status_code,
        error=error,
        extra=extra
    )
    
    log_dict = record.dict()
    log_method = getattr(logger, level.lower())
    log_method(json.dumps(log_dict))