from fastapi import Request, HTTPException, Depends, status
from fastapi.security.api_key import APIKeyHeader
from typing import Optional, List

from ..config.settings import settings

# Define API key header
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)

async def get_api_key(
    api_key_header: Optional[str] = Depends(api_key_header),
) -> str:
    """
    ตรวจสอบ API key ที่ส่งมาในส่วนหัวของ HTTP request
    
    Args:
        api_key_header: API key จาก HTTP header
        
    Returns:
        str: API key ที่ผ่านการตรวจสอบแล้ว
    
    Raises:
        HTTPException: ถ้า API key ไม่ถูกต้อง
    """
    if api_key_header in settings.API_KEYS:
        return api_key_header
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": settings.API_KEY_HEADER},
    )

class APIKeyMiddleware:
    """
    Middleware สำหรับตรวจสอบ API key ในทุก request
    
    Note:
        ใช้เป็น middleware ทั่วไปที่ไม่ต้องการละเอียดเท่ากับการใช้ Depends
    """
    
    def __init__(self, allowed_paths: List[str] = None):
        """
        Initialize middleware
        
        Args:
            allowed_paths: รายการ path ที่ไม่ต้องใช้ API key (เช่น health check)
        """
        self.allowed_paths = allowed_paths or ["/health", "/docs", "/redoc", "/openapi.json"]
    
    async def __call__(self, request: Request, call_next):
        """
        Process HTTP request
        
        Args:
            request: FastAPI Request object
            call_next: handler ถัดไป
            
        Returns:
            Response จาก handler ถัดไป
        """
        # ถ้าเป็น path ที่ได้รับการยกเว้น ให้ผ่านไปยัง handler ถัดไปโดยไม่ต้องตรวจสอบ
        if any(request.url.path.startswith(path) for path in self.allowed_paths):
            return await call_next(request)
        
        # ตรวจสอบ API key
        api_key = request.headers.get(settings.API_KEY_HEADER)
        if api_key and api_key in settings.API_KEYS:
            return await call_next(request)
        
        # ถ้า API key ไม่ถูกต้อง แสดงข้อผิดพลาด
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": settings.API_KEY_HEADER},
        )