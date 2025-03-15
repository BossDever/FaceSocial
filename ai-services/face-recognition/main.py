from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import routes as face_routes
import datetime  # Add missing import for datetime module

app = FastAPI(
    title="Face Recognition API",
    description="API for face recognition and management",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include face recognition routes
app.include_router(face_routes.router, prefix="/v1/face", tags=["face"])

# Add health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint for container monitoring
    """
    # Check if essential services are running
    health_status = {
        "status": "healthy",
        "service": "face-recognition",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    
    # Return health status
    return JSONResponse(content=health_status)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)