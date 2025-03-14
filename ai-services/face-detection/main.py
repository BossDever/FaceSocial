from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="FaceSocial Face Detection Service",
    description="API for detecting faces in images",
    version="0.1.0",
)

app.include_router(router, prefix="/v1/face")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)