import os
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional

class Settings(BaseSettings):
    # API Settings
    API_VERSION: str = "v1"
    APP_NAME: str = "FaceSocial AI Services"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Security Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-for-development-only")
    API_KEY_HEADER: str = "X-API-KEY"
    API_KEYS: List[str] = [os.getenv("API_KEY", "development-api-key")]
    
    # Milvus Settings
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_USER: str = os.getenv("MILVUS_USER", "")
    MILVUS_PASSWORD: str = os.getenv("MILVUS_PASSWORD", "")
    
    # Redis Settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # Model Settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/home/suwit/FaceSocial/ai-services/models")
    USE_TENSORRT: bool = os.getenv("USE_TENSORRT", "True").lower() == "true"
    PRECISION: str = os.getenv("PRECISION", "fp16")  # fp16, fp32, int8
    
    # GPU Settings
    USE_GPU: bool = os.getenv("USE_GPU", "True").lower() == "true"
    GPU_IDS: List[int] = [int(id) for id in os.getenv("GPU_IDS", "0").split(",")]
    
    # Face Recognition Settings
    FACE_RECOGNITION_MODELS: Dict[str, float] = {
        "facenet": 0.35,
        "arcface": 0.35,
        "elasticface": 0.20,
        "adaface": 0.10
    }
    
    # Threshold Settings
    FACE_DETECTION_THRESHOLD: float = float(os.getenv("FACE_DETECTION_THRESHOLD", "0.5"))
    FACE_RECOGNITION_THRESHOLD: float = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.6"))
    LIVENESS_THRESHOLD: float = float(os.getenv("LIVENESS_THRESHOLD", "0.7"))
    DEEPFAKE_THRESHOLD: float = float(os.getenv("DEEPFAKE_THRESHOLD", "0.8"))
    
    # Batch Processing Settings
    DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "16"))
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "64"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create singleton settings instance
settings = Settings()