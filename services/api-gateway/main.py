from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
from typing import Optional
import json

app = FastAPI(title="FaceSocial API Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# สร้าง HTTP client
client = httpx.AsyncClient()

@app.get("/")
async def read_root():
    return {"message": "Welcome to FaceSocial API Gateway"}

@app.post("/api/v1/face-detection")
async def detect_face(image: UploadFile = File(...)):
    # อ่านไฟล์ภาพและแปลงเป็น base64
    content = await image.read()
    base64_img = base64.b64encode(content).decode("utf-8")
    
    # ส่งคำขอไปยังบริการตรวจจับใบหน้า
    response = await client.post(
        "http://face-detection:5000/detect",
        json={"image": base64_img}
    )
    
    return response.json()

@app.post("/api/v1/face-recognition/compare")
async def compare_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    model_weights: Optional[str] = Form(None)
):
    # อ่านไฟล์ภาพและแปลงเป็น base64
    content1 = await image1.read()
    content2 = await image2.read()
    base64_img1 = base64.b64encode(content1).decode("utf-8")
    base64_img2 = base64.b64encode(content2).decode("utf-8")
    
    # แปลง model_weights เป็น JSON ถ้ามี
    weights = {}
    if model_weights:
        weights = json.loads(model_weights)
    
    # ส่งคำขอไปยังบริการรู้จำใบหน้า
    response = await client.post(
        "http://face-recognition:5001/compare",
        json={
            "image1": base64_img1,
            "image2": base64_img2,
            "model_weights": weights
        }
    )
    
    return response.json()

@app.post("/api/v1/security/check")
async def security_check(
    image: UploadFile = File(...),
    checks: Optional[str] = Form("liveness,deepfake,spoofing")
):
    # อ่านไฟล์ภาพและแปลงเป็น base64
    content = await image.read()
    base64_img = base64.b64encode(content).decode("utf-8")
    
    # แยกตัวเลือกการตรวจสอบ
    check_options = checks.split(",") if checks else ["liveness", "deepfake", "spoofing"]
    
    result = {"is_real_face": True}
    
    # ตรวจสอบความมีชีวิต (liveness)
    if "liveness" in check_options:
        liveness_response = await client.post(
            "http://liveness:5002/check",
            json={"image": base64_img}
        )
        liveness_result = liveness_response.json()
        result["liveness"] = liveness_result
        if not liveness_result.get("is_live", True):
            result["is_real_face"] = False
    
    # ตรวจสอบ Deepfake
    if "deepfake" in check_options:
        deepfake_response = await client.post(
            "http://deepfake:5003/detect",
            json={"image": base64_img}
        )
        deepfake_result = deepfake_response.json()
        result["deepfake"] = deepfake_result
        if deepfake_result.get("is_fake", False):
            result["is_real_face"] = False
    
    # ตรวจสอบการปลอมแปลง (spoofing) - อาจเป็นส่วนหนึ่งของ liveness
    if "spoofing" in check_options and "liveness" not in check_options:
        spoofing_response = await client.post(
            "http://liveness:5002/check-spoofing",
            json={"image": base64_img}
        )
        spoofing_result = spoofing_response.json()
        result["spoofing"] = spoofing_result
        if spoofing_result.get("is_attack", False):
            result["is_real_face"] = False
    
    return result

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
