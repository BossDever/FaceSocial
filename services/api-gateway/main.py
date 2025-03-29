from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
from typing import Optional
import json
import datetime

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
    try:
        response = await client.post(
            "http://face-detection:5000/detect",
            json={"image": base64_img}
        )
        
        # Check Content-Type and handle JSON parsing
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                return response.json()
            except Exception as e:
                return {"error": f"JSON parsing error: {str(e)}", "raw_content": response.text[:100]}
        else:
            return {"error": f"Non-JSON response: {response.headers.get('content-type')}", "raw_content": response.text[:100]}
            
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

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
    try:
        response = await client.post(
            "http://face-recognition:5001/compare",
            json={
                "image1": base64_img1,
                "image2": base64_img2,
                "model_weights": weights
            }
        )
        
        # Check Content-Type and handle JSON parsing
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                return response.json()
            except Exception as e:
                return {"error": f"JSON parsing error: {str(e)}", "raw_content": response.text[:100]}
        else:
            return {"error": f"Non-JSON response: {response.headers.get('content-type')}", "raw_content": response.text[:100]}
            
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

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
        try:
            liveness_response = await client.post(
                "http://liveness:5002/check",
                json={"image": base64_img}
            )
            
            # Check Content-Type and handle JSON parsing
            if liveness_response.headers.get("content-type", "").startswith("application/json"):
                try:
                    liveness_result = liveness_response.json()
                except Exception as e:
                    liveness_result = {"error": f"JSON parsing error: {str(e)}", "raw_content": liveness_response.text[:100]}
            else:
                liveness_result = {"error": f"Non-JSON response: {liveness_response.headers.get('content-type')}", "raw_content": liveness_response.text[:100]}
                
        except Exception as e:
            liveness_result = {"error": f"Request failed: {str(e)}"}
        
        result["liveness"] = liveness_result
        if not liveness_result.get("is_live", True):
            result["is_real_face"] = False
    
    # ตรวจสอบ Deepfake
    if "deepfake" in check_options:
        try:
            deepfake_response = await client.post(
                "http://deepfake:5003/detect",
                json={"image": base64_img}
            )
            
            # Check Content-Type and handle JSON parsing
            if deepfake_response.headers.get("content-type", "").startswith("application/json"):
                try:
                    deepfake_result = deepfake_response.json()
                except Exception as e:
                    deepfake_result = {"error": f"JSON parsing error: {str(e)}", "raw_content": deepfake_response.text[:100]}
            else:
                deepfake_result = {"error": f"Non-JSON response: {deepfake_response.headers.get('content-type')}", "raw_content": deepfake_response.text[:100]}
                
        except Exception as e:
            deepfake_result = {"error": f"Request failed: {str(e)}"}
        
        result["deepfake"] = deepfake_result
        if deepfake_result.get("is_fake", False):
            result["is_real_face"] = False
    
    # ตรวจสอบการปลอมแปลง (spoofing) - อาจเป็นส่วนหนึ่งของ liveness
    if "spoofing" in check_options and "liveness" not in check_options:
        try:
            spoofing_response = await client.post(
                "http://liveness:5002/check-spoofing",
                json={"image": base64_img}
            )
            
            # Check Content-Type and handle JSON parsing
            if spoofing_response.headers.get("content-type", "").startswith("application/json"):
                try:
                    spoofing_result = spoofing_response.json()
                except Exception as e:
                    spoofing_result = {"error": f"JSON parsing error: {str(e)}", "raw_content": spoofing_response.text[:100]}
            else:
                spoofing_result = {"error": f"Non-JSON response: {spoofing_response.headers.get('content-type')}", "raw_content": spoofing_response.text[:100]}
                
        except Exception as e:
            spoofing_result = {"error": f"Request failed: {str(e)}"}
        
        result["spoofing"] = spoofing_result
        if spoofing_result.get("is_attack", False):
            result["is_real_face"] = False

    # ถ้ามีผลการตรวจ Liveness
    if "liveness" in result:
        if not result["liveness"].get("is_live", True):
            # ให้น้ำหนักกับผล liveness มากกว่า
            result["is_real_face"] = False
            result["primary_reason"] = "liveness_failure"

    # ถ้ามีผลการตรวจ Deepfake และยังไม่มีสาเหตุหลัก
    if "deepfake" in result and not result.get("primary_reason"):
        if result["deepfake"].get("is_fake", False):
            # ถ้า liveness ผ่าน แต่ deepfake ไม่ผ่าน
            # ให้ตรวจสอบค่า score ด้วย
            if result["deepfake"].get("score", 0) > 0.60:
                result["is_real_face"] = False
                result["primary_reason"] = "deepfake_failure"

    return result

@app.get("/api/v1/status")
async def check_services_status():
    services = {
        "face-detection": "http://face-detection:5000/health",
        "face-recognition": "http://face-recognition:5001/health",
        "liveness": "http://liveness:5002/health",
        "deepfake": "http://deepfake:5003/health",
    }
    
    results = {}
    
    for service_name, url in services.items():
        try:
            response = await client.get(url, timeout=3.0)
            
            # Check Content-Type and handle JSON parsing
            if response.headers.get("content-type", "").startswith("application/json"):
                try:
                    service_data = response.json()
                except Exception as e:
                    service_data = {"error": f"JSON parsing error: {str(e)}", "raw_content": response.text[:100]}
            else:
                service_data = {"error": f"Non-JSON response: {response.headers.get('content-type')}", "raw_content": response.text[:100]}
            
            if response.status_code == 200:
                results[service_name] = {
                    "status": "online",
                    "models": service_data.get("models", []),
                    "version": service_data.get("version", "unknown")
                }
            else:
                results[service_name] = {"status": "error", "message": f"Status code: {response.status_code}"}
        except Exception as e:
            results[service_name] = {"status": "offline", "message": str(e)}
    
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "services": results
    }

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
