# FaceSocial AI Services

ระบบบริการ AI สำหรับโครงการ FaceSocial ซึ่งเป็น microservice แยกต่างหากจาก backend หลัก

## วัตถุประสงค์

- รองรับการตรวจจับและจดจำใบหน้าที่มีประสิทธิภาพสูง
- ป้องกันการปลอมแปลงด้วยระบบ Passive Liveness Detection
- ตรวจจับภาพปลอมที่สร้างจาก AI (Deepfake) ด้วย passive security
- จัดการ face embeddings ในฐานข้อมูลเวกเตอร์
- ปรับปรุงโมเดลอย่างต่อเนื่องจากข้อมูลใหม่
- ให้บริการ API ที่ปลอดภัยและมีประสิทธิภาพสำหรับระบบอื่นๆ
- รองรับการประมวลผลภาพใบหน้าจำนวนมากพร้อมกัน (batch processing)

## เทคโนโลยีหลัก

- **ภาษาและเฟรมเวิร์ค**: Python 3.10.13, FastAPI 0.109.0, Uvicorn 0.27.0
- **AI และ Computer Vision**: ONNX Runtime 1.15.1, TensorRT 8.6, PyTorch 2.3.0+, Insightface library
- **ฐานข้อมูล**: Milvus 2.3.4, Redis
- **Containerization**: Docker, NVIDIA Container Toolkit
- **Hardware Acceleration**: CUDA 12.8, cuDNN 8.9.5
- **Parallel Processing**: Ray 2.9.0

## การติดตั้ง

(จะเพิ่มคำแนะนำการติดตั้งที่นี่)

## การใช้งาน

(จะเพิ่มคำแนะนำการใช้งานที่นี่)

## การพัฒนา

(จะเพิ่มคำแนะนำสำหรับนักพัฒนาที่นี่)