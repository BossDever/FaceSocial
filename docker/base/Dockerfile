# ใช้ image จาก NVIDIA ที่มาพร้อมกับ CUDA 11.8
FROM nvcr.io/nvidia/pytorch:22.12-py3

# ตั้งค่า labels
LABEL maintainer="FaceSocial AI Team"
LABEL version="11.8"
LABEL description="Base image for FaceSocial AI services with CUDA 11.8 and PyTorch"

# ตั้งค่า environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"

# อัพเดท apt และติดตั้ง dependencies ที่จำเป็น
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# อัพเกรด pip และติดตั้ง packages พื้นฐาน
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# สร้างโฟลเดอร์สำหรับ application
WORKDIR /app

# สร้าง non-root user เพื่อความปลอดภัยยิ่งขึ้น
RUN groupadd -r facesocial && useradd -r -g facesocial facesocial
RUN chown -R facesocial:facesocial /app

# ตรวจสอบเวอร์ชัน CUDA, cuDNN, และ PyTorch
RUN nvcc --version && \
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'Not available'); print('cuDNN version:', torch.backends.cudnn.version() if torch.cuda.is_available() else 'Not available')"

# ติดตั้ง ONNX Runtime ที่เข้ากับ CUDA 11.8
RUN pip install --no-cache-dir onnxruntime-gpu==1.14.1

# ตรวจสอบ ONNX Runtime providers
RUN python -c "import onnxruntime as ort; print('ONNX Runtime version:', ort.__version__); print('Available providers:', ort.get_available_providers())"

# สร้าง volume สำหรับ model storage
VOLUME ["/app/models"]

# คง entrypoint เดิมของ NVIDIA PyTorch container
ENTRYPOINT ["bash", "-c"]
CMD ["bash"]