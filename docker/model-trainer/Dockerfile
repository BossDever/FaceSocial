FROM facesocial-base:12.8

LABEL maintainer="FaceSocial AI Team"
LABEL version="1.0"
LABEL description="Model Training service for FaceSocial"

# ติดตั้ง dependencies สำหรับการฝึกโมเดล
RUN pip install --no-cache-dir \
    onnx==1.15.0 \
    onnxruntime-gpu==1.15.1 \
    facenet-pytorch \
    insightface==0.7.1 \
    timm==0.9.5 \
    torch-optimizer==0.3.0 \
    albumentations==1.3.1 \
    wandb==0.16.0 \
    tensorboard==2.15.1 \
    PyYAML==6.0.1 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.3 \
    seaborn==0.13.0 \
    tqdm==4.66.1

# ติดตั้ง NVIDIA Apex สำหรับ mixed precision training
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ && \
    cd .. && rm -rf apex

# สร้างโฟลเดอร์ที่จำเป็นสำหรับการฝึกโมเดล
RUN mkdir -p /app/model-trainer/logs /app/model-trainer/checkpoints /app/model-trainer/outputs

# คัดลอก application code
COPY . /app/

# เปลี่ยนเป็น non-root user
USER facesocial

# ตั้งค่า entrypoint และคำสั่งเริ่มต้น
CMD ["python /app/model-trainer/main.py"]