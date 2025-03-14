FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir fastapi \
    uvicorn \
    opencv-python \
    mtcnn \
    python-multipart

# Copy application code
COPY . /app/

# Expose port
EXPOSE 8000

# กำหนด working directory อีกครั้งเพื่อความมั่นใจ
WORKDIR /app

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]