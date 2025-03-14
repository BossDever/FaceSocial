from setuptools import setup, find_packages

setup(
    name="FaceSocial-AI-Services",
    version="0.1.0",
    description="AI Services for FaceSocial Project",
    author="Your Team",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "tensorflow>=2.16.1",
        "opencv-python>=4.8.1",
        "dlib>=19.24.2",
        "numpy>=1.24.3",
        "scipy>=1.11.3",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "efficientnet>=1.1.1",
        "pymilvus>=2.3.4",
        "redis>=4.6.0",
        "celery>=5.3.4",
    ],
    python_requires=">=3.10",
)docker-test.sh