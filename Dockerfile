FROM dataloopai/dtlpy-agent:cpu.py3.10.pytorch2

USER root

RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    wget \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

USER 1000
WORKDIR /tmp
ENV HOME=/tmp

RUN pip install --upgrade pip && pip install --user \
    numpy>=1.21.0 \
    opencv-python>=4.5.0 \
    torch>=1.9.0 \
    torchvision>=0.10.0 \
    ultralytics>=8.3.152 \
    matplotlib>=3.3.0 \
    ffmpeg-python>=0.2.0 \
    munkres>=1.1.4 \
    scipy>=1.7.0 \
    Pillow>=8.3.0 \
    tqdm>=4.62.0 \
    tensorboard>=2.7.0 \
    scikit-learn>=1.0.0\
    gdown>=4.6.0

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/simple-hrnet-adapter:0.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/simple-hrnet-adapter:0.0.1
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/simple-hrnet-adapter:0.0.1 bash