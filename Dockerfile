# Use an NVIDIA CUDA base image with Python 3 and CUDA runtime
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install system dependencies and essential tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    wget \
    unzip \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy project files into the container
COPY . /app
WORKDIR /app

# Install project dependencies from requirements.txt
RUN pip install -r requirements.txt || \
    (wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/nvidia-tensorrt.list && \
    apt-get update && apt-get install -y \
    libnvinfer8 \
    libnvinfer-dev \
    libnvinfer-plugin8 \
    python3-libnvinfer)


# Install pinned PyTorch, TorchVision, TorchAudio, Torch-TensorRT for CUDA 11.8
RUN pip install \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    # torchaudio==2.0.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install torch-tensorrt==1.4.0 \
    -f https://github.com/NVIDIA/Torch-TensorRT/releases


# Download styling models
RUN chmod +x ./download_styling_models.sh && ./download_styling_models.sh

# Set the working directory for input/output
WORKDIR /data

# Set the container's entry point
ENTRYPOINT ["python3", "/app/neural_style/neural_style.py"]

