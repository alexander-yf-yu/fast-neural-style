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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy project files into the container
COPY . /app
WORKDIR /app

# Install project dependencies and download styling models
RUN pip install -r requirements.txt
RUN chmod +x ./download_styling_models.sh && ./download_styling_models.sh

# Ensure CUDA compatibility with PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set the working directory for input/output
WORKDIR /data
