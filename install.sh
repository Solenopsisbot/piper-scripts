#!/usr/bin/env bash
set -eo pipefail

# System dependencies
sudo apt update
sudo apt upgrade
sudo apt install -y \
    python3-dev \
    python3-venv \
    espeak \
    ffmpeg \
    build-essential \
    libsndfile1
sudo apt install -y nvidia-cuda-toolkit nvidia-cudnn

# Set up virtual environment
cd ~/
# rm -rf piper  # Remove existing installation
if [ ! -d "piper" ]; then
    git clone https://github.com/rhasspy/piper.git
fi
cd piper/src/python/
python3 -m venv .venv
source .venv/bin/activate

# Install base dependencies with specific versions
python3 -m pip install --upgrade wheel setuptools
python3 -m pip install pip==24

# Install specific NumPy version first
python3 -m pip install numpy==1.24.0

# Install PyTorch dependencies (CPU version for Raspberry Pi)
python3 -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Install ML dependencies
python3 -m pip install pytorch-lightning==1.9.4 torchmetrics==0.11.4

# Install audio processing dependencies
python3 -m pip install librosa==0.10.1 soundfile==0.12.1 numba==0.58.1

# Install other required packages
python3 -m pip install Cython==3.0.8 piper-phonemize onnxruntime

# Install Piper in development mode
python3 -m pip install -e .

# Build monotonic align extension
bash build_monotonic_align.sh