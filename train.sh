#!/usr/bin/env bash
set -eo pipefail

# Activate virtual environment
cd ~/piper/src/python/
python3 -m venv .venv
source .venv/bin/activate

# Run preprocessing
python3 -m piper_train.preprocess \
    --language en \
    --input-dir ~/dataset \
    --output-dir ~/training \
    --dataset-format ljspeech \
    --single-speaker \
    --sample-rate 22050 \
    --max-workers 1

# Download pre-trained model if not exists
if [ ! -f ~/piper/epoch=2164-step=1355540.ckpt ]; then
    wget -P ~/piper/ \
        https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt
fi

export CUDA_LAUNCH_BLOCKING=1

# Run training with CPU configuration
python3 -m piper_train \
    --dataset-dir ~/training \
    --accelerator 'gpu' \
    --devices 1 \
    --batch-size 8 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 6000 \
    --resume_from_checkpoint ~/piper/epoch=2164-step=1355540.ckpt \
    --checkpoint-epochs 1 \
    --precision 32