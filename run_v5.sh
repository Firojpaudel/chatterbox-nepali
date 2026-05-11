#!/bin/bash

# Load environment variables (WANDB_API_KEY, HF_TOKEN, etc)
export $(grep -v '^#' .env | xargs)

export PYTHONPATH=src
export WANDB_PROJECT="chatterbox-multilingual-v5"

# Run V5 Training
.venv/bin/python3 src/chatterbox/train_multilingual_v5.py \
    --data_path data/chatterbox-multilingual-data \
    --batch_size 4 \
    --accum_steps 32 \
    --epochs 20 \
    --push_to_hub "Firoj112/chatterbox-multilingual-t3-v5" \
    --resume_from t3_v5_epoch_3.pt \
    $@

# Note: Added $@ to allow passing --resume_from dynamically if needed
