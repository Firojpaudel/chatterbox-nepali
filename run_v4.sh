#!/bin/bash
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=src
.venv/bin/python3 src/chatterbox/train_multilingual_v4.py \
    --data_path data/chatterbox-multilingual-data \
    --batch_size 4 \
    --accum_steps 32 \
    --lr 2e-5 \
    --epochs 10 \
    --push_to_hub Firoj112/chatterbox-multilingual-v4-balanced \
    --resume_from t3_v4_epoch_3.pt
