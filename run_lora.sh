#!/bin/bash

# Load environment variables (HF_TOKEN, etc)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export PYTHONPATH=src

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run LoRA Training with MAXIMUM memory safety (Batch 4, Accum 16)
.venv/bin/python src/chatterbox/train_lora.py \
    --resume lora_nepali_epoch_4 \
    --manifest data/chatterbox-multilingual-data/manifests/train_manifest_mtl.jsonl \
    --ckpt_dir lora_nepali_epoch_4 \
    --batch_size 8 \
    --accum_steps 8 \
    --epochs 20 \
    --lr 1e-4 \
    --num_workers 8 \
    --save_every 1 \
    --push_to_hub Firoj112/nepali_maithili_lora \
    --fp16 \
    $@
