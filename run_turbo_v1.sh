#!/bin/bash

# Training configuration for RTX 4090
DATA_PATH="data/chatterbox-multilingual-data"
EVAL_AUDIO="samples/Prakash.mp3"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source .venv/bin/activate

python src/chatterbox/train_turbo_v1.py \
    --data_path "$DATA_PATH" \
    --batch_size 8 \
    --accum_steps 4 \
    --epochs 30 \
    --lr 5e-5 \
    --bf16 \
    --speech_loss_weight 5.0 \
    --text_loss_weight 1.0 \
    --save_every 1 \
    --eval_audio_path "$EVAL_AUDIO" \
    --lora_r 64 \
    --lora_alpha 128 \
    --num_workers 8 \
    --hub_repo "Firoj112/chatterbox_nep_mai_turbo_LoRA"
# To resume, add: --init_lora_path "turbo_lora_epoch_0"
