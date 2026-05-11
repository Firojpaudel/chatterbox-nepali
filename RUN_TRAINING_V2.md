# Training v2 — RTX 4090 Launch Guide

## Step 1: Kill any leftover processes

```bash
pkill -f "chatterbox.train_multilingual" 2>/dev/null
pkill -f "from datasets import load_dataset" 2>/dev/null
```

## Step 2: Launch in tmux

```bash
tmux new-session -d -s train -c /home/firojpaudel/chatterbox-nepali \
  'ln -sf /home/firojpaudel/chatterbox-nepali/.venv/bin/python3 /home/firojpaudel/chatterbox-nepali/.venv/bin/chatterbox-tts-train && \
   PYTHONPATH=/home/firojpaudel/chatterbox-nepali/src /home/firojpaudel/chatterbox-nepali/.venv/bin/chatterbox-tts-train -m chatterbox.train_multilingual \
   --repo_id Firoj112/chatterbox-multilingual-data \
   --push_to_hub Firoj112/chatterbox-multilingual-t3-v1-4090 \
   --epochs 20 \
   --batch_size 4 \
   --accum_steps 4 \
   --lr 3e-5 \
   --warmup_steps 500 \
   --min_lr 1e-6 \
   --weight_decay 0.01 \
   --max_grad_norm 1.0 \
   --estimated_steps_per_epoch 6000 \
   --bf16 \
   --save_every 2 \
   --num_workers 2 \
   --device cuda 2>&1 | tee training_v2.log'
```

## Step 3: Monitor

```bash
# Attach to the session
tmux attach -t train

# Or tail the log from another terminal
tail -f /home/firojpaudel/chatterbox-nepali/training_v2.log
```

## Step 4: Detach from tmux (leave training running)

Press `Ctrl+B` then `D` to detach.

---

## What's Different This Time

| Setting | Value | Why |
|---|---|---|
| `lr` | `3e-5` | Slightly lower peak, prevents noise |
| `warmup_steps` | `500` | Linear warmup 0→3e-5, smooth start |
| `min_lr` | `1e-6` | Cosine decays to this, prevents overfitting |
| `weight_decay` | `0.01` | Regularization on AdamW |
| `max_grad_norm` | `1.0` | Gradient clipping, no explosions |
| `bf16` | ✅ | Native 4090 support, more stable than fp16 |
| `save_every` | `2` | Saves at epochs 2, 4, 6... for better safety |
| `epochs` | `20` | With cosine decay, safe from overfitting |

## WandB Graphs You'll See

- **loss** — total combined loss
- **loss_text** — text prediction loss (language-specific)
- **loss_speech** — speech token loss (audio quality)
- **learning_rate** — should show warmup ramp then cosine curve
- **epoch** — epoch counter
- **epoch_avg_loss** — per-epoch average (logged at end of each epoch)
- **epoch_avg_loss_text** — per-epoch text loss average
- **epoch_avg_loss_speech** — per-epoch speech loss average
