# Training V3 — Layer Freezing + Stable LR

## What Changed from V2

| Setting | V2 | V3 | Why |
|---|---|---|---|
| `lr` | `3e-5` | `5e-6` | 6x lower — prevents destroying pretrained acoustics |
| `lr_schedule` | `cosine` | `constant` | Constant LR after warmup is more stable for fine-tuning |
| `warmup_steps` | `500` | `200` | Shorter warmup since LR is lower |
| `freeze_layers` | `0` (all trainable) | `15` (half frozen) | Preserves universal acoustic foundation |
| `speech_loss_weight` | `1.0` | `3.0` | Forces model to focus on speech quality over text |
| `text_loss_weight` | `1.0` | `1.0` | Same — text learning is already fast |
| Checkpoint prefix | `t3_multilingual_epoch_` | `t3_v3_epoch_` | Distinguishes V3 checkpoints |

## Why These Changes Fix the Problems

1. **No more hallucination**: Lower LR + frozen layers = model doesn't lose acoustic coherence
2. **No more noise**: Speech-weighted loss forces acoustic quality over text memorization
3. **English preserved**: Bottom 15 layers (universal acoustics) are frozen — English stays intact
4. **Nepali/Maithili improved**: Top 15 layers still learn language-specific patterns
5. **No robotic voice**: Constant LR avoids the "too hot for too long" problem of cosine decay

## Step 1: Kill any leftover processes

```bash
pkill -f "chatterbox.train_multilingual" 2>/dev/null
```

## Step 2: Launch in tmux

### Option A: Fresh start from base model (recommended)

```bash
tmux new-session -d -s train_v3 -c /home/firojpaudel/chatterbox-nepali \
  'ln -sf /home/firojpaudel/chatterbox-nepali/.venv/bin/python3 /home/firojpaudel/chatterbox-nepali/.venv/bin/chatterbox-tts-train && \
   PYTHONPATH=/home/firojpaudel/chatterbox-nepali/src /home/firojpaudel/chatterbox-nepali/.venv/bin/chatterbox-tts-train -m chatterbox.train_multilingual_v3 \
   --repo_id Firoj112/chatterbox-multilingual-data \
   --push_to_hub Firoj112/chatterbox-multilingual-t3-v1-4090 \
   --epochs 20 \
   --batch_size 4 \
   --accum_steps 4 \
   --lr 5e-6 \
   --lr_schedule constant \
   --warmup_steps 200 \
   --min_lr 1e-6 \
   --weight_decay 0.01 \
   --max_grad_norm 1.0 \
   --freeze_layers 15 \
   --speech_loss_weight 3.0 \
   --text_loss_weight 1.0 \
   --estimated_steps_per_epoch 6000 \
   --bf16 \
   --save_every 2 \
   --num_workers 2 \
   --device cuda 2>&1 | tee training_v3.log'
```

### Option B: Resume from V2 epoch-6 checkpoint

```bash
tmux new-session -d -s train_v3 -c /home/firojpaudel/chatterbox-nepali \
  'ln -sf /home/firojpaudel/chatterbox-nepali/.venv/bin/python3 /home/firojpaudel/chatterbox-nepali/.venv/bin/chatterbox-tts-train && \
   PYTHONPATH=/home/firojpaudel/chatterbox-nepali/src /home/firojpaudel/chatterbox-nepali/.venv/bin/chatterbox-tts-train -m chatterbox.train_multilingual_v3 \
   --repo_id Firoj112/chatterbox-multilingual-data \
   --push_to_hub Firoj112/chatterbox-multilingual-t3-v1-4090 \
   --resume_from t3_multilingual_epoch_6.pt \
   --epochs 20 \
   --batch_size 4 \
   --accum_steps 4 \
   --lr 5e-6 \
   --lr_schedule constant \
   --warmup_steps 200 \
   --min_lr 1e-6 \
   --weight_decay 0.01 \
   --max_grad_norm 1.0 \
   --freeze_layers 15 \
   --speech_loss_weight 3.0 \
   --text_loss_weight 1.0 \
   --estimated_steps_per_epoch 6000 \
   --bf16 \
   --save_every 2 \
   --num_workers 2 \
   --device cuda 2>&1 | tee training_v3.log'
```

## Step 3: Monitor

```bash
# Attach to the session
tmux attach -t train_v3

# Or tail the log
tail -f /home/firojpaudel/chatterbox-nepali/training_v3.log
```

## What to Look For in V3 Logs

### Good signs:
- `L_txt` stays in the **0.3-0.8** range (not dropping below 0.1 like V2)
- `L_spch` steadily decreasing
- No sudden loss spikes

### Red flags:
- `L_txt` drops below 0.1 → text overfitting (shouldn't happen with speech weighting)
- `L_spch` stalls completely → frozen too many layers (try `--freeze_layers 10`)
- Loss goes up → LR too high (try `--lr 2e-6`)

## Recommended: Fresh Start (Option A)

Starting fresh from the base model with V3 settings is recommended because:
1. The V2 checkpoint already has some catastrophic forgetting damage
2. With proper settings, V3 will reach the same point in ~4-6 epochs but with much cleaner output
3. The frozen layers ensure the acoustic foundation is never damaged
