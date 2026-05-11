# Chatterbox Multilingual TTS: Deployment & Integration Guide

This document provides technical instructions for integrating the Chatterbox Multilingual TTS model into production or internal company repositories.

---

## 1. Core Dependencies

To run the model, the following environment must be prepared. It is recommended to use **Python 3.10+**.

### 📦 Package Management (Recommended: uv)
The project uses `pyproject.toml` to manage all dependencies. For the fastest and most reliable setup, use [uv](https://github.com/astral-sh/uv).

```bash
# Install all dependencies as defined in pyproject.toml
uv sync

# If using pip (not recommended for production)
pip install .
```

### 📋 Key Dependencies (Defined in pyproject.toml)
- **PyTorch/Audio:** `torch>=2.4.0`, `torchaudio>=2.4.0`
- **Audio:** `librosa==0.11.0`, `numpy<2.0.0`
- **Transformers:** `transformers==5.2.0`, `diffusers==0.29.0`
- **UI:** `gradio==6.8.0`

### 🛠️ Hardware Requirements
- **Recommended:** NVIDIA GPU with 24GB VRAM (RTX 3090/4090) for simultaneous training/inference.
- **Minimum:** 8GB VRAM (inference only).
- **Precision:** Supports `bf16` (RTX 30/40 series) and `fp16`.

---

## 2. Project Structure & PYTHONPATH

The `chatterbox` logic resides in the `src` directory. To ensure the code can find its own modules without being installed as a package, always set the `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/chatterbox-nepali/src
```

---

## 3. Handling Dual Model Formats (.pt & .safetensors)

The system is designed to be format-agnostic. While production models often use `.safetensors` for security and speed, training checkpoints are generated as `.pt` files.

### 🔄 Loading Logic
When loading a checkpoint, use the following pattern to handle both types:

```python
import torch
from safetensors.torch import load_file as load_safetensors

def load_checkpoint(model, checkpoint_path):
    device = next(model.parameters()).device
    
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_safetensors(checkpoint_path, device="cpu")
    else:
        # Weights_only=True is critical for security when loading .pt files
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Strip prefixes if they exist (common in DDP/Patched training)
    clean_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in state_dict.items()}
    
    # Load into the model
    model.load_state_dict(clean_state, strict=False)
    model.to(device).eval()
    return model
```

---

## 4. Local Server Integration (No Download Required)

If you are switching repositories on the same server (e.g., from `chatterbox-nepali` to a `company-core` repo), you do **not** need to re-download checkpoints. 

### 📂 Shared Path Strategy
Update your configuration to point directly to the absolute path of the training output:

```python
CHECKPOINTS = {
    "latest-training": "/home/firojpaudel/chatterbox-nepali/t3_multilingual_epoch_2.pt"
}
```

### 🔗 Symlink Trick (Recommended)
To keep the codebase clean, create a symlink in your new repository pointing to the checkpoint directory:

```bash
ln -s /home/firojpaudel/chatterbox-nepali/t3_multilingual_epoch_2.pt ./models/chatterbox_v2.pt
```

---

## 5. Running Inference

To launch the Gradio interface with a custom process name for monitoring in `nvidia-smi`:

```bash
# 1. Create a custom-named binary in your venv
ln -sf /path/to/venv/bin/python3 /path/to/venv/bin/chatterbox-tts-inference

# 2. Launch
PYTHONPATH=src /path/to/venv/bin/chatterbox-tts-inference gradio_app.py
```

---

## 6. Training Resumption
Checkpoints are saved as `.pt` files in the root directory. To resume or fine-tune further, ensure the `src/chatterbox/train_multilingual.py` is configured to load the state dict of the previous epoch before starting the loop.

> [!IMPORTANT]
> Always verify that the `text_emb.weight` size matches your tokenizer if you have expanded the vocabulary between training runs. The `resize_text_embeddings` method in the model wrapper handles this automatically.
