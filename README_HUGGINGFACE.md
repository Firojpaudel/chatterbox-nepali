---
language:
  - ne
  - en
  - hi
license: mit
tags:
  - text-to-speech
  - tts
  - nepali
  - chatterbox
  - fine-tuned
  - llama
  - safetensors
  - audio
datasets:
  - Firoj112/voxcpm-nepali-data
base_model: ResembleAI/chatterbox
pipeline_tag: text-to-speech
library_name: chatterbox
---

# Chatterbox Nepali TTS — Fine-tuned & Merged

Fine-tuned weights for the [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) multilingual TTS model, adapted for high-fidelity **Nepali (Devanagari)** speech synthesis. The final model applies a **linear weight interpolation (model merging)** strategy to recover cross-lingual stability after fine-tuning.

---

## Why This Exists

The base Chatterbox model supports 23 languages but does **not** include Nepali. This repository contains:

1. The fine-tuned T3 backbone trained on ~8,000 Nepali utterances.
2. A merged variant that blends the fine-tuned and base weights to restore English prosody (see [Weight Merging](#weight-merging-mathematics) below).
3. Intermediate epoch checkpoints for regression testing and research.

---

## Model Architecture

Chatterbox is a **two-stage cascaded TTS system**. Only the first stage (T3) was fine-tuned. The vocoder (S3Gen) is unchanged from the base model.

### Stage 1: T3 — Text-to-Speech Token LLM

The T3 model is a **causal language model** built on a Llama-style transformer that autoregressively predicts discrete speech tokens from text tokens.

```
T3 Architecture
├── Backbone:         LlamaModel (30 decoder layers, d_model=1024)
│   ├── Attention:    Multi-head self-attention (1024→1024, 8 heads, RoPE)
│   └── FFN:          SwiGLU (1024 → 4096 → 1024)
│
├── Input Embeddings:
│   ├── text_emb:     Embedding(2455 tokens, 1024)   ← Language tags live here
│   ├── speech_emb:   Embedding(8194 codes, 1024)
│   ├── text_pos_emb: Learned positional (2050 slots)
│   └── speech_pos_emb: Learned positional (4100 slots)
│
├── Conditioning (T3CondEnc):
│   ├── spkr_enc:     Linear(256 → 1024)             ← Voice encoder embedding
│   ├── emotion_adv:  Linear(1 → 1024)               ← Exaggeration scalar
│   └── perceiver:    Cross-attention Perceiver (compresses prompt speech tokens)
│
└── Output Heads:
    ├── text_head:    Linear(1024 → 2455)
    └── speech_head:  Linear(1024 → 8194)             ← This is what we train
```

**Key numbers:**
| Parameter | Value |
| :--- | :--- |
| Transformer layers | 30 |
| Hidden dimension | 1024 |
| Text vocabulary | 2,455 tokens (includes `[ne]` at index 2454) |
| Speech codebook size | 8,194 codes |
| Positional encoding | Learned embeddings (not RoPE for speech) |
| Total T3 parameters | ~534M |

### Stage 2: S3Gen — Vocoder (Unchanged)

The S3Gen converts the discrete speech tokens back into a waveform. It consists of:
- **S3Tokenizer**: An audio encoder (Conv1d + 6 ResidualAttentionBlocks at 1280-dim) with an FSQ vector quantizer that produces 6,561 unique speech codes at 25Hz.
- **Speaker Encoder (CAMPPlus)**: A CAM++ x-vector model that extracts 256-dim speaker embeddings from a reference audio clip.
- **Waveform Decoder**: A flow-matching or diffusion-based decoder that synthesizes 24kHz audio.

### Inference Flow

```
Text (Nepali/English)
       │
       ▼
  MTLTokenizer ──► [ne] tag + grapheme tokens
       │
       ▼
    T3 LLM  ◄── Speaker Embedding (VoiceEncoder)
       │         ◄── Prompt Speech Tokens (Perceiver)
       │         ◄── Exaggeration Scalar
       ▼
 Speech Tokens (25Hz discrete codes)
       │
       ▼
   S3Gen Vocoder ◄── Reference audio (CAMPPlus speaker)
       │
       ▼
  Audio Waveform (24kHz)
```

---

## Nepali Language Support — How It Was Added

### 1. Vocabulary Expansion

The base multilingual grapheme tokenizer was extended with all Devanagari characters and combined Nepali grapheme clusters using a custom mapping file (`grapheme_mtl_merged_expanded_v1.json`). A new language tag `[ne]` was registered at index **2454** in the text embedding table.

### 2. Weight Transplantation (Cold-Start)

Instead of randomly initializing the `[ne]` embedding, we copied the weights from the `[hi]` (Hindi) embedding:

```python
# In train_nepali.py — initialization step
ne_idx = tokenizer.lang_to_idx["ne"]   # 2454
hi_idx = tokenizer.lang_to_idx["hi"]   # 722

with torch.no_grad():
    model.t3.text_emb.weight[ne_idx] = model.t3.text_emb.weight[hi_idx].clone()
```

**Why Hindi?** Nepali and Hindi share the same Devanagari script, similar phoneme inventory, and close linguistic roots. This gave the model a meaningful starting point rather than noise, which measurably accelerated early convergence.

### 3. Training Configuration

| Parameter | Value |
| :--- | :--- |
| Base model | `ResembleAI/chatterbox` |
| Dataset | `Firoj112/voxcpm-nepali-data` (~8,000 clips) |
| Hardware | 2x NVIDIA T4 (Kaggle, 30GB total VRAM) |
| Training mode | Distributed Data Parallel (DDP) |
| Precision | Mixed Precision FP16 |
| Optimizer | AdamW, lr=2e-5 |
| Gradient Checkpointing | Enabled |
| Batch size | 4 per GPU |
| Total epochs | 50 |
| Checkpoint frequency | Every 5 epochs |

### 4. Training Loss Progression

| Checkpoint | Epoch | Approx. Loss |
| :--- | :--- | :--- |
| `t3_nepali_epoch_20.pt` | 20 | ~6.7 |
| `t3_nepali_epoch_30.pt` | 30 | ~2.1 |
| `t3_nepali_epoch_35.pt` | 35 | ~1.0 |
| `t3_nepali_epoch_40.pt` | 40 | ~0.45 |
| `t3_nepali_epoch_45.pt` | 45 | ~0.21 |
| `t3_mtl_nepali_final.safetensors` | 50 | ~0.11 |

---

## Weight Merging: Mathematics

After 50 epochs of fine-tuning, the model produced excellent Nepali speech but suffered from **Catastrophic Forgetting** — its ability to speak English with native prosody had degraded significantly. The English output was heavily Nepali-accented.

### The Problem

Neural networks store knowledge as distributed weight patterns. When you fine-tune a model exclusively on a new language/domain, gradient updates systematically overwrite the previously learned weight patterns for other tasks. This is not a bug — it is a fundamental property of gradient descent on non-stationary distributions.

### The Solution: Linear Weight Interpolation

We applied **linear interpolation** (also called a linear merge or model soups) between the original base weights and the fine-tuned weights:

$$W_{\text{merged}} = (1 - \alpha) \cdot W_{\text{base}} + \alpha \cdot W_{\text{fine-tuned}}$$

where $\alpha \in [0, 1]$ is the merge ratio.

**We chose $\alpha = 0.7$**, meaning:
- **70%** of the final weight comes from the fine-tuned Nepali model
- **30%** comes from the original base multilingual model

This was implemented in `merge_models.py`:

```python
def merge_models(base_path, fine_tuned_path, output_path, alpha=0.7):
    """
    W_merged = (1 - alpha) * W_base + alpha * W_fine
    """
    base_sd   = load_file(base_path)
    fine_sd   = load_file(fine_tuned_path)
    merged_sd = {}

    for key in set(fine_sd.keys()) | set(base_sd.keys()):
        if key in fine_sd and key in base_sd:
            w_base = base_sd[key]
            w_fine = fine_sd[key]

            if w_base.shape == w_fine.shape:
                # Standard interpolation for matching dimensions
                merged_sd[key] = (1.0 - alpha) * w_base + alpha * w_fine
            else:
                # Handle vocabulary expansion (embedding tables grow for new language tags)
                new_w = w_fine.clone()
                common_rows = min(w_base.shape[0], w_fine.shape[0])
                common_cols = min(w_base.shape[1], w_fine.shape[1])
                new_w[:common_rows, :common_cols] = (
                    (1.0 - alpha) * w_base[:common_rows, :common_cols]
                    + alpha * w_fine[:common_rows, :common_cols]
                )
                merged_sd[key] = new_w
        elif key in fine_sd:
            merged_sd[key] = fine_sd[key]   # New keys (e.g., [ne] embedding)
        else:
            merged_sd[key] = base_sd[key]   # Base-only keys

    save_file(merged_sd, output_path)
```

### Why $\alpha = 0.7$?

The ratio was selected empirically by listening to output quality:

| Alpha | Nepali Quality | English Prosody |
| :--- | :--- | :--- |
| 1.0 | Excellent | Heavy Nepali accent (broken) |
| 0.8 | Very Good | Slight Nepali accent |
| **0.7** | **Good** | **Near-native English** |
| 0.5 | Acceptable | Native English |
| 0.0 | No Nepali | Native English (base model) |

At $\alpha = 0.7$, the model generates:
- Fluent Nepali with correct Devanagari phonetics and natural rhythm
- English with prosody close enough to the original that it is clearly intelligible and accent-neutral

### The Geometry of Merging

Intuitively, each weight tensor in the network lives in a high-dimensional parameter space. Both the base and fine-tuned models sit in the same loss basin (since the fine-tuned model started from the base). Linear interpolation walks a straight line between these two points in weight space. Because they share the same basin, all intermediate points also produce a valid, functional model — this is the key insight from the "Model Soups" paper (Wortsman et al., 2022).

---

## Inference Stability: AlignmentStreamAnalyzer

Nepali synthesis has a specific failure mode: the model sometimes fails to predict the EOS (End-of-Speech) token after finishing the utterance, producing repetitive gibberish at the tail of the generation.

We implemented and tuned the `AlignmentStreamAnalyzer` to fix this. It works by:

1. **Monitoring cross-attention maps** during each generation step.
2. **Estimating the text pointer** — the position in the input text that the model is currently attending to.
3. **Detecting alignment saturation** — when the pointer stops advancing and remains stuck at the last grapheme for several steps.
4. **Forcefully terminating generation** and trimming the output at that point.

This heuristic is particularly important for Devanagari, where grapheme clusters can cause alignment drift due to the conjunct character system.

---

## Repository Contents

| File | Size | Description |
| :--- | :--- | :--- |
| `t3_mtl_nepali_merged.safetensors` | 2.14 GB | **Recommended.** 0.7/0.3 blend of FT and base. |
| `t3_mtl_nepali_final.safetensors` | 2.14 GB | Raw Epoch 50 weights (best Nepali, broken English). |
| `t3_nepali_epoch_20.pt` | 2.14 GB | Checkpoint at epoch 20. |
| `t3_nepali_epoch_30.pt` | 2.14 GB | Checkpoint at epoch 30. |
| `t3_nepali_epoch_35.pt` | 2.14 GB | Checkpoint at epoch 35. |
| `t3_nepali_epoch_40.pt` | 2.14 GB | Checkpoint at epoch 40. |
| `t3_nepali_epoch_45.pt` | 2.14 GB | Checkpoint at epoch 45. |
| `merge_models.py` | — | Utility script for weight interpolation. |

---

## Loading the Model

```python
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load base model structure (downloads from ResembleAI/chatterbox)
model = ChatterboxMultilingualTTS.from_pretrained(device)

# 2. Overlay the merged Nepali weights
state_dict = load_file("t3_mtl_nepali_merged.safetensors")
clean_state = {
    k.replace("patched_model.", "").replace("model.", ""): v
    for k, v in state_dict.items()
}
model.t3.load_state_dict(clean_state, strict=False)
model.t3.eval()

# 3. Generate Nepali speech
wav = model.generate(
    "नमस्ते, म नेपालीमा पनि राम्रोसँग बोल्न सक्छु।",
    language_id="ne",
    audio_prompt_path="path/to/nepali_reference.wav",
    cfg_weight=0.5,
    temperature=0.8,
)

import torchaudio
torchaudio.save("output.wav", wav, model.sr)
```

---

## Reproducing the Merge

```bash
# Download the fine-tuned model first
python merge_models.py \
  --fine t3_mtl_nepali_final.safetensors \
  --output t3_mtl_nepali_merged.safetensors \
  --alpha 0.7
# The script auto-downloads the base model from ResembleAI/chatterbox
```

---

## Known Limitations

- The merged model may produce a slight Hindi-adjacent accent in Nepali output, particularly on retroflex consonants. This is a consequence of the Hindi weight initialization and the 50-epoch training budget.
- Very long inputs (>200 characters) should be chunked at sentence boundaries before generation.
- The `language_id="ne"` tag must be explicitly set; auto-detection is not implemented.

---

## Citation & Credits

- **Fine-tuned by**: Firoj Paudel
- **Base Model**: [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox)
- **Dataset**: [Firoj112/voxcpm-nepali-data](https://huggingface.co/datasets/Firoj112/voxcpm-nepali-data)
- **Merging reference**: Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time." ICML 2022.
