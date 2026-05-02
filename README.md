<p align="center">
  <img src="Chatterbox-Multilingual.png" width="800px" alt="Chatterbox Multilingual"/>
</p>

# Chatterbox — Nepali Fine-tune

> **A community extension of [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) adding native Nepali (Devanagari) speech synthesis.**

[![Hugging Face - Weights](https://img.shields.io/badge/HuggingFace-Weights-yellow?logo=huggingface)](https://huggingface.co/Firoj112/chatterbox-nepali-runs)
[![Base Model](https://img.shields.io/badge/Base-ResembleAI%2Fchatterbox-blue?logo=github)](https://github.com/resemble-ai/chatterbox)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

This repository is a fork of the official [Chatterbox](https://github.com/resemble-ai/chatterbox) TTS system by [Resemble AI](https://resemble.ai). It extends the multilingual model (`ChatterboxMultilingualTTS`) with Nepali language support through supervised fine-tuning on native Devanagari speech data.

**All fine-tuned model weights are hosted on Hugging Face:**
**[Firoj112/chatterbox-nepali-runs](https://huggingface.co/Firoj112/chatterbox-nepali-runs)**

---

## What's Different from Upstream

| Feature | Upstream (`resemble-ai/chatterbox`) | This Fork |
| :--- | :--- | :--- |
| Languages | 23 languages | 23 + **Nepali (`ne`)** |
| Nepali weights | Not included | Fine-tuned for high-fidelity native speech |
| Hallucination fix | Standard EOS | `AlignmentStreamAnalyzer` + Out-of-bounds token filtering |
| Gradio app | Standard | Nepali-focused UI (`gradio_app.py`) |
| Text sanitization | English-only | Language-aware (Nepali & English support) |

---

## Key Features & Stability

This fork introduces several critical improvements over the base repository to ensure production stability:

1. **Nepali-Centric Sanitization**: A custom normalization layer that correctly expands Nepali numbers, dates, and currency while handling Devanagari punctuation.
2. **Language-Aware Processing**: The sanitizer now intelligently switches between Nepali and English rules—preserving apostrophes and hyphens in English while applying aggressive normalization to Nepali to prevent hallucinations.
3. **Inference Stability (CUDA Fix)**: Implemented a robust token-filtering layer that strips "out-of-bounds" hallucinations. This prevents the `device-side assert` crashes common when models generate unexpected high-index tokens.
4. **Hallucination Prevention**: Integrated `AlignmentStreamAnalyzer` to force termination of speech when the model enters an infinite repetition loop, a common issue in Devanagari synthesis.

---

## Quick Start

### Installation

```bash
# Clone this fork
git clone https://github.com/Firojpaudel/chatterbox-nepali.git
cd chatterbox-nepali
pip install -e .
```

### Download Weights from Hugging Face

```python
from huggingface_hub import hf_hub_download

# Download the final model (default — best Nepali quality)
hf_hub_download(
    repo_id="Firoj112/chatterbox-nepali-runs",
    filename="t3_mtl_nepali_final.safetensors",
    local_dir="."
)
```

### Generate Nepali Speech

```python
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model structure
model = ChatterboxMultilingualTTS.from_pretrained(device)

# Load Nepali-tuned weights
state_dict = load_file("t3_mtl_nepali_final.safetensors")
clean_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in state_dict.items()}
model.t3.load_state_dict(clean_state, strict=False)
model.t3.eval()

# Synthesize
wav = model.generate(
    "नमस्ते, म नेपालीमा पनि राम्रोसँग बोल्न सक्छु।",
    language_id="ne",
    audio_prompt_path="samples/Prakash.mp3",  # reference voice
    cfg_weight=0.5,
    temperature=0.8,
)
ta.save("output.wav", wav, model.sr)
```

### Run the Gradio App

```bash
# Full-featured Nepali TTS app (recommended)
python gradio_app.py

# Or use the upstream multilingual app (now with Nepali support)
python multilingual_app.py
```

---

## Available Model Checkpoints

All weights are on [Hugging Face](https://huggingface.co/Firoj112/chatterbox-nepali-runs):

| File | Recommendation | Description |
| :--- | :--- | :--- |
| `t3_mtl_nepali_final.safetensors` | **Default** | Pure 50-epoch fine-tune (best Nepali quality) |
| `t3_mtl_nepali_merged.safetensors` | Balanced | 70% fine-tuned + 30% base (better English) |
| `t3_nepali_epoch_45.pt` | Research | Epoch 45 checkpoint |
| `t3_nepali_epoch_40.pt` | Research | Epoch 40 checkpoint |
| `t3_nepali_epoch_30.pt` | Research | Epoch 30 checkpoint |

### Why Two Final Models?

Fine-tuning for 50 epochs on Nepali data caused **Catastrophic Forgetting** — the model's English prosody degraded significantly. To fix this without retraining, we applied linear weight interpolation:

```
W_merged = 0.7 × W_fine_tuned + 0.3 × W_base
```

This restores English clarity while preserving 50 epochs of Nepali learning. Full details and math in the [Hugging Face model card](https://huggingface.co/Firoj112/chatterbox-nepali-runs).

---

## Training Details

- **Dataset**: [Firoj112/voxcpm-nepali-data](https://huggingface.co/datasets/Firoj112/voxcpm-nepali-data) (~8,000 clips)
- **Hardware**: 2x NVIDIA T4 (Kaggle, 30GB VRAM total)
- **Training strategy**: DDP + FP16 + Gradient Checkpointing
- **Epochs**: 50 total (loss: 6.7 → 0.11)
- **Language bootstrap**: `[ne]` tag initialized from `[hi]` weights for phonetic head-start

See the [Hugging Face model card](https://huggingface.co/Firoj112/chatterbox-nepali-runs) for full training details, architecture breakdown, and the merging mathematics.

---

## Nepali-Specific Features

### Text Sanitizer (`src/chatterbox/utils/sanitizer.py`)

Normalizes Nepali text before synthesis:
- Converts Devanagari numerals (१, २, ३...) to spoken form
- Expands abbreviations and units (km → किलोमिटर)
- Removes symbols that cause hallucinations

```python
from chatterbox.utils.sanitizer import sanitize_text
text = sanitize_text("नेपालको क्षेत्रफल १,४७,१८१ km² छ।", lang="ne")
```

### AlignmentStreamAnalyzer

Monitors cross-attention maps during inference to detect when the model has finished speaking all input text, then forcefully terminates generation. This prevents the long-tail gibberish that occurs with Devanagari inputs (where conjunct characters cause alignment drift).

### Smart Chunking

Long Nepali texts are split at sentence boundaries (`।`, `.`, `?`, `!`) before generation, then concatenated. This significantly improves quality for paragraphs.

---

## Original Chatterbox Tips

> These tips from the upstream repo apply to Nepali synthesis as well.

- **Reference audio**: Use a native Nepali speaker clip for best results. A Hindi reference will produce a Hindi-accented Nepali voice.
- **CFG weight**: Set `cfg_weight=0` for neutral prosody when using a cross-language reference.
- **Default settings**: `exaggeration=0.5, cfg_weight=0.5` work well for most inputs.
- **Fast reference speakers**: Lower `cfg_weight` to ~0.3 to prevent rushed speech.

---

## Supported Languages

This fork supports all 23 upstream languages plus Nepali:

`ar` Arabic • `da` Danish • `de` German • `el` Greek • `en` English • `es` Spanish • `fi` Finnish • `fr` French • `he` Hebrew • `hi` Hindi • `it` Italian • `ja` Japanese • `ko` Korean • `ms` Malay • `ne` **Nepali** • `nl` Dutch • `no` Norwegian • `pl` Polish • `pt` Portuguese • `ru` Russian • `sv` Swedish • `sw` Swahili • `tr` Turkish • `zh` Chinese

---

## Acknowledgements

- **[Resemble AI](https://resemble.ai)** — for the open-source Chatterbox model and inference engine
- **[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)**, **[S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)** — upstream dependencies
- **Wortsman et al., 2022** — Model Soups (linear weight interpolation for merging)

---

## Citation

If you use this work, please cite both the base model and this fine-tune:

```bibtex
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}

@misc{chatterboxnepali2025,
  author       = {Firoj Paudel},
  title        = {{Chatterbox Nepali TTS Fine-tune}},
  year         = {2025},
  howpublished = {\url{https://github.com/Firojpaudel/chatterbox-nepali}},
  note         = {Community fine-tune for Nepali language support}
}
```

---

## Disclaimer

This is a community fine-tune and is not officially affiliated with or endorsed by Resemble AI. Use responsibly. Do not use this model to generate misleading or harmful audio content. All generated audio includes [PerTh watermarking](https://github.com/resemble-ai/perth) from the upstream model.
