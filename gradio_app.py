"""
Chatterbox Nepali TTS - Gradio App
Two modes:
  - Base (multilingual v2) — original 23-language model
  - Nepali Fine-tune — same model + epoch-30 Nepali weights
"""

import os
import sys
import re
import random
import glob
import time
import numpy as np
import torch
import librosa
import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.append(str(Path("src").absolute()))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from chatterbox.utils.sanitizer import sanitize_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

# --- Global State ---
MODEL = None
BASE_T3_STATE = None
FINETUNE_STATES = {}  # { "nepali-epoch-30": state, "nepali-epoch-40": state }
CURRENT_MODEL_TYPE = "base"

LANGUAGE_CONFIG = {
    "ne": {"text": "नमस्ते, मेरो नाम चैटरबक्स हो। म नेपालीमा बोल्न सक्छु।"},
    "en": {"text": "Hello, this is Chatterbox TTS fine-tuned for Nepali with thirty epochs of training."},
    "hi": {"text": "नमस्ते, यह चैटरबॉक्स टीटीएस है।"},
}


def get_or_load_model():
    global MODEL, BASE_T3_STATE, FINETUNE_T3_STATE, CURRENT_MODEL_TYPE

    if MODEL is not None:
        return MODEL

    print("Loading Chatterbox Multilingual v2 base model...")
    MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
    # Speed optimization: float16 for T3 only (the LLM backbone where 99% of compute is)
    # s3gen and ve must stay float32 — they have internal precision-sensitive ops
    if DEVICE == "cuda":
        MODEL.t3.half()
        print("⚡ Float16 inference enabled (T3 backbone)")

    # Cache base T3 weights
    BASE_T3_STATE = {k: v.clone().cpu() for k, v in MODEL.t3.state_dict().items()}
    print("✅ Base weights cached.")

    # Cache Nepali fine-tuned weights
    REPO_ID = "Firoj112/chatterbox-nepali-runs"
    CHECKPOINTS = {
        "nepali-final": "t3_mtl_nepali_final.safetensors",
        "nepali-epoch-45": "t3_nepali_epoch_45.pt",
        "nepali-epoch-40": "t3_nepali_epoch_40.pt",
        "nepali-epoch-30": "t3_nepali_epoch_30.pt",
    }

    for name, filename in CHECKPOINTS.items():
        try:
            # This will use cached file if available, or download if missing
            print(f"Loading weights for {name} ({filename})...")
            path = hf_hub_download(repo_id=REPO_ID, filename=filename)
            if filename.endswith(".safetensors"):
                from safetensors.torch import load_file as load_safetensors
                raw = load_safetensors(path)
            else:
                raw = torch.load(path, map_location='cpu', weights_only=True)
            
            # Remove "patched_model." or "model." prefix if present
            state_dict = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in raw.items()}
            FINETUNE_STATES[name] = state_dict
            print(f"DONE: {name} weights cached.")
        except Exception as e:
            print(f"ERROR: Could not load {name}: {e}")

    # Set default model: prefer nepali-final, then epoch-45, then base
    if "nepali-final" in FINETUNE_STATES:
        CURRENT_MODEL_TYPE = "nepali-final"
        MODEL.t3.load_state_dict(FINETUNE_STATES["nepali-final"], strict=False)
        print("Active: nepali-final")
    elif "nepali-epoch-45" in FINETUNE_STATES:
        CURRENT_MODEL_TYPE = "nepali-epoch-45"
        MODEL.t3.load_state_dict(FINETUNE_STATES["nepali-epoch-45"], strict=False)
        print("Active: nepali-epoch-45")
    else:
        CURRENT_MODEL_TYPE = "base"
        print("Active: base")

    print(f"Model ready on {MODEL.device}")
    return MODEL


def switch_model(model_type):
    global CURRENT_MODEL_TYPE

    model = get_or_load_model()

    if model_type == CURRENT_MODEL_TYPE:
        return f"Already active: {model_type}"

    if model_type in FINETUNE_STATES:
        state = FINETUNE_STATES[model_type]
        model.t3.load_state_dict(state, strict=False)
        model.t3.to(DEVICE).eval()
        model.t3.compiled = False
        CURRENT_MODEL_TYPE = model_type
        print(f"🔄 Switched to: {model_type}")
        return f"✅ Active: {model_type}"

    elif model_type == "base":
        model.t3.load_state_dict(BASE_T3_STATE, strict=True)
        model.t3.to(DEVICE).eval()
        model.t3.compiled = False
        CURRENT_MODEL_TYPE = "base"
        print("🔄 Switched to: Base")
        return "✅ Active: Base (Multilingual v2)"

    return f"Unknown: {model_type}"


# Pre-load
try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: {e}")


def set_seed(seed):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def smart_chunk(text):
    """Split text strictly on sentence boundaries to preserve natural prosody."""
    # Split on sentence boundaries
    chunks = re.split(r'([।\.?!\n])', text)
    raw = []
    for i in range(0, len(chunks) - 1, 2):
        raw.append((chunks[i] + chunks[i + 1]).strip())
    if len(chunks) % 2 == 1 and chunks[-1].strip():
        raw.append(chunks[-1].strip())

    # Merge extremely short chunks (like "हो।") to the previous sentence if possible
    processed = []
    buf = ""
    for s in raw:
        if len(buf + s) < 20:
            buf += s + " "
        else:
            processed.append((buf + s).strip())
            buf = ""
    if buf:
        if processed:
            processed[-1] += " " + buf.strip()
        else:
            processed.append(buf.strip())
            
    return processed if processed else [text]


def generate_tts_audio(
    text_input, language_id, ref_dropdown_input, audio_prompt_input,
    exaggeration_input, cfg_weight_input, temperature_input,
    seed_num_input, enable_sanitizer, enable_chunking, speed_input
):
    model = get_or_load_model()
    if model is None:
        raise RuntimeError("No model loaded.")

    if seed_num_input and int(seed_num_input) != 0:
        set_seed(int(seed_num_input))

    # Resolve audio prompt: uploaded takes priority
    audio_prompt = audio_prompt_input or ref_dropdown_input
    if not audio_prompt:
        raise gr.Error("❌ Please select or upload a reference audio.")
    print(f"Using reference: {audio_prompt}")

    # Sanitize
    if enable_sanitizer:
        text_input = sanitize_text(text_input, lang=language_id)
        print(f"Sanitized ({language_id}): {text_input}")

    # Chunk text to prevent OOM/slowdown on massive paragraphs,
    # but ONLY chunk on natural sentence boundaries (as defined in smart_chunk)
    if enable_chunking and len(text_input) > 80:
        chunks = smart_chunk(text_input)
    else:
        chunks = [text_input]

    print(f"[{CURRENT_MODEL_TYPE}] Processing {len(chunks)} chunk(s)...")
    all_wavs = []
    start = time.time()

    # Pre-compute reference audio conditionals ONCE (instead of per-chunk)
    model.prepare_conditionals(audio_prompt, exaggeration=exaggeration_input)
    print(f"  Reference audio processed.")

    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] {chunk}")
        # Give the model plenty of tokens to finish the sentence safely
        estimated_tokens = min(len(chunk) * 6 + 150, 1000)
        estimated_tokens = max(estimated_tokens, 200)
        
        with torch.inference_mode():
            wav = model.generate(
                chunk,
                language_id=language_id,
                # Don't pass audio_prompt_path — conditionals already cached
                exaggeration=exaggeration_input,
                cfg_weight=cfg_weight_input,
                temperature=temperature_input,
                max_new_tokens=estimated_tokens,
            )
            all_wavs.append(wav.squeeze(0).cpu().numpy())

    if not all_wavs:
        raise gr.Error("No audio generated.")

    final_wav = np.concatenate(all_wavs, axis=-1)
    final_wav, _ = librosa.effects.trim(final_wav, top_db=50)

    # Adjust final audio speed if requested
    if speed_input != 1.0:
        # rate > 1.0 speeds up, rate < 1.0 slows down
        final_wav = librosa.effects.time_stretch(final_wav, rate=speed_input)

    elapsed = time.time() - start
    print(f"DONE in {elapsed:.1f}s | {len(chunks)} chunk(s)")
    return (model.sr, final_wav)


# ===================== GRADIO UI =====================

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Nepali Chatterbox TTS
        Fine-tuned Chatterbox Multilingual v2 for Nepali — with sanitization & smart chunking.

        Based on [ResembleAI/Chatterbox](https://github.com/resemble-ai/chatterbox) | MIT License
        """
    )

    lang_display = " • ".join([f"**{name}** (`{code}`)" for code, name in sorted(SUPPORTED_LANGUAGES.items())])
    gr.Markdown(lang_display)

    with gr.Row():
        with gr.Column():
            # Model Selector
            choices = ["base"] + list(FINETUNE_STATES.keys())
            model_selector = gr.Radio(
                choices=choices,
                value=CURRENT_MODEL_TYPE,
                label="Model",
                info="Select model checkpoint: base (multilingual) or fine-tuned Nepali epochs"
            )
            model_status = gr.Textbox(
                value=f"Active: {CURRENT_MODEL_TYPE}",
                label="Model Status",
                interactive=False
            )

            # Language
            language_id = gr.Dropdown(
                choices=list(SUPPORTED_LANGUAGES.keys()),
                value="en",
                label="Language"
            )

            # Text
            text = gr.Textbox(
                value=LANGUAGE_CONFIG["en"]["text"],
                label="Text to synthesize",
                max_lines=5,
                placeholder="Type your text here..."
            )

            # Reference audio dropdown
            available_refs = sorted(
                glob.glob("samples/*.wav") + glob.glob("samples/*.mp3") +
                glob.glob("*.wav") + glob.glob("*.mp3")
            )
            # Prioritize Prakash.mp3 if it exists
            prakash_ref = next((r for r in available_refs if "Prakash" in r), None)
            default_ref = prakash_ref if prakash_ref else (available_refs[0] if available_refs else None)

            ref_dropdown = gr.Dropdown(
                choices=available_refs,
                value=default_ref,
                label="Reference Voice",
                info="Select from available audio files"
            )

            # Custom upload
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="OR Upload / Record Custom Reference (5-10s)",
            )

            gr.Markdown(
                "💡 **Tip**: Lower CFG = slower & more expressive. "
                "Set CFG to 0 for cross-language transfer."
            )

            exaggeration = gr.Slider(
                0.25, 0.8, step=0.05,
                label="Exaggeration (Emotion, 0.5 = neutral)",
                value=0.25
            )
            cfg_weight = gr.Slider(
                minimum=0.0, maximum=3.0, value=0.8, step=0.1,
                label="CFG / Pace (Higher = stricter accent, but speaks faster)",
            )

            with gr.Row():
                enable_sanitizer = gr.Checkbox(label="Sanitize Numbers & Acronyms", value=True)
                enable_chunking = gr.Checkbox(label="Smart Chunking (long text)", value=True)

            with gr.Accordion("Advanced Options", open=False):
                speed_slider = gr.Slider(0.5, 1.5, step=0.05, value=1.0, label="Audio Speed (Pace control)")
                temp = gr.Slider(0.05, 5.0, step=0.05, label="Temperature", value=0.8)
                seed_num = gr.Number(value=0, label="Random Seed (0 = random)")

            run_btn = gr.Button("🎙️ Generate Speech", variant="primary", size="lg")

        with gr.Column():
            audio_output = gr.Audio(label="Generated Audio")

    # --- Events ---
    model_selector.change(
        fn=switch_model,
        inputs=[model_selector],
        outputs=[model_status],
    )

    def on_language_change(lang):
        return LANGUAGE_CONFIG.get(lang, {}).get("text", "")

    language_id.change(
        fn=on_language_change,
        inputs=[language_id],
        outputs=[text],
        show_progress=False
    )

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text, language_id, ref_dropdown, ref_wav,
            exaggeration, cfg_weight, temp,
            seed_num, enable_sanitizer, enable_chunking, speed_slider
        ],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
