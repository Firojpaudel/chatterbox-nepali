import os
import sys
import re
import time
import random
import librosa
import torch
import numpy as np
import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

# =========================================================================
# FORCE LOCAL CODE USAGE (DO NOT REMOVE)
# =========================================================================
CURRENT_DIR = os.getcwd()
SRC_PATH = os.path.join(CURRENT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import chatterbox.utils.sanitizer as sanitizer
from chatterbox.utils.sanitizer import sanitize_text
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals

print(f"--- STARTUP CONFIG ---")
print(f"DEBUG: Using sanitizer from: {sanitizer.__file__}")
# =========================================================================

# Model path config
# Clean UI: Only keep Base and dynamically found LoRA checkpoints
CHECKPOINTS = {
    "base": ("local", "base"),
}

# Dynamically find LoRA checkpoints
lora_checkpoints = sorted([d for d in os.listdir(".") if d.startswith("lora_nepali_epoch_") and os.path.isdir(d)])
for cp in lora_checkpoints:
    epoch_num = cp.split("_")[-1]
    CHECKPOINTS[f"LoRA-Epoch-{epoch_num}"] = ("local", cp)

EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]

CUSTOM_CSS = """
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}
.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    color: #334155 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    cursor: pointer !important;
}
.tag-btn:hover {
    background: #e2e8f0 !important;
}
"""

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#main_textbox textarea');
    if (!textarea) return current_text + " " + tag_val; 
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = (start === 0 || current_text[start - 1] === ' ') ? "" : " ";
    let suffix = (end < current_text.length && current_text[end] === ' ') ? "" : " ";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = None
BASE_T3_STATE = None
CURRENT_MODEL_TYPE = "v4-epoch-10"

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# Supported Languages mapping
SUPPORTED_LANGUAGES = {
    "ne": "Nepali", "mai": "Maithili", "en": "English"
}

import gc
import copy

# We only store the filenames, not the actual weights, to save RAM
FINETUNE_FILES = {}

def get_or_load_model():
    global model, BASE_T3_STATE, CURRENT_MODEL_TYPE, FINETUNE_FILES
    if model is not None: return model

    print("Loading Chatterbox Multilingual v2 base model...")
    model = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
    model.t3.eval()
    
    # Identify available checkpoints locally
    for name, (repo_id, filename) in CHECKPOINTS.items():
        if os.path.exists(filename):
            FINETUNE_FILES[name] = filename
            print(f"Registered checkpoint: {name} ({filename})")
        else:
            print(f"⚠️ Checkpoint not found locally: {filename}")

    # Load the default model on demand (without caching yet)
    default_model = "base" if "base" in CHECKPOINTS else list(CHECKPOINTS.keys())[0]
    print(f"Loading default model ({default_model})...")
    switch_model(default_model)
    
    return model

def switch_model(model_type):
    global CURRENT_MODEL_TYPE, model, BASE_T3_STATE, FINETUNE_FILES
    from peft import PeftModel
    
    if model is None: 
        print("Error: Model not loaded yet!")
        return "Model not loaded"
    
    # Clear cache before starting
    torch.cuda.empty_cache()
    gc.collect()

    try:
        if model_type in FINETUNE_FILES:
            # LAZY CACHE: Save base state only when switching away from base for the first time
            if BASE_T3_STATE is None and CURRENT_MODEL_TYPE == "base":
                print("Caching base T3 state for future switching...")
                BASE_T3_STATE = {k: v.cpu().clone() for k, v in model.t3.state_dict().items()}

            filename = FINETUNE_FILES[model_type]
            print(f"--- Loading weights for: {model_type} ({filename}) ---")
            
            # If current model is a PeftModel, unload it first to restore base weights
            if isinstance(model.t3.tfmr, PeftModel):
                print("Unloading previous LoRA adapter...")
                model.t3.tfmr = model.t3.tfmr.unload()

            if os.path.isdir(filename):
                # Handle LoRA Directory
                print(f"Applying LoRA adapter from directory: {filename}")
                
                # Check and resize if needed before loading embeddings
                emb_path = os.path.join(filename, "text_emb.pt")
                if os.path.exists(emb_path):
                    state = torch.load(emb_path, map_location="cpu", weights_only=True)
                    state_vocab_size = state["weight"].shape[0] if "weight" in state else state.shape[0]
                    model_vocab_size = model.t3.hp.text_tokens_dict_size
                    if state_vocab_size != model_vocab_size:
                        print(f"Resizing T3 vocabulary for LoRA from {model_vocab_size} to {state_vocab_size}")
                        model.t3.resize_text_embeddings(state_vocab_size)
                    
                    print(f"Loading text embeddings from {emb_path}")
                    model.t3.text_emb.load_state_dict(torch.load(emb_path, map_location=DEVICE, weights_only=True))
                
                # Apply LoRA
                model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, filename)
                model.t3.tfmr.to(DEVICE)
                
                # Force activate the adapter
                model.t3.tfmr.set_adapter("default")
                model.t3.tfmr.eval()
                
                trainable = sum(p.numel() for p in model.t3.tfmr.parameters() if p.requires_grad)
                print(f"✅ LoRA Active! Trainable params: {trainable:,}")
                
                model.t3.to(DEVICE).eval()
                
                # Merge LoRA for maximum performance (RTX 4090 win)
                try:
                    model.t3.tfmr = model.t3.tfmr.merge_and_unload()
                    print("🚀 LoRA Merged into base weights for maximum speed.")
                except Exception as e:
                    print(f"⚠️ Could not merge LoRA (performance may be lower): {e}")
            else:
                # Handle standard .pt / .safetensors files
                if filename.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state = load_file(filename, device="cpu")
                else:
                    state = torch.load(filename, map_location="cpu", weights_only=True)
                
                clean_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in state.items()}
                
                if "text_emb.weight" in clean_state:
                    state_vocab_size = clean_state["text_emb.weight"].shape[0]
                    model_vocab_size = model.t3.hp.text_tokens_dict_size
                    if state_vocab_size != model_vocab_size:
                        print(f"Resizing T3 vocabulary from {model_vocab_size} to {state_vocab_size}")
                        model.t3.resize_text_embeddings(state_vocab_size)
                
                model.t3.load_state_dict(clean_state, strict=False)
                model.t3.to(DEVICE).eval()
            
            # Enable TensorFloat32 for better 4090 performance
            torch.set_float32_matmul_precision('high')
            
            gc.collect()
            CURRENT_MODEL_TYPE = model_type
            print(f"Successfully loaded {model_type}")
            return f"Active: {model_type}"
            
        elif model_type == "base":
            print("--- Switching to Base Multilingual model ---")
            
            # If current model is a PeftModel, unload it
            if isinstance(model.t3.tfmr, PeftModel):
                print("Unloading LoRA adapter...")
                model.t3.tfmr = model.t3.tfmr.unload()

            if BASE_T3_STATE is None:
                return "Base state missing"
            
            if "text_emb.weight" in BASE_T3_STATE:
                state_vocab_size = BASE_T3_STATE["text_emb.weight"].shape[0]
                model_vocab_size = model.t3.hp.text_tokens_dict_size
                if state_vocab_size != model_vocab_size:
                    print(f"Resizing T3 vocabulary back from {model_vocab_size} to {state_vocab_size} for base model")
                    model.t3.resize_text_embeddings(state_vocab_size)
            
            model.t3.load_state_dict(BASE_T3_STATE, strict=True)
            model.t3.to(DEVICE).eval()
            model.t3.compiled = False
            
            gc.collect()
            CURRENT_MODEL_TYPE = "base"
            return "Active: Base"
        else:
            print(f"Error: Unknown model type {model_type}")
            return f"Unknown: {model_type}"
    except Exception as e:
        print(f"CRITICAL ERROR during model switch: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

def smart_chunk(text, max_chars=300):
    # 1. First split by sentence markers
    sentences = re.split(r'([।\.?!\n])', text)
    
    parts = []
    for i in range(0, len(sentences)-1, 2):
        s = (sentences[i] + sentences[i+1]).strip()
        if not s: continue
        
        if len(s) > 150 and "," in s:
            # 2. If a sentence is long, split by commas as well
            subparts = re.split(r'(,)', s)
            for j in range(0, len(subparts)-1, 2):
                parts.append(subparts[j] + subparts[j+1])
            if len(subparts) % 2 != 0:
                parts.append(subparts[-1])
        else:
            parts.append(s)
            
    # Handle last part if any
    if len(sentences) % 2 != 0 and sentences[-1].strip():
        parts.append(sentences[-1].strip())

    final_chunks = []
    current_chunk = ""
    for part in parts:
        if not part.strip(): continue
        if len(current_chunk) + len(part) < max_chars:
            current_chunk += " " + part
        else:
            if current_chunk.strip():
                final_chunks.append(current_chunk.strip())
            current_chunk = part
            
    if current_chunk.strip():
        final_chunks.append(current_chunk.strip())
        
    return [c for c in final_chunks if c.strip()]

def generate_tts_audio(
    text_input, language_id, ref_dropdown_input, audio_prompt_input,
    exaggeration_input, cfg_weight_input, temperature_input,
    seed_num_input, enable_sanitizer, enable_chunking, enable_protection, speed_input,
    repetition_penalty_input, top_p_input, min_p_input
):
    global model
    model = get_or_load_model()

    # --- FULL SANITIZATION ---
    if enable_sanitizer:
        text_input = sanitize_text(text_input, lang=language_id)
        print(f"🧹 Sanitized Text: {text_input[:200]}...")

    # --- TURBO ADAPTER ROUTING ---
    from peft import PeftModel
    if isinstance(model.t3.tfmr, PeftModel):
        if language_id == "en":
            # Disable adapters for English to use base multilingual weights
            if not getattr(model.t3.tfmr.base_model, "disable_adapters", False):
                print("⚡ Speed-Mode: Bypassing LoRA for English")
                model.t3.tfmr.base_model.disable_adapters = True
        else:
            # Re-enable adapters for Nepali/Maithili
            if getattr(model.t3.tfmr.base_model, "disable_adapters", False):
                print(f"⚡ Speed-Mode: Activating LoRA for {language_id}")
                model.t3.tfmr.base_model.disable_adapters = False
            model.t3.tfmr.set_adapter("default")

    ref_path = audio_prompt_input or os.path.join("samples", ref_dropdown_input)
    if not os.path.exists(ref_path): ref_path = "samples/Prakash.mp3"
    
    chunks = smart_chunk(text_input) if enable_chunking else [text_input]
    print(f"🚀 Synthesis: {len(chunks)} chunks | Lang: {language_id} | Protection: {enable_protection}")

    model.prepare_conditionals(wav_fpath=ref_path, exaggeration=exaggeration_input)
    
    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    all_wavs = []
    total_start = time.time()
    
    for i, chunk in enumerate(chunks):
        if not chunk.strip(): continue
        
        chunk_start = time.time()
        print(f"--- Processing Chunk {i+1}/{len(chunks)} ---")
        
        with torch.inference_mode():
            wav = model.generate(
                chunk, language_id=language_id, exaggeration=exaggeration_input,
                cfg_weight=cfg_weight_input, temperature=temperature_input,
                repetition_penalty=repetition_penalty_input, top_p=top_p_input,
                min_p=min_p_input, max_new_tokens=max(128, int(len(chunk) * 4)),
                enable_protection=enable_protection
            )
            all_wavs.append(wav.squeeze(0).cpu().numpy())
        
        chunk_elapsed = time.time() - chunk_start
        audio_duration = len(wav.squeeze(0)) / model.sr
        rtf = chunk_elapsed / audio_duration if audio_duration > 0 else 0
        
        if i == 0:
            print(f"⏱️  TTFA (Time to First Audio): {chunk_elapsed:.3f}s")
        
        print(f"   Done in {chunk_elapsed:.2f}s | RTF: {rtf:.2f}")

    if not all_wavs: return None
    
    final_wav = np.concatenate(all_wavs)
    if speed_input != 1.0:
        final_wav = librosa.effects.time_stretch(final_wav, rate=speed_input)

    total_elapsed = time.time() - total_start
    total_audio_duration = len(final_wav) / model.sr
    total_rtf = total_elapsed / total_audio_duration if total_audio_duration > 0 else 0
    
    print(f"✅ SUCCESS: {total_audio_duration:.1f}s audio generated in {total_elapsed:.1f}s (Total RTF: {total_rtf:.2f})")
    
    # Convert to 16-bit PCM to prevent Gradio warnings and ensure compatibility
    final_wav = np.clip(final_wav, -1, 1)
    final_wav = (final_wav * 32767).astype(np.int16)
    
    return (model.sr, final_wav)

# ===================== GRADIO UI =====================

with gr.Blocks(title="Chatterbox Nepali TTS", css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🎙️ Chatterbox Nepali TTS\nState-of-the-Art Nepali Speech Synthesis")

    with gr.Row():
        with gr.Column(scale=2):
            text = gr.Textbox(
                label="Text to Synthesize", lines=8, elem_id="main_textbox",
                value="नेपाल दक्षिण एसियामा अवस्थित एक भूपरिवेष्टित देश हो। यसको जनसंख्या लगभग ३ करोड छ।"
            )
            with gr.Row(elem_classes=["tag-container"]):
                for tag in EVENT_TAGS:
                    btn = gr.Button(tag, elem_classes=["tag-btn"])
                    btn.click(fn=None, inputs=[btn, text], outputs=text, js=INSERT_TAG_JS)
            
            with gr.Row():
                enable_sanitizer = gr.Checkbox(label="Aggressive Sanitizer", value=True)
                enable_chunking = gr.Checkbox(label="Smart Chunking", value=True)
                enable_protection = gr.Checkbox(label="Hallucination Protection", value=True)
            
            with gr.Row():
                language_id = gr.Dropdown(choices=list(SUPPORTED_LANGUAGES.keys()), value="ne", label="Language")
                ref_dropdown = gr.Dropdown(choices=[f.name for f in Path("samples").glob("*") if f.suffix in [".mp3", ".wav"]], value="Prakash.mp3", label="Reference Voice")
            ref_wav = gr.Audio(label="OR Upload Custom Reference", type="filepath")

        with gr.Column(scale=1):
            default_model = "LoRA-Epoch-4" if "LoRA-Epoch-4" in CHECKPOINTS else "base"
            model_selector = gr.Radio(
                choices=list(CHECKPOINTS.keys()),
                value=default_model,
                label="Checkpoint"
            )
            model_status = gr.Markdown(f"**Status:** `Active: {default_model}`")
            
            exaggeration = gr.Slider(0.0, 3.0, value=0.0, step=0.1, label="Exaggeration")
            cfg_weight = gr.Slider(0.0, 3.0, value=0.8, step=0.1, label="CFG (Pace)")
            
            with gr.Accordion("Advanced Options", open=False):
                speed_slider = gr.Slider(0.5, 1.5, step=0.05, value=1.0, label="Speed")
                temp = gr.Slider(0.05, 2.0, step=0.05, label="Temperature", value=0.7)
                top_p = gr.Slider(0.0, 1.0, step=0.05, label="Top P", value=1.0)
                min_p = gr.Slider(0.0, 1.0, step=0.01, label="Min P", value=0.05)
                repetition_penalty = gr.Slider(1.0, 5.0, step=0.1, label="Repetition Penalty", value=2.5)
                seed_num = gr.Number(value=0, label="Seed")

            run_btn = gr.Button("Generate Speech", variant="primary")
            audio_output = gr.Audio(label="Output Audio")

    gr.Examples(
        examples=[
            ["नेपाल दक्षिण एसियामा अवस्थित एक भूपरिवेष्टित देश हो।", "ne", "Prakash.mp3"],
            ["यसको जनसंख्या लगभग ३ करोड छ। [laugh]", "ne", "Prakash.mp3"],
            ["WHO ले सन् २०२० मा COVID-19 लाई विश्वव्यापी महामारी घोषणा गर्यो।", "ne", "Prakash.mp3"],
            ["सगरमाथाको उचाइ 8848.86m छ।", "ne", "Prakash.mp3"]
        ],
        inputs=[text, language_id, ref_dropdown]
    )

    def load_wrapper(): 
        get_or_load_model()
        # If default is not base, switch to it immediately
        default_val = "LoRA-Epoch-4" if "LoRA-Epoch-4" in CHECKPOINTS else "base"
        if default_val != "base":
            print(f"🔄 Auto-loading default checkpoint: {default_val}")
            switch_model(default_val)
        return None

    demo.load(fn=load_wrapper, outputs=None)
    model_selector.change(fn=switch_model, inputs=[model_selector], outputs=[model_status])
    run_btn.click(
        fn=generate_tts_audio,
        inputs=[text, language_id, ref_dropdown, ref_wav, exaggeration, cfg_weight, temp, seed_num, enable_sanitizer, enable_chunking, enable_protection, speed_slider, repetition_penalty, top_p, min_p],
        outputs=[audio_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
