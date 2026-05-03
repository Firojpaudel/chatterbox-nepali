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
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.vllm.tts import ChatterboxTTS

print(f"--- STARTUP CONFIG ---")
print(f"DEBUG: Using sanitizer from: {sanitizer.__file__}")
# =========================================================================

# Model path config
REPO_ID = "Firoj112/chatterbox-nepali-runs"
CHECKPOINTS = {
    "nepali-epoch-30": "t3_nepali_epoch_30.pt",
    "nepali-epoch-40": "t3_nepali_epoch_40.pt",
    "nepali-epoch-45": "t3_nepali_epoch_45.pt",
    "nepali-final": "t3_mtl_nepali_final.safetensors",
    "nepali-merged": "t3_mtl_nepali_merged.safetensors",  # 0.7 FT / 0.3 Base blend
}

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
vllm_model = None
BASE_T3_STATE = None
# We no longer pre-load states to save RAM
CURRENT_MODEL_TYPE = "nepali-final"
# Set ENABLE_VLLM=1 to skip standard model and load vLLM at startup
VLLM_MODE = os.environ.get("ENABLE_VLLM", "").strip() in ("1", "true", "True", "yes")

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# Supported Languages mapping
SUPPORTED_LANGUAGES = {
    "en": "English", "ne": "Nepali", "hi": "Hindi", "bn": "Bengali",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu"
}

import gc
import copy

# We only store the filenames, not the actual weights, to save RAM
FINETUNE_FILES = {}

def get_or_load_model(use_vllm=False):
    global model, vllm_model, BASE_T3_STATE, CURRENT_MODEL_TYPE, FINETUNE_FILES
    
    if use_vllm:
        if vllm_model is not None: return vllm_model
        # Free the standard model from GPU before loading vLLM — they can't coexist on T4
        if model is not None:
            print("⚠️ Unloading standard model to free GPU memory for vLLM...")
            try:
                # Move internal sub-models to CPU to free GPU memory
                for attr_name in ['t3', 's3gen', 've']:
                    sub = getattr(model, attr_name, None)
                    if sub is not None and hasattr(sub, 'cpu'):
                        sub.cpu()
            except Exception as e:
                print(f"   Warning during model offload: {e}")
            del model
            model = None
            BASE_T3_STATE = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"   GPU memory freed: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
        print("Loading Chatterbox vLLM model...")
        model_name = CHECKPOINTS.get(CURRENT_MODEL_TYPE, "t3_mtl_nepali_merged.safetensors")
        vllm_model = ChatterboxTTS.from_nepali(model_filename=model_name)
        return vllm_model
    else:
        if model is not None: return model
        print("Loading Chatterbox Multilingual v2 base model...")
        model = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
        model.t3.eval()
        
        # Save a copy of the base T3 state for switching back
        print("Caching base T3 state...")
        BASE_T3_STATE = {k: v.cpu().clone() for k, v in model.t3.state_dict().items()}
        
        # Identify available checkpoints but DON'T load them yet
        for name, filename in CHECKPOINTS.items():
            try:
                # Use standard HF cache to avoid double downloads
                path = hf_hub_download(repo_id=REPO_ID, filename=filename)
                FINETUNE_FILES[name] = path
                print(f"Registered checkpoint: {name}")
            except Exception as e:
                print(f"Could not download {filename}: {e}")
                continue

        # Load the default model (nepali-final) on demand
        if "nepali-final" in FINETUNE_FILES:
            print("Loading default Nepali model (nepali-final)...")
            switch_model("nepali-final")
        
        return model

def switch_model(model_type):
    global CURRENT_MODEL_TYPE, model, BASE_T3_STATE, FINETUNE_FILES
    if model is None: 
        print("Error: Model not loaded yet!")
        return "Model not loaded"
    
    # Clear cache before starting
    torch.cuda.empty_cache()
    gc.collect()

    try:
        if model_type in FINETUNE_FILES:
            filename = FINETUNE_FILES[model_type]
            print(f"--- Loading weights for: {model_type} ({filename}) ---")
            
            # Reset vLLM model if it exists, so it reloads with new weights next time
            global vllm_model
            if vllm_model is not None:
                vllm_model.shutdown()
                vllm_model = None

            if filename.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(filename, device="cpu")
            else:
                state = torch.load(filename, map_location="cpu", weights_only=True)
            
            clean_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in state.items()}
            
            if model is not None:
                model.t3.load_state_dict(clean_state, strict=False)
                model.t3.to(DEVICE).eval()
                model.t3.compiled = False
            
            # Immediately delete state to free RAM
            del state
            del clean_state
            gc.collect()
            
            CURRENT_MODEL_TYPE = model_type
            print(f"Successfully loaded {model_type}")
            return f"Active: {model_type}"
            
        elif model_type == "base":
            print("--- Switching to Base Multilingual model ---")
            if BASE_T3_STATE is None:
                return "Base state missing"
            
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

def smart_chunk(text, max_chars=120):
    raw_chunks = re.split(r'([।\.?!\n])', text)
    final_chunks = []
    for i in range(0, len(raw_chunks)-1, 2):
        sentence = (raw_chunks[i] + raw_chunks[i+1]).strip()
        if not sentence: continue
        if len(sentence) > max_chars:
            sub_chunks = re.split(r'([,])', sentence)
            if len(sub_chunks) > 1:
                for j in range(0, len(sub_chunks)-1, 2):
                    sub = (sub_chunks[j] + sub_chunks[j+1]).strip()
                    if sub: final_chunks.append(sub)
                if len(sub_chunks) % 2 != 0 and sub_chunks[-1].strip():
                    final_chunks.append(sub_chunks[-1].strip())
            else:
                words = sentence.split(' '); mid = len(words) // 2
                if mid > 0:
                    final_chunks.append(" ".join(words[:mid]))
                    final_chunks.append(" ".join(words[mid:]))
                else: final_chunks.append(sentence)
        else: final_chunks.append(sentence)
    if len(raw_chunks) % 2 != 0 and raw_chunks[-1].strip():
        final_chunks.append(raw_chunks[-1].strip())
    return final_chunks

def generate_tts_audio(
    text_input, language_id, ref_dropdown_input, audio_prompt_input,
    exaggeration_input, cfg_weight_input, temperature_input,
    seed_num_input, enable_sanitizer, enable_chunking, enable_protection, speed_input,
    repetition_penalty_input, top_p_input, min_p_input, use_vllm
):
    total_start = time.time()
    model = get_or_load_model(use_vllm=use_vllm)
    ref_path = audio_prompt_input or os.path.join("samples", ref_dropdown_input)
    if not os.path.exists(ref_path): ref_path = "samples/Prakash.mp3"
    
    if enable_sanitizer:
        text_input = sanitize_text(text_input, lang=language_id)
        print(f"Sanitized Text: {text_input}")
    
    chunks = smart_chunk(text_input) if enable_chunking else [text_input]
    print(f"Split into {len(chunks)} chunk(s). Starting synthesis...")

    if use_vllm:
        # vLLM handles everything in one batch
        print("Using vLLM for inference...")
        if seed_num_input != 0:
            set_seed(int(seed_num_input))
            
        audios = model.generate(
            chunks,
            audio_prompt_path=ref_path,
            language_id=language_id,
            exaggeration=exaggeration_input,
            temperature=temperature_input,
            top_p=top_p_input,
            repetition_penalty=repetition_penalty_input,
            max_tokens=max(512, int(max(len(c) for c in chunks) * 4))
        )
        all_wavs = [a.squeeze(0).cpu().numpy() for a in audios]
    else:
        model.prepare_conditionals(wav_fpath=ref_path, exaggeration=exaggeration_input)
        
        if seed_num_input != 0:
            set_seed(int(seed_num_input))

        all_wavs = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip(): continue
            
            chunk_start = time.time()
            print(f"[Processing Chunk {i+1}/{len(chunks)}]")
            
            # Add tiny trailing space
            if not chunk.endswith(" "): chunk += " "
            
            with torch.no_grad():
                wav = model.generate(
                    chunk, language_id=language_id, exaggeration=exaggeration_input,
                    cfg_weight=cfg_weight_input, temperature=temperature_input,
                    repetition_penalty=repetition_penalty_input, top_p=top_p_input,
                    min_p=min_p_input, max_new_tokens=max(128, int(len(chunk) * 4)),
                    enable_protection=enable_protection
                )
                all_wavs.append(wav.squeeze(0).cpu().numpy())
            
            chunk_elapsed = time.time() - chunk_start
            print(f"   Chunk {i+1} done in {chunk_elapsed:.1f}s")

    if not all_wavs: return None
    
    final_wav = np.concatenate(all_wavs)
    if speed_input != 1.0:
        final_wav = librosa.effects.time_stretch(final_wav, rate=speed_input)

    total_elapsed = time.time() - total_start
    print(f"SUCCESS: All {len(chunks)} chunks synthesized in {total_elapsed:.1f}s total!")
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
                use_vllm = gr.Checkbox(label="Enable vLLM (Fast Inference)", value=VLLM_MODE)
            
            with gr.Row():
                language_id = gr.Dropdown(choices=list(SUPPORTED_LANGUAGES.keys()), value="ne", label="Language")
                ref_dropdown = gr.Dropdown(choices=[f.name for f in Path("samples").glob("*") if f.suffix in [".mp3", ".wav"]], value="Prakash.mp3", label="Reference Voice")
            ref_wav = gr.Audio(label="OR Upload Custom Reference", type="filepath")

        with gr.Column(scale=1):
            model_selector = gr.Radio(
                choices=["base", "nepali-epoch-30", "nepali-epoch-40", "nepali-epoch-45", "nepali-final", "nepali-merged"],
                value="nepali-final",
                label="Checkpoint"
            )
            model_status = gr.Markdown(f"**Status:** `Active: nepali-final`")
            
            exaggeration = gr.Slider(0.0, 3.0, value=0.0, step=0.1, label="Exaggeration")
            cfg_weight = gr.Slider(0.0, 3.0, value=0.8, step=0.1, label="CFG (Pace)")
            
            with gr.Accordion("Advanced Options", open=False):
                speed_slider = gr.Slider(0.5, 1.5, step=0.05, value=1.0, label="Speed")
                temp = gr.Slider(0.05, 2.0, step=0.05, label="Temperature", value=0.8)
                top_p = gr.Slider(0.0, 1.0, step=0.05, label="Top P", value=1.0)
                min_p = gr.Slider(0.0, 1.0, step=0.01, label="Min P", value=0.05)
                repetition_penalty = gr.Slider(1.0, 5.0, step=0.1, label="Repetition Penalty", value=2.0)
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

    def load_wrapper(use_vllm): 
        if not VLLM_MODE:  # Only auto-load if not pre-loaded at startup
            get_or_load_model(use_vllm=use_vllm)
        return None

    demo.load(fn=load_wrapper, inputs=[use_vllm], outputs=None)
    model_selector.change(fn=switch_model, inputs=[model_selector], outputs=[model_status])
    run_btn.click(
        fn=generate_tts_audio,
        inputs=[text, language_id, ref_dropdown, ref_wav, exaggeration, cfg_weight, temp, seed_num, enable_sanitizer, enable_chunking, enable_protection, speed_slider, repetition_penalty, top_p, min_p, use_vllm],
        outputs=[audio_output]
    )

if __name__ == "__main__":
    if VLLM_MODE:
        print("🚀 ENABLE_VLLM is set — pre-loading vLLM at startup (outside Gradio request handler)...")
        get_or_load_model(use_vllm=True)
        print("✅ vLLM model ready!")
    demo.launch(server_name="0.0.0.0", share=True)
