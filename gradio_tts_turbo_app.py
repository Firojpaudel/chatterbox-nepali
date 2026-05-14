import os
import random
import numpy as np
import torch
import gradio as gr
from pathlib import Path
from peft import PeftModel
from chatterbox.tts_turbo import ChatterboxTurboTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]

SUPPORTED_LANGUAGES = {
    "ne": "Nepali",
    "mai": "Maithili",
    "en": "English (Base)"
}

# --- Checkpoint Detection ---
TURBO_CHECKPOINTS = {"Base": "base"}
turbo_dirs = sorted([d for d in os.listdir(".") if d.startswith("turbo_lora_epoch_") and os.path.isdir(d)])
for d in turbo_dirs:
    epoch = d.split("_")[-1]
    TURBO_CHECKPOINTS[f"Epoch {epoch}"] = d

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
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}

.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
"""

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#main_textbox textarea');
    if (!textarea) return current_text + " " + tag_val; 

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;

    let prefix = " ";
    let suffix = " ";

    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";

    if (end < current_text.length && current_text[end] === ' ') suffix = "";

    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_base_model():
    print(f"🚀 Loading Chatterbox-Turbo Base on {DEVICE}...")
    model = ChatterboxTurboTTS.from_pretrained(DEVICE)
    model.is_lora_active = False
    return model

def switch_checkpoint(model, ckpt_name):
    print(f"🔄 Preparing clean state for {ckpt_name}...")
    
    # 1. ALWAYS reload the base model to guarantee clean weights and embeddings!
    # Otherwise, resized embeddings and merged LoRAs permanently corrupt the base model.
    model = load_base_model()
    
    ckpt_path = TURBO_CHECKPOINTS[ckpt_name]
    
    if ckpt_path == "base":
        print("   ✅ Base model restored cleanly.")
        return model, f"Active: Base"

    print(f"🔄 Applying checkpoint: {ckpt_name} ({ckpt_path})")

    # 2. Update Tokenizer with new tags (Crucial for [ne], [mai])
    try:
        from transformers import AutoTokenizer
        from chatterbox.tts_turbo import REPO_ID
        print(f"   📖 Initializing tokenizer with custom tags...")
        model.tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        
        # Add the language tags so they aren't split during inference
        num_added = model.tokenizer.add_tokens(["[ne]", "[mai]"])
        print(f"   ➕ Added {num_added} custom tokens. Vocab size: {len(model.tokenizer)}")
    except Exception as e:
        print(f"   ⚠️ Could not setup tokenizer: {e}")

    # 3. Handle Vocabulary Resizing & Embeddings
    emb_path = os.path.join(ckpt_path, "text_emb.pt")
    if os.path.exists(emb_path):
        state = torch.load(emb_path, map_location=DEVICE, weights_only=True)
        # Handle both dict and tensor cases
        new_vocab_size = state["weight"].shape[0] if "weight" in state else state.shape[0]
        curr_vocab_size = model.t3.hp.text_tokens_dict_size
        
        if new_vocab_size != curr_vocab_size:
            print(f"   📏 Resizing T3 vocabulary: {curr_vocab_size} -> {new_vocab_size}")
            model.t3.resize_text_embeddings(new_vocab_size)
        
        print(f"   📥 Loading text embeddings from {emb_path}")
        model.t3.text_emb.load_state_dict(state)

    # 3. Load Text Head
    head_path = os.path.join(ckpt_path, "text_head.pt")
    if os.path.exists(head_path):
        print(f"   📥 Loading text head from {head_path}")
        head_state = torch.load(head_path, map_location=DEVICE, weights_only=True)
        model.t3.text_head.load_state_dict(head_state)

    # 4. Apply LoRA
    print(f"   💉 Injecting LoRA adapter from {ckpt_path}")
    model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, ckpt_path)
    model.t3.tfmr.to(DEVICE).eval()
    
    # Set flag for padding (not actually used by base model code anymore, but good to have)
    model.is_lora_active = True
    
    # Merge for inference speed on 4090
    try:
        model.t3.tfmr = model.t3.tfmr.merge_and_unload()
        print("   🚀 LoRA Merged for maximum inference speed.")
    except Exception as e:
        print(f"   ⚠️ Could not merge LoRA: {e}")

    print(f"✅ Checkpoint {ckpt_name} loaded successfully.")
    return model, f"Active: {ckpt_name}"

def generate_tts(
    model,
    text,
    language_id,
    audio_prompt_path,
    temperature,
    seed_num,
    min_p,
    top_p,
    top_k,
    repetition_penalty,
    norm_loudness
):
    if model is None:
        model = load_base_model()

    if seed_num != 0:
        set_seed(int(seed_num))

    # Prepend language tag if not already present AND if it's not English Base
    if language_id != "en":
        lang_tag = f"[{language_id}]"
        if not text.strip().startswith("["):
            text = f"{lang_tag} {text}"
            print(f"🏷️ Auto-tagged text: {text[:50]}...")
    else:
        # For English Base, ensure no tags are accidentally added
        print(f"🏷️ English Base selected, sending raw text...")

    print(f"🎙️ Synthesizing: {text[:100]}... | Temp: {temperature}")
    
    with torch.inference_mode():
        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            temperature=temperature,
            min_p=min_p,
            top_p=top_p,
            top_k=int(top_k),
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness,
        )
    
    # Convert to 16-bit PCM for Gradio
    audio_data = wav.squeeze(0).cpu().numpy()
    audio_data = np.clip(audio_data, -1, 1)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    return (model.sr, audio_data)

with gr.Blocks(title="Chatterbox Turbo Experimenter", css=CUSTOM_CSS) as demo:
    gr.Markdown("# ⚡ Chatterbox Turbo Experimenter\nTesting Nepali/Maithili Fine-tuned LoRA Checkpoints")

    model_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                value="नमस्ते, म नेपाली टर्बो मोडल हुँ। तपाईँलाई कस्तो छ?",
                label="Text to Synthesize",
                lines=5,
                elem_id="main_textbox"
            )

            with gr.Row(elem_classes=["tag-container"]):
                for tag in EVENT_TAGS:
                    btn = gr.Button(tag, elem_classes=["tag-btn"])
                    btn.click(fn=None, inputs=[btn, text_input], outputs=text_input, js=INSERT_TAG_JS)

            with gr.Row():
                language = gr.Dropdown(choices=list(SUPPORTED_LANGUAGES.keys()), value="ne", label="Target Language Tag")
                ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Voice (Min 5s required)")

            run_btn = gr.Button("Generate Speech ⚡", variant="primary")

        with gr.Column(scale=1):
            audio_out = gr.Audio(label="Output Audio")
            
            # Checkpoint selector
            default_ckpt = list(TURBO_CHECKPOINTS.keys())[-1] # Usually the latest epoch
            ckpt_selector = gr.Dropdown(
                choices=list(TURBO_CHECKPOINTS.keys()),
                value=default_ckpt,
                label="Turbo Checkpoint"
            )
            ckpt_status = gr.Markdown(f"**Status:** `Loading base...`")

            with gr.Accordion("Advanced Options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 2.0, step=.05, label="Temperature", value=0.8)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P", value=0.95)
                top_k = gr.Slider(0, 1000, step=10, label="Top K", value=1000)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.05, label="Repetition Penalty", value=1.2)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="Min P", value=0.00)
                norm_loudness = gr.Checkbox(value=True, label="Normalize Loudness")

    # Initialization & Events
    def init_app():
        m = load_base_model()
        # Auto-load the default (latest) checkpoint
        m, status = switch_checkpoint(m, default_ckpt)
        return m, status

    demo.load(fn=init_app, inputs=[], outputs=[model_state, ckpt_status])
    
    ckpt_selector.change(fn=switch_checkpoint, inputs=[model_state, ckpt_selector], outputs=[model_state, ckpt_status])
    
    run_btn.click(
        fn=generate_tts,
        inputs=[model_state, text_input, language, ref_wav, temp, seed_num, min_p, top_p, top_k, repetition_penalty, norm_loudness],
        outputs=audio_out,
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=True)
