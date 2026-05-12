import json
import os
import sys
import torch
import soundfile as sf
from peft import PeftModel

# Ensure src is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

INPUT_FILE = os.path.join(CURRENT_DIR, "banking_test_cases.jsonl")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output_audio")
CKPT_DIR = os.path.join(PROJECT_ROOT, "lora_nepali_epoch_4")
REF_AUDIO = os.path.join(PROJECT_ROOT, "samples/Prakash.mp3")

def run_inference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Loading model on {device}...")
    
    model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)
    t3 = model_wrapper.t3
    
    # Resize vocab
    vocab_size = len(model_wrapper.tokenizer.tokenizer.get_vocab())
    t3.resize_text_embeddings(vocab_size)
    
    # Load LoRA
    print(f"🎨 Applying LoRA from: {CKPT_DIR}")
    t3.tfmr = PeftModel.from_pretrained(t3.tfmr, CKPT_DIR)
    
    # Load embeddings
    emb_path = os.path.join(CKPT_DIR, "text_emb.pt")
    if os.path.exists(emb_path):
        t3.text_emb.load_state_dict(torch.load(emb_path, map_location=device, weights_only=True))
    
    t3.eval()
    
    # Process a subset for quick testing
    samples = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            samples.append(json.loads(line))

    print(f"Generating {len(samples)} audio samples...")
    
    for i, sample in enumerate(samples):
        text = sample["sanitized_text"]
        lang = sample["language"]
        category = sample["category"].replace(" ", "_").replace("(", "").replace(")", "")
        # Map "codemix" to "ne" for model inference
        inf_lang = "en" if lang == "english" else "ne"
        
        output_path = os.path.join(OUTPUT_DIR, f"{i:03d}_{category}_{lang}.wav")
        print(f"[{i+1}/{len(samples)}] Processing [{lang}]: {text[:50]}...")
        
        if os.path.exists(output_path):
            print(f"⏭️ Skipping already exists: {output_path}")
            continue

        with torch.no_grad():
            wav = model_wrapper.generate(
                text,
                language_id=inf_lang,
                audio_prompt_path=REF_AUDIO,
                temperature=0.7
            )
            
        sf.write(output_path, wav.squeeze(0).cpu().numpy(), model_wrapper.sr)
        print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    run_inference()
