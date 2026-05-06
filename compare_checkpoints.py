import torch
import os
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file as load_safetensors
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.s3tokenizer import drop_invalid_tokens
import soundfile as sf
import librosa
import numpy as np

# Configuration
CHECKPOINTS = {
    "Base Model": "pre-trained",
    "Step 27000 (Epoch 4)": "checkpoint_step_27000.safetensors", 
    "Step 29000 (Epoch 5 Mid)": "checkpoint_step_29000.safetensors",
}

# Updated TEST_CASES with proper language tags
TEST_CASES = [
    {"lang": "ne", "text": "नमस्ते, अब म धेरै राम्रोसँग र शुद्धसँग नेपाली बोल्न सक्छु।", "name": "nepali_fluency"},
    {"lang": "ne", "text": "यस अत्याधुनिक प्रविधिको प्रयोगले नेपाली भाषालाई अझै सुदृढ बनाउनेछ।", "name": "nepali_complex"},
    {"lang": "ne", "text": "आज मौसम कस्तो छ? मलाई लाग्छ आज पानी पर्छ।", "name": "nepali_casual"},
    {"lang": "en", "text": "This multilingual chatterbox model demonstrates impressive synthesis capabilities.", "name": "english_fluency"},
    {"lang": "en", "text": "Hello! How are you doing today? It is a beautiful day outside.", "name": "english_casual"},
    {"lang": "mai", "text": "अहाँक नाम की अछि? आ अहाँ कतय सँ अईलो अछि?", "name": "maithili_greeting"},
    {"lang": "mai", "text": "हमर गाम बहुत सुन्दर अछि। हमरा गामक लोक सब निक छथि।", "name": "maithili_descriptive"},
]

def compare():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")

    output_dir = Path("samples/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📦 Loading base model structure...")
    model = ChatterboxMultilingualTTS.from_pretrained(device)
    
    # --- FIX: Resize embeddings upfront to prevent CUDA crash ---
    # The tokenizer dynamically adds [ne] and [mai], expanding the vocab.
    # We must resize the model to match or it will crash on the GPU.
    tokenizer_vocab_size = len(model.tokenizer.tokenizer.get_vocab())
    print(f"  📏 Synchronizing model vocab size to tokenizer: {tokenizer_vocab_size}")
    model.t3.resize_text_embeddings(tokenizer_vocab_size)
    # -----------------------------------------------------------
    
    model.t3.to(device)
    model.s3gen.to(device)
    model.ve.to(device)

    ref_audio = "samples/Prakash.mp3"
    
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n--- Testing {ckpt_name} ---")
        
        is_finetuned = ckpt_path != "pre-trained"
        
        if is_finetuned:
            if not os.path.exists(ckpt_path):
                print(f"⚠️ Skipping {ckpt_name}: File {ckpt_path} not found.")
                continue
            
            state_dict = load_safetensors(ckpt_path)
            cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            if "text_emb.weight" in cleaned_state:
                checkpoint_vocab_size = cleaned_state["text_emb.weight"].shape[0]
                if checkpoint_vocab_size != model.t3.text_emb.weight.shape[0]:
                    print(f"  📏 Resizing vocabulary to {checkpoint_vocab_size}...")
                    model.t3.resize_text_embeddings(checkpoint_vocab_size)
            
            model.t3.load_state_dict(cleaned_state, strict=False)
        
        # Always use the model's config for start/stop tokens (255/0)
        current_sot = model.t3.hp.start_text_token
        current_eot = model.t3.hp.stop_text_token
        print(f"  🎯 Token IDs: Start={current_sot}, Stop={current_eot}, is_multilingual={model.t3.hp.is_multilingual}")
        
        model.t3.eval()

        def generate_with_pattern(text, lang, sot, eot):
            # 1. Get tokens (puts [lang] at index 0)
            text_tokens = model.tokenizer.text_to_tokens(text, language_id=lang.lower()).to(device)
            
            # Use the same pattern for both base and finetuned: [START] [LANG] [TEXT...] [STOP]
            # text_to_tokens already prepends [LANG], so we just wrap with START/STOP
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)

            # CFG doubling
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

            with torch.inference_mode():
                speech_tokens = model.t3.inference(
                    t3_cond=model.conds.t3,
                    text_tokens=text_tokens,
                    repetition_penalty=1.3,
                    cfg_weight=2.5,
                    temperature=0.7,
                    min_p=0.1,
                    max_new_tokens=400
                )
                speech_tokens = drop_invalid_tokens(speech_tokens[0])
                if speech_tokens.numel() == 0: return torch.zeros((1, 1, 1024))
                
                wav, _ = model.s3gen.inference(
                    speech_tokens=speech_tokens.to(device),
                    ref_dict=model.conds.gen,
                )
                return torch.from_numpy(wav.squeeze(0).detach().cpu().numpy()).unsqueeze(0)

        for test in TEST_CASES:
            print(f"  🔊 Generating: {test['name']} ({test['lang']})...")
            try:
                # Use original reference audio directly
                if os.path.exists(ref_audio):
                    model.prepare_conditionals(ref_audio)
                
                wav = generate_with_pattern(
                    test['text'], 
                    lang=test['lang'],
                    sot=current_sot,
                    eot=current_eot,
                )
                
                # Ultimate Tail Suppression
                audio_data = wav.squeeze().cpu().numpy()
                mse = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)[0]
                threshold = np.max(mse) * 0.05
                loud_frames = np.where(mse > threshold)[0]
                if len(loud_frames) > 0:
                    last_loud_sample = loud_frames[-1] * 512 + 1024
                    true_end = min(len(audio_data), last_loud_sample + int(0.1 * 24000))
                    audio_data = audio_data[:true_end]
                
                fade_len = int(0.05 * 24000)
                if len(audio_data) > fade_len:
                    audio_data[-fade_len:] *= np.linspace(1, 0, fade_len)
                
                filename = output_dir / f"{ckpt_name.replace(' ', '_').lower()}_{test['name']}.wav"
                sf.write(filename, audio_data, 24000)
                print(f"  ✅ Saved: {filename}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")

if __name__ == "__main__":
    compare()
