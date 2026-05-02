import os
import torch
import soundfile as sf
import glob
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path("src").absolute()))

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ImportError:
    print("Please install requirements: pip install -e .")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path to the downloaded weights (from previous command output)
# I'll just search for it in the cache to be sure
# Path to the merged weights
model_path = "t3_mtl_nepali_merged.safetensors"
if not os.path.exists(model_path):
    print(f"Merged model {model_path} not found.")
    sys.exit(1)
print(f"Loading merged weights from {model_path}...")

# Load the base model (downloads other components from ResembleAI/chatterbox)
model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)

# Load the fine-tuned T3 weights
from safetensors.torch import load_file
state_dict = load_file(model_path)
cleaned_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in state_dict.items()}
model_wrapper.t3.load_state_dict(cleaned_state, strict=False)
model_wrapper.t3.to(device).eval()

# Voice prompt
prompt_wav = "samples/achyut_ref_10s.wav"
if not os.path.exists(prompt_wav):
    # Fallback to any wav in samples if achyut_ref_10s.wav is missing
    alt_wavs = glob.glob("samples/*.wav")
    if alt_wavs:
        prompt_wav = alt_wavs[0]
        print(f"Using alternative prompt: {prompt_wav}")
    else:
        print(f"No prompt wav found in samples/")
        sys.exit(1)

# 1. Test English
en_text = "Hello! This is a test to verify that I can still speak English clearly after being merged with the base model. The goal is to check for catastrophic forgetting recovery."
print(f"Generating English: {en_text}")
en_wav = model_wrapper.generate(en_text, language_id="en", audio_prompt_path=prompt_wav)
# Convert from torch tensor [1, T] to numpy [T]
sf.write("test_english_merged.wav", en_wav.squeeze(0).cpu().numpy(), model_wrapper.sr)

# 2. Test Nepali
ne_text = "नमस्ते, म नेपालीमा पनि राम्रोसँग बोल्न सक्छु। मेरो आवाज कस्तो छ?"
print("Generating Nepali...") # Removed ne_text from print to avoid console encoding issues
import time
start = time.time()
ne_wav = model_wrapper.generate(ne_text, language_id="ne", audio_prompt_path=prompt_wav)
end = time.time()
sf.write("test_nepali_local.wav", ne_wav.squeeze(0).cpu().numpy(), model_wrapper.sr)
print(f"Nepali generation took {end - start:.2f} seconds")

print("\nDONE! Samples saved as 'test_english_merged.wav' and 'test_nepali_local.wav'")
