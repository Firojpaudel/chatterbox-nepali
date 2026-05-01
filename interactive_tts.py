import os
import torch
import soundfile as sf
import sys
import glob
from pathlib import Path

# Add src to path
sys.path.append(str(Path("src").absolute()))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Chatterbox Interactive CLI ---")
    print(f"Using device: {device}")

    # 1. Load Model
    # Try to find the local fine-tuned model first
    cache_dir = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub" / "models--Firoj112--chatterbox-nepali-runs"
    weights = list(cache_dir.glob("**/t3_nepali_epoch_30.pt"))
    
    model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)
    
    if weights:
        model_path = str(weights[0])
        print(f"Applying fine-tuned weights from: {model_path}")
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        cleaned_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in state_dict.items()}
        model_wrapper.t3.load_state_dict(cleaned_state, strict=False)
    else:
        print("Using base multilingual weights (no fine-tuned checkpoint found).")
    
    model_wrapper.t3.to(device).eval()

    # 2. Interactive Loop
    while True:
        print("\n" + "="*40)
        full_text = input("Enter text (or 'exit' to quit): ").strip()
        if full_text.lower() == 'exit':
            break
        if not full_text:
            continue

        # Language selection
        lang = "ne" if any(c in full_text for c in "\u0900-\u097F") else "en"
        print(f"Detected language: {lang} (Enter to confirm, or type 'en'/'ne')")
        lang_input = input(f"Language [{lang}]: ").strip().lower()
        if lang_input:
            lang = lang_input

        # Voice selection
        prompts = glob.glob("samples/*.wav")
        prompt_path = prompts[0] if prompts else None
        if not prompt_path:
            prompt_path = input("Enter path to reference .wav: ").strip()

        # --- CHUNKING LOGIC ---
        # Split by Nepali full stop (।), English period (.), Question mark (?), Exclamation (!)
        import re
        chunks = re.split(r'([।\.?!\n])', full_text)
        
        # Recombine delimiters with sentences
        processed_chunks = []
        current = ""
        for i in range(0, len(chunks)-1, 2):
            processed_chunks.append(chunks[i] + chunks[i+1])
        if len(chunks) % 2 == 1:
            processed_chunks.append(chunks[-1])
        
        # Filter out empty or whitespace-only chunks
        processed_chunks = [c.strip() for c in processed_chunks if c.strip()]
        
        if not processed_chunks:
            processed_chunks = [full_text]

        print(f"\nProcessing {len(processed_chunks)} sentences...")
        all_wavs = []
        
        try:
            for i, chunk in enumerate(processed_chunks):
                print(f"  [{i+1}/{len(processed_chunks)}] Synthesizing: {chunk[:30]}...")
                with torch.no_grad():
                    wav = model_wrapper.generate(
                        chunk, 
                        language_id=lang, 
                        audio_prompt_path=prompt_path
                    )
                    all_wavs.append(wav.squeeze(0).cpu())
            
            # Combine all chunks
            if all_wavs:
                final_wav = torch.cat(all_wavs, dim=-1)
                output_file = "output.wav"
                sf.write(output_file, final_wav.numpy(), model_wrapper.sr)
                print(f"\n✅ Done! Combined audio saved to {os.path.abspath(output_file)}")
                
        except Exception as e:
            print(f"❌ Error during generation: {e}")

if __name__ == "__main__":
    main()
