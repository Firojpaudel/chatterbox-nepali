import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import sys
from pathlib import Path

def inspect_model():
    print("Loading model for inspection...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the base model
    model = ChatterboxMultilingualTTS.from_pretrained(device)
    
    output_file = Path("model_architecture.txt")
    
    print(f"Writing architecture to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("========================================================\n")
        f.write("      CHATTERBOX MULTILINGUAL TTS ARCHITECTURE\n")
        f.write("========================================================\n\n")
        
        f.write("--- Overall Model Wrapper ---\n")
        f.write(str(model))
        f.write("\n\n")
        
        f.write("--- T3 Backbone (Text-to-Speech LLM) ---\n")
        f.write(str(model.t3))
        f.write("\n\n")
        
        f.write("--- S3Gen (Vocoder / Tokenizer) ---\n")
        f.write(str(model.s3gen))
        f.write("\n\n")
        
        f.write("--- Voice Encoder (VE) ---\n")
        f.write(str(model.ve))
        f.write("\n\n")
        
        # Count parameters
        t3_params = sum(p.numel() for p in model.t3.parameters())
        s3gen_params = sum(p.numel() for p in model.s3gen.parameters())
        ve_params = sum(p.numel() for p in model.ve.parameters())
        total_params = t3_params + s3gen_params + ve_params
        
        f.write("--- Parameter Summary ---\n")
        f.write(f"T3 Backbone:   {t3_params:,} parameters\n")
        f.write(f"S3Gen Vocoder: {s3gen_params:,} parameters\n")
        f.write(f"Voice Encoder: {ve_params:,} parameters\n")
        f.write(f"----------------------------------------\n")
        f.write(f"TOTAL:         {total_params:,} parameters\n")

    print(f"Done! Architecture saved to {output_file.absolute()}")

if __name__ == "__main__":
    inspect_model()
