import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download
from pathlib import Path
import os
import argparse

def merge_models(base_path, fine_tuned_path, output_path, alpha=0.7):
    """
    Merges two models using linear interpolation.
    W_merged = (1 - alpha) * W_base + alpha * W_fine
    """
    print(f"Loading base model from {base_path}...")
    base_sd = load_file(base_path)
    
    print(f"Loading fine-tuned model from {fine_tuned_path}...")
    fine_sd = load_file(fine_tuned_path)
    
    merged_sd = {}
    
    # Identify keys
    all_keys = set(fine_sd.keys()) | set(base_sd.keys())
    
    for key in all_keys:
        if key in fine_sd and key in base_sd:
            w_base = base_sd[key]
            w_fine = fine_sd[key]
            
            if w_base.shape == w_fine.shape:
                # Standard linear interpolation
                merged_sd[key] = (1.0 - alpha) * w_base + alpha * w_fine
            else:
                print(f"Warning: Shape mismatch for '{key}': Base {w_base.shape} vs Fine {w_fine.shape}")
                # This usually happens for the embedding layer if tokens were added
                # We'll take the fine-tuned shape and fill the overlapping part
                new_w = w_fine.clone()
                common_rows = min(w_base.shape[0], w_fine.shape[0])
                common_cols = min(w_base.shape[1], w_fine.shape[1])
                
                # Blend the overlapping weights
                new_w[:common_rows, :common_cols] = (1.0 - alpha) * w_base[:common_rows, :common_cols] + alpha * w_fine[:common_rows, :common_cols]
                merged_sd[key] = new_w
                print(f"   -> Blended overlapping region of size {common_rows}x{common_cols}")
        
        elif key in fine_sd:
            print(f"Info: Key '{key}' only in Fine-tuned model. Keeping as is.")
            merged_sd[key] = fine_sd[key]
        else:
            print(f"Info: Key '{key}' only in Base model. Keeping as is.")
            merged_sd[key] = base_sd[key]
            
    print(f"Saving merged model to {output_path}...")
    save_file(merged_sd, output_path)
    print(f"Success! Merged model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge fine-tuned TTS model with base model to recover language capabilities.")
    parser.add_argument("--base", type=str, help="Path to base .safetensors file (optional, will download if missing)")
    parser.add_argument("--fine", type=str, default="t3_mtl_nepali_final.safetensors", help="Path to fine-tuned .safetensors")
    parser.add_argument("--output", type=str, default="t3_mtl_nepali_merged.safetensors", help="Path to save merged model")
    parser.add_argument("--alpha", type=float, default=0.7, help="Merge ratio (1.0 = 100% fine-tuned, 0.0 = 100% base)")
    
    args = parser.parse_args()
    
    base_model_path = args.base
    if not base_model_path:
        repo_id = "ResembleAI/chatterbox"
        print(f"Downloading base model from Hugging Face: {repo_id}...")
        ckpt_dir = snapshot_download(repo_id=repo_id, allow_patterns=["t3_mtl23ls_v2.safetensors"])
        base_model_path = os.path.join(ckpt_dir, "t3_mtl23ls_v2.safetensors")
    
    if not os.path.exists(args.fine):
        print(f"❌ Error: Fine-tuned model not found at {args.fine}")
        exit(1)
        
    merge_models(base_model_path, args.fine, args.output, alpha=args.alpha)
