import sys
from unittest.mock import MagicMock

# --- KAGGLE/ENVIRONMENT FIX ---
try:
    import torchvision
except (ImportError, RuntimeError) as e:
    if isinstance(e, ImportError) or "torchvision::nms" in str(e):
        mock_tv = MagicMock()
        mock_tv.__version__ = "0.0.0"
        sys.modules["torchvision"] = mock_tv
        sys.modules["torchvision.transforms"] = MagicMock()
        sys.modules["torchvision.ops"] = MagicMock()

import argparse
import os
import json
import time
import io
import soundfile as sf
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from pathlib import Path
from datasets import load_dataset, Audio
from huggingface_hub import HfApi

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import S3_SR

# Strict mapping to ISO codes used by the base model
LANG_MAP = {
    "nepali": "ne",
    "ne-np": "ne",
    "ne": "ne",
    "maithili": "mai",
    "mai": "mai",
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "en": "en"
}

class BalancedMultilingualDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        
        print(f"Loading local dataset from {data_path}...")
        # Look for parquets in the directory and its subdirectories
        data_files = [str(p) for p in Path(data_path).rglob("*.parquet")]
        if not data_files:
            raise FileNotFoundError(f"No parquet files found in {data_path}")
        
        full_ds = load_dataset("parquet", data_files=data_files, split="train")
        full_ds = full_ds.cast_column("audio", Audio(sampling_rate=S3_SR, decode=False))
        
        # Split into language buckets for balanced sampling
        self.buckets = {"en": [], "ne": [], "mai": []}
        
        print("Sorting data into language buckets...")
        for i in tqdm(range(len(full_ds))):
            item = full_ds[i]
            raw_lang = str(item.get('language', 'ne')).lower()
            lang = LANG_MAP.get(raw_lang, "ne")
            
            if lang in self.buckets:
                self.buckets[lang].append(i)
        
        self.full_ds = full_ds
        self.langs = ["en", "ne", "mai"]
        # We target a 20% English, 40% Nepali, 40% Maithili split per epoch
        self.total_size = len(full_ds)
        
        for lang in self.langs:
            print(f"  Bucket {lang}: {len(self.buckets[lang])} samples")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Weighted random sampling to ensure balance
        # 20% chance for 'en', 40% for 'ne', 40% for 'mai'
        r = np.random.random()
        if r < 0.2:
            lang = "en"
        elif r < 0.6:
            lang = "ne"
        else:
            lang = "mai"
            
        # If bucket is empty, fallback to Nepali
        if not self.buckets[lang]: lang = "ne"
        
        # Pick a random sample from the chosen bucket
        ds_idx = np.random.choice(self.buckets[lang])
        item = self.full_ds[ds_idx]
        
        audio_data = item['audio']
        text = item.get('text') or ""
        
        # Prepend the correct ISO tag for the model
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=lang).squeeze(0)
        
        # Manual padding to match Chatterbox inference [START] text... [STOP]
        text_tokens = F.pad(text_tokens, (1, 0), value=255) # [START]
        text_tokens = F.pad(text_tokens, (0, 1), value=0)   # [STOP]

        # Process audio
        if isinstance(audio_data, dict) and 'bytes' in audio_data and audio_data['bytes'] is not None:
            with io.BytesIO(audio_data['bytes']) as f:
                wav, orig_sr = sf.read(f)
            if orig_sr != S3_SR:
                import torchaudio.functional as F_audio
                wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
                wav_t = F_audio.resample(wav_t, orig_sr, S3_SR)
                wav = wav_t.squeeze(0).numpy()
        else:
            wav = audio_data['array']
        
        wav = wav.astype(np.float32)
        if len(wav) > 15 * S3_SR: wav = wav[:15 * S3_SR] # Clip long audio

        return {
            "text_tokens": text_tokens,
            "wav": wav,
            "lang": lang
        }

def collate_fn(batch):
    # This matches the inference pattern exactly
    text_tokens = [item['text_tokens'] for item in batch]
    wavs = [item['wav'] for item in batch]
    
    text_token_lens = torch.tensor([len(t) for t in text_tokens])
    # Pad text with 705 [PAD] (standard Llama pad)
    text_tokens_padded = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=705)
    
    return {
        "text_tokens": text_tokens_padded,
        "text_token_lens": text_token_lens,
        "wavs": wavs
    }

def save_checkpoint(epoch, model, args, is_initial=False):
    ckpt_name = f"t3_v4_epoch_{epoch}.pt"
    torch.save(model.state_dict(), ckpt_name)
    print(f"💾 Saved {ckpt_name} locally")
    
    if args.push_to_hub and not is_initial:
        try:
            api = HfApi()
            token = os.environ.get("HF_TOKEN")
            print(f"🚀 Pushing {ckpt_name} to {args.push_to_hub}...")
            api.upload_file(
                path_or_fileobj=ckpt_name,
                path_in_repo=ckpt_name,
                repo_id=args.push_to_hub,
                token=token
            )
            print(f"✅ Pushed to {args.push_to_hub}")
            
            # Cleanup previous local checkpoint only after successful push
            prev_ckpt = f"t3_v4_epoch_{epoch-1}.pt"
            if os.path.exists(prev_ckpt):
                os.remove(prev_ckpt)
                print(f"🧹 Cleaned up local {prev_ckpt}")
        except Exception as e:
            print(f"⚠️ Hub push failed: {e}")

def train(args):
    device = torch.device(args.device)
    if not args.no_wandb:
        wandb.init(project="chatterbox-multilingual-v4", config=args)

    print("Loading Base model...")
    model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)
    t3 = model_wrapper.t3
    tokenizer = model_wrapper.tokenizer
    s3_tokenizer = model_wrapper.s3gen.tokenizer.to(device)
    ve = model_wrapper.ve.to(device)
    
    # 1. Resize embeddings to include new tags [ne], [mai]
    print("Resizing text embeddings...")
    t3.resize_text_embeddings(len(tokenizer.tokenizer.get_vocab()))
    
    # 2. THE SECRET SAUCE: Hindi weight cloning
    # Copy [hi] weights to [ne] and [mai] to kill start-up noise
    vocab = tokenizer.tokenizer.get_vocab()
    hi_idx = vocab.get("[hi]")
    ne_idx = vocab.get("[ne]")
    mai_idx = vocab.get("[mai]")
    
    if hi_idx is not None:
        with torch.no_grad():
            for target_idx, name in [(ne_idx, "[ne]"), (mai_idx, "[mai]")]:
                if target_idx is not None:
                    print(f"🎯 Cloning [hi] weights ({hi_idx}) to {name} ({target_idx})...")
                    t3.text_emb.weight[target_idx] = t3.text_emb.weight[hi_idx].clone()
                    t3.text_head.weight[target_idx] = t3.text_head.weight[hi_idx].clone()
    
    # 3. Resume from checkpoint if provided
    start_epoch = 1
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"♻️ Resuming from {args.resume_from}...")
        state = torch.load(args.resume_from, map_location=device, weights_only=True)
        t3.load_state_dict(state)
        # Detect epoch from filename (e.g., t3_v4_epoch_2.pt -> starts at 3)
        import re
        match = re.search(r"epoch_(\d+)", args.resume_from)
        if match:
            start_epoch = int(match.group(1)) + 1
            print(f"📈 Detected checkpoint from epoch {match.group(1)}. Starting from Epoch {start_epoch}")
    
    t3.to(device).train()
    
    # Unfreeze ALL layers for proper linguistic learning
    for p in t3.parameters(): p.requires_grad = True
    print("⚡ Full fine-tuning enabled (30 layers unfrozen)")

    if device.type == "cuda":
        t3.tfmr.gradient_checkpointing_enable()
        print("🧠 Gradient checkpointing enabled to save memory")

    dataset = BalancedMultilingualDataset(args.data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    
    optimizer = AdamW(t3.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = GradScaler("cuda", enabled=True)
    
    # Simulating large batch 128
    # Effective Batch = batch_size (8) * accum_steps (16) = 128
    accum_steps = args.accum_steps 
    
    # Save initial baseline only if we are starting from scratch
    if start_epoch == 1:
        save_checkpoint(0, t3, args, is_initial=True)
    
    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        
        for i, batch in enumerate(pbar):
            text_tokens = batch['text_tokens'].to(device)
            text_token_lens = batch['text_token_lens'].to(device)
            wavs = batch['wavs']
            
            # Feature extraction (Speech tokens + Speaker Emb)
            with torch.no_grad():
                speech_tokens, _ = s3_tokenizer.forward(wavs)
                speech_tokens = speech_tokens.to(device)
                speech_token_lens = torch.tensor([len(s) for s in speech_tokens], device=device)
                
                speaker_emb = torch.from_numpy(ve.embeds_from_wavs(wavs, sample_rate=S3_SR)).to(device)
            
            # Prepare conditioning (0.5 exaggeration as baseline)
            t3_cond = T3Cond(
                speaker_emb=speaker_emb,
                cond_prompt_speech_tokens=speech_tokens[:, :t3.hp.speech_cond_prompt_len],
                emotion_adv=0.5 * torch.ones(len(wavs), 1, 1, device=device)
            )
            
            with autocast("cuda", dtype=torch.bfloat16):
                loss_text, loss_speech = t3.loss(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_token_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_token_lens
                )
                loss = (loss_text + (loss_speech * 3.0)) / accum_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(t3.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                
                if not args.no_wandb:
                    wandb.log({
                        "loss": loss.item() * accum_steps,
                        "L_txt": loss_text.item(),
                        "L_spch": loss_speech.item(),
                        "step": global_step
                    })
            
            pbar.set_postfix({"L": loss.item() * accum_steps})
            
            # Memory cleanup
            del loss, loss_text, loss_speech, t3_cond
            if (i + 1) % 10 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()
            
        # Save and push current epoch
        save_checkpoint(epoch, t3, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=16) # Effective batch 128
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--push_to_hub", type=str, help="Hugging Face repo ID")
    parser.add_argument("--resume_from", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args)
