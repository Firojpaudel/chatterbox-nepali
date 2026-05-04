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
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import librosa
import numpy as np
from safetensors.torch import save_file
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from huggingface_hub import HfApi
from datasets import load_dataset, Audio

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import S3_SR

class MultilingualStreamingDataset(IterableDataset):
    def __init__(self, repo_id, tokenizer, s3_tokenizer, voice_encoder, device):
        self.repo_id = repo_id
        self.tokenizer = tokenizer
        self.s3_tokenizer = s3_tokenizer
        self.voice_encoder = voice_encoder
        self.device = device
        
        # Load in streaming mode with shuffle buffer and explicit audio casting
        self.ds = load_dataset(repo_id, split="train", streaming=True)
        self.ds = self.ds.cast_column("audio", Audio(sampling_rate=S3_SR))
        self.ds = self.ds.shuffle(seed=42, buffer_size=1000)

    def __iter__(self):
        for item in self.ds:
            try:
                audio_data = item['audio']
                text = item.get('text') or ""
                lang = item.get('language', 'ne')

                # Process text
                text_tokens = self.tokenizer.text_to_tokens(text, language_id=lang, lowercase=False).squeeze(0)
                text_tokens = F.pad(text_tokens, (1, 1), value=0)
                text_tokens[0] = 255 # Start token

                # Process audio
                wav = audio_data['array']
                orig_sr = audio_data['sampling_rate']
                if orig_sr != S3_SR:
                    wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=S3_SR)
                
                with torch.no_grad():
                    speech_tokens, _ = self.s3_tokenizer.forward([wav])
                    speech_tokens = speech_tokens.squeeze(0)
                    ve_embed = torch.from_numpy(self.voice_encoder.embeds_from_wavs([wav], sample_rate=S3_SR))
                    ve_embed = ve_embed.mean(axis=0, keepdim=True)

                yield {
                    "text_tokens": text_tokens,
                    "speech_tokens": speech_tokens,
                    "speaker_emb": ve_embed
                }
            except Exception as e:
                print(f"⚠️ Skipping sample: {e}")
                continue

def collate_fn(batch):
    text_tokens = [item['text_tokens'] for item in batch]
    speech_tokens = [item['speech_tokens'] for item in batch]
    speaker_embs = [item['speaker_emb'] for item in batch]
    text_token_lens = torch.tensor([len(t) for t in text_tokens])
    speech_token_lens = torch.tensor([len(s) for s in speech_tokens])
    text_tokens_padded = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=0)
    speech_tokens_padded = torch.nn.utils.rnn.pad_sequence(speech_tokens, batch_first=True, padding_value=6562)
    speaker_embs = torch.cat(speaker_embs, dim=0)
    return {
        "text_tokens": text_tokens_padded,
        "text_token_lens": text_token_lens,
        "speech_tokens": speech_tokens_padded,
        "speech_token_lens": speech_token_lens,
        "speaker_emb": speaker_embs
    }

def train(args):
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        device = torch.device(args.device)
        rank = 0

    if rank == 0 and not args.no_wandb:
        wandb.init(project="chatterbox-multilingual-streaming")
        wandb.config.update(args)

    model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)
    t3 = model_wrapper.t3
    tokenizer = model_wrapper.tokenizer
    t3.resize_text_embeddings(len(tokenizer.tokenizer.get_vocab()))
    
    s3_tokenizer = model_wrapper.s3gen.tokenizer.cpu()
    voice_encoder = model_wrapper.ve.cpu()
    
    t3.to(device)
    
    # Optional: Resume from checkpoint
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"🔄 Resuming training from {args.resume_from}...")
            state_dict = torch.load(args.resume_from, map_location="cpu", weights_only=True)
            # Handle possible DDP prefix if needed, though state_dict is usually clean
            cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
            t3.load_state_dict(cleaned_state, strict=False)
        else:
            print(f"⚠️ Checkpoint {args.resume_from} not found. Starting from scratch.")

    if args.distributed:
        t3 = DDP(t3, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    t3.train()
    
    dataset = MultilingualStreamingDataset(args.repo_id, tokenizer, s3_tokenizer, voice_encoder, device="cpu")
    # For streaming, we don't use DistributedSampler, datasets handles it
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
    
    optimizer = AdamW(t3.parameters(), lr=args.lr)
    if device.type == "cuda":
        (t3.module if args.distributed else t3).tfmr.gradient_checkpointing_enable()
    
    use_amp = args.fp16 and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None
    
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if rank == 0 else dataloader
        for i, batch in enumerate(pbar):
            global_step += 1
            text_tokens = batch['text_tokens'].to(device)
            text_token_lens = batch['text_token_lens'].to(device)
            speech_tokens = batch['speech_tokens'].to(device)
            speech_token_lens = batch['speech_token_lens'].to(device)
            speaker_emb = batch['speaker_emb'].to(device)
            
            t3_model = t3.module if args.distributed else t3
            prompt_len = t3_model.hp.speech_cond_prompt_len
            cond_prompt_tokens = speech_tokens[:, :prompt_len] if speech_tokens.size(1) > prompt_len else None
            
            t3_cond = T3Cond(
                speaker_emb=speaker_emb,
                cond_prompt_speech_tokens=cond_prompt_tokens,
                emotion_adv=0.5 * torch.ones(text_tokens.size(0), 1, 1, device=device)
            )
            
            with autocast("cuda", enabled=use_amp, dtype=torch.float16):
                loss_text, loss_speech = t3_model.loss(t3_cond, text_tokens, text_token_lens, speech_tokens, speech_token_lens)
            
            loss = (loss_text + loss_speech) / args.accum_steps
            if scaler:
                scaler.scale(loss).backward()
                if (i+1) % args.accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (i+1) % args.accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            if rank == 0: 
                pbar.set_postfix({"loss": loss.item() * args.accum_steps})
                
                # Save step-based checkpoint every 1000 steps for precise tracking
                if global_step % 1000 == 0:
                    step_ckpt = f"checkpoint_step_{global_step}.pt"
                    torch.save(t3_model.state_dict(), step_ckpt)
                    if args.push_to_hub:
                        try: HfApi().upload_file(path_or_fileobj=step_ckpt, path_in_repo=step_ckpt, repo_id=args.push_to_hub, token=os.environ.get("HF_TOKEN"))
                        except: pass
        
        if rank == 0:
            ckpt_path = f"t3_multilingual_epoch_{epoch}.pt"
            torch.save(t3_model.state_dict(), ckpt_path)
            if args.push_to_hub:
                try: HfApi().upload_file(path_or_fileobj=ckpt_path, path_in_repo=ckpt_path, repo_id=args.push_to_hub, token=os.environ.get("HF_TOKEN"))
                except Exception as e: print(f"⚠️ Push failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="HF Dataset Repo ID")
    parser.add_argument("--push_to_hub", type=str, help="HF Model Repo ID to push checkpoints")
    parser.add_argument("--resume_from", type=str, help="Path to local checkpoint .pt file to resume from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8) # Lowered for T4
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1) # Lowered for streaming
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)
