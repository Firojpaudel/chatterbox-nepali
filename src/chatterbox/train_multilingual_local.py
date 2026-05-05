import sys
from unittest.mock import MagicMock

# --- ENVIRONMENT FIX ---
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
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm
import torch.distributed as dist
import io
import soundfile as sf
import librosa
import numpy as np
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from huggingface_hub import HfApi
from datasets import load_dataset, Audio

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import S3_SR

class MultilingualDataset(Dataset):
    def __init__(self, repo_id, tokenizer):
        self.repo_id = repo_id
        self.tokenizer = tokenizer
        
        print(f"📦 Loading dataset {repo_id} to local disk...")
        self.ds = load_dataset(repo_id, split="train", streaming=False)
        self.ds = self.ds.cast_column("audio", Audio(sampling_rate=S3_SR, decode=False))
        print(f"✅ Loaded {len(self.ds)} samples.")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        try:
            audio_data = item['audio']
            text = item.get('text') or ""
            lang = item.get('language', 'ne')

            # Process text
            text_tokens = self.tokenizer.text_to_tokens(text, language_id=lang, lowercase=False).squeeze(0)
            text_tokens = F.pad(text_tokens, (1, 1), value=0)
            text_tokens[0] = 255 # Start token

            # Process audio (Manual Decoding)
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
            
            # Cap duration to avoid OOM (20 seconds)
            if len(wav) > 20 * S3_SR:
                wav = wav[:20 * S3_SR]

            return {
                "text_tokens": text_tokens,
                "wav": wav,
                "lang": lang
            }
        except Exception as e:
            print(f"⚠️ Skipping sample {idx}: {e}")
            # Return a small silent sample instead of None to avoid batching issues
            return {
                "text_tokens": torch.zeros(5, dtype=torch.long),
                "wav": np.zeros(S3_SR, dtype=np.float32),
                "lang": "ne"
            }

def collate_fn(batch):
    text_tokens = [item['text_tokens'] for item in batch]
    wavs = [item['wav'] for item in batch]
    
    text_token_lens = torch.tensor([len(t) for t in text_tokens])
    text_tokens_padded = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=0)
    
    return {
        "text_tokens": text_tokens_padded,
        "text_token_lens": text_token_lens,
        "wavs": wavs
    }

def train(args):
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device(args.device)
        rank = 0
        world_size = 1

    if rank == 0 and not args.no_wandb:
        wandb.init(project="chatterbox-multilingual-local")
        wandb.config.update(args)

    model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)
    t3 = model_wrapper.t3
    tokenizer = model_wrapper.tokenizer
    t3.resize_text_embeddings(len(tokenizer.tokenizer.get_vocab()))
    
    t3.to(device)
    
    # Auxiliary models for feature extraction
    s3_tokenizer = model_wrapper.s3gen.tokenizer.to(device)
    voice_encoder = model_wrapper.ve.to(device)

    # Optional: Resume from checkpoint
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"🔄 Resuming training from {args.resume_from}...")
            state_dict = torch.load(args.resume_from, map_location="cpu", weights_only=True)
            cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
            t3.load_state_dict(cleaned_state, strict=False)
        else:
            print(f"⚠️ Checkpoint {args.resume_from} not found. Starting from scratch.")

    if args.distributed:
        t3 = DDP(t3, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    t3.train()
    
    dataset = MultilingualDataset(args.repo_id, tokenizer)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if args.distributed else None
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn, 
        num_workers=args.num_workers, 
        pin_memory=True,
        sampler=sampler,
        shuffle=(sampler is None)
    )
    
    optimizer = AdamW(t3.parameters(), lr=args.lr)
    if device.type == "cuda":
        (t3.module if args.distributed else t3).tfmr.gradient_checkpointing_enable()
    
    use_amp = (args.fp16 or args.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = GradScaler("cuda", enabled=args.fp16) if args.fp16 else None
    
    global_step = 0
    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if rank == 0 else dataloader
        
        for i, batch in enumerate(pbar):
            global_step += 1
            text_tokens = batch['text_tokens'].to(device)
            text_token_lens = batch['text_token_lens'].to(device)
            wavs = batch['wavs']
            
            # Feature extraction on GPU (Batch)
            with torch.no_grad():
                speech_tokens, _ = s3_tokenizer.forward(wavs)
                speech_tokens = speech_tokens.to(device)
                speech_token_lens = torch.tensor([len(s) for s in speech_tokens], device=device)
                speech_tokens = torch.nn.utils.rnn.pad_sequence(speech_tokens, batch_first=True, padding_value=6562)
                
                ve_embeds = voice_encoder.embeds_from_wavs(wavs, sample_rate=S3_SR)
                if isinstance(ve_embeds, list):
                    speaker_emb = torch.stack([torch.from_numpy(v).to(device).mean(dim=0) for v in ve_embeds])
                else:
                    ve_embeds_t = torch.from_numpy(ve_embeds).to(device)
                    speaker_emb = ve_embeds_t.mean(dim=1) if len(ve_embeds_t.shape) == 3 else ve_embeds_t
            
            t3_model = t3.module if args.distributed else t3
            prompt_len = t3_model.hp.speech_cond_prompt_len
            cond_prompt_tokens = speech_tokens[:, :prompt_len] if speech_tokens.size(1) > prompt_len else None
            
            t3_cond = T3Cond(
                speaker_emb=speaker_emb,
                cond_prompt_speech_tokens=cond_prompt_tokens,
                emotion_adv=0.5 * torch.ones(text_tokens.size(0), 1, 1, device=device)
            )
            
            with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss_text, loss_speech = t3_model.loss(
                    t3_cond=t3_cond, 
                    text_tokens=text_tokens, 
                    text_token_lens=text_token_lens, 
                    speech_tokens=speech_tokens, 
                    speech_token_lens=speech_token_lens
                )
            
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
                if not args.no_wandb:
                    wandb.log({
                        "loss": loss.item() * args.accum_steps,
                        "global_step": global_step,
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]['lr']
                    })
            
            # Periodic cache clearing to prevent fragmentation
            if (i+1) % 10 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                if rank == 0 and global_step % 1000 == 0:
                    step_ckpt = f"checkpoint_step_{global_step}.pt"
                    torch.save(t3_model.state_dict(), step_ckpt)
        
        if rank == 0 and epoch % args.save_every == 0:
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (recommended for RTX 30/40 series)")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)
