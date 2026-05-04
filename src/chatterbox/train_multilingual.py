import sys
from unittest.mock import MagicMock

# --- KAGGLE/ENVIRONMENT FIX: Mock torchvision if it's broken ---
try:
    import torchvision
except (ImportError, RuntimeError) as e:
    if isinstance(e, ImportError) or "torchvision::nms" in str(e):
        mock_tv = MagicMock()
        mock_tv.__version__ = "0.0.0"
        sys.modules["torchvision"] = mock_tv
        sys.modules["torchvision.transforms"] = MagicMock()
        sys.modules["torchvision.ops"] = MagicMock()
        print("⚠️ Warning: torchvision is missing or broken. Mocked to allow transformers import.")

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import librosa
import numpy as np
from safetensors.torch import save_file, load_file as load_safetensors
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import HfApi

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import S3_SR, S3Tokenizer
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import MTLTokenizer

class MultilingualDataset(Dataset):
    def __init__(self, manifest_path, tokenizer, s3_tokenizer, voice_encoder, device, wav_dir=None):
        self.tokenizer = tokenizer
        self.s3_tokenizer = s3_tokenizer
        self.voice_encoder = voice_encoder
        self.device = device
        self.data = []
        
        self.manifest_path = Path(manifest_path)
        self.wav_dir = Path(wav_dir) if wav_dir else self.manifest_path.parent / "wavs"

        if self.manifest_path.suffix == '.jsonl':
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    raw_audio = item.get('audio_filepath') or item.get('audio_path') or item.get('audio')
                    if raw_audio:
                        resolved = self._resolve_path(raw_audio)
                        if resolved and os.path.exists(resolved):
                            item['audio_path'] = resolved
                            self.data.append(item)
                        else:
                            # print(f"⚠️ Warning: Could not find audio file for {raw_audio}. Skipping.")
                            pass
        else:
            print("⚠️ Only .jsonl manifest is supported for Multilingual training.")

    def _resolve_path(self, audio_val):
        p = Path(audio_val)
        if p.is_absolute() and p.exists(): return str(p)
        if p.exists(): return str(p.absolute())
        
        search_dirs = [self.wav_dir, self.manifest_path.parent, self.manifest_path.parent.parent / "wavs"]
        for base_dir in search_dirs:
            if not base_dir: continue
            cand = base_dir / p.name
            if cand.exists(): return str(cand.absolute())
            cand = base_dir / p
            if cand.exists(): return str(cand.absolute())
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item['audio_path']
        text = item.get('text') or ""
        lang = item.get('language', 'ne') # Support ne, mai, en from manifest

        # Dynamically use language_id from manifest
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=lang, lowercase=False).squeeze(0)
        
        # Pad [START] and [STOP]
        text_tokens = F.pad(text_tokens, (1, 0), value=255) # start_text_token
        text_tokens = F.pad(text_tokens, (0, 1), value=0)   # stop_text_token

        # Load audio
        wav, _ = librosa.load(audio_path, sr=S3_SR)
        
        with torch.no_grad():
            speech_tokens, _ = self.s3_tokenizer.forward([wav])
            speech_tokens = speech_tokens.squeeze(0)
            ve_embed = torch.from_numpy(self.voice_encoder.embeds_from_wavs([wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        return {
            "text_tokens": text_tokens,
            "speech_tokens": speech_tokens,
            "speaker_emb": ve_embed,
            "wav": wav
        }

def collate_fn(batch):
    text_tokens = [item['text_tokens'] for item in batch]
    speech_tokens = [item['speech_tokens'] for item in batch]
    speaker_embs = [item['speaker_emb'] for item in batch]
    text_token_lens = torch.tensor([len(t) for t in text_tokens])
    speech_token_lens = torch.tensor([len(s) for s in speech_tokens])
    text_tokens_padded = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=0)
    speech_tokens_padded = torch.nn.utils.rnn.pad_sequence(speech_tokens, batch_first=True, padding_value=6562) # stop_speech_token
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
        wandb.init(project="chatterbox-multilingual-finetune")
        wandb.config.update(args)

    # Load pretrained components
    model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)
    t3 = model_wrapper.t3
    tokenizer = model_wrapper.tokenizer
    
    # Resize embeddings if needed for new tokens ([mai] etc)
    t3.resize_text_embeddings(len(tokenizer.tokenizer.get_vocab()))
    
    s3_tokenizer = model_wrapper.s3gen.tokenizer.cpu()
    voice_encoder = model_wrapper.ve.cpu()
    
    # Optional: Initialize [mai] from [hi] as well?
    # Hindi is 722, Nepali is 2454, Maithili is 2455 (approx)
    # We'll just do Nepali for now or let them learn.
    
    t3.to(device)
    if args.distributed:
        t3 = DDP(t3, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    t3.train()
    
    dataset = MultilingualDataset(args.manifest, tokenizer, s3_tokenizer, voice_encoder, device="cpu", wav_dir=args.wav_dir)
    sampler = DistributedSampler(dataset) if args.distributed else None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    optimizer = AdamW(t3.parameters(), lr=args.lr)
    if device.type == "cuda":
        t3_model = t3.module if args.distributed else t3
        t3_model.tfmr.gradient_checkpointing_enable()
    
    use_amp = args.fp16 and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None
    
    for epoch in range(args.epochs):
        if args.distributed: sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if rank == 0 else dataloader
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            text_tokens = batch['text_tokens'].to(device)
            text_token_lens = batch['text_token_lens'].to(device)
            speech_tokens = batch['speech_tokens'].to(device)
            speech_token_lens = batch['speech_token_lens'].to(device)
            speaker_emb = batch['speaker_emb'].to(device)
            
            prompt_len = t3.module.hp.speech_cond_prompt_len if args.distributed else t3.hp.speech_cond_prompt_len
            cond_prompt_tokens = speech_tokens[:, :prompt_len] if speech_tokens.size(1) > prompt_len else None
            
            t3_cond = T3Cond(
                speaker_emb=speaker_emb,
                cond_prompt_speech_tokens=cond_prompt_tokens,
                emotion_adv=0.5 * torch.ones(text_tokens.size(0), 1, 1, device=device)
            )
            
            with autocast("cuda", enabled=use_amp, dtype=torch.float16):
                loss_text, loss_speech = t3.module.loss(t3_cond, text_tokens, text_token_lens, speech_tokens, speech_token_lens) if args.distributed else t3.loss(t3_cond, text_tokens, text_token_lens, speech_tokens, speech_token_lens)
            
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
        
        if rank == 0 and epoch % args.save_every == 0:
            ckpt_path = f"t3_multilingual_epoch_{epoch}.pt"
            torch.save(t3.module.state_dict() if args.distributed else t3.state_dict(), ckpt_path)
            if args.push_to_hub:
                try:
                    HfApi().upload_file(path_or_fileobj=ckpt_path, path_in_repo=ckpt_path, repo_id=args.push_to_hub, token=os.environ.get("HF_TOKEN"))
                except Exception as e: print(f"⚠️ Push failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--wav_dir", type=str)
    parser.add_argument("--push_to_hub", type=str)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)
