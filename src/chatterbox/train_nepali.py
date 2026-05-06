import sys
from unittest.mock import MagicMock

# --- KAGGLE/ENVIRONMENT FIX: Mock torchvision if it's broken ---
# This prevents 'RuntimeError: operator torchvision::nms does not exist' which blocks transformers imports.
try:
    import torchvision
except (ImportError, RuntimeError) as e:
    # If torchvision is missing or throws a registration error (common on Kaggle with torch mismatch)
    # we mock it because Chatterbox doesn't actually use torchvision.
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

class NepaliDataset(Dataset):
    def __init__(self, manifest_path, tokenizer, s3_tokenizer, voice_encoder, device, wav_dir=None):
        self.tokenizer = tokenizer
        self.s3_tokenizer = s3_tokenizer
        self.voice_encoder = voice_encoder
        self.device = device
        self.data = []
        
        self.manifest_path = Path(manifest_path)
        # If wav_dir not provided, we try to look in a 'wavs' folder next to the manifest
        self.wav_dir = Path(wav_dir) if wav_dir else self.manifest_path.parent / "wavs"

        if self.manifest_path.suffix == '.jsonl':
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # Support multiple key names for audio
                    raw_audio = item.get('audio_path') or item.get('audio') or item.get('audio_filepath')
                    if raw_audio:
                        resolved = self._resolve_path(raw_audio)
                        if resolved and os.path.exists(resolved):
                            item['audio_path'] = resolved
                            self.data.append(item)
                        else:
                            print(f"⚠️ Warning: Could not find audio file for {raw_audio}. Skipping.")
        else:
            # Handle CSV (usually | for this dataset) or TSV (\t)
            delimiter = '\t' if self.manifest_path.suffix == '.tsv' else '|'
            
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(delimiter)
                    if len(parts) >= 2:
                        fname = parts[0]
                        text = parts[1]
                        # Support potential [ne] tag in text or just raw text
                        audio_path = self._resolve_path(fname)
                        if audio_path and os.path.exists(audio_path):
                            self.data.append({"audio_path": str(audio_path), "text": text})
                        else:
                            print(f"⚠️ Warning: Could not find audio file for {fname}. Skipping.")
            
            if not self.data:
                print(f"⚠️ Warning: No data loaded from {self.manifest_path}. Check delimiters or wav path.")
            else:
                print(f"✅ Loaded {len(self.data)} items from {self.manifest_path}")

    def _resolve_path(self, audio_val):
        """Aggressively resolves audio filename or path against multiple candidates."""
        p = Path(audio_val)
        
        # 1. Absolute path check
        if p.is_absolute() and p.exists():
            return str(p)
        
        # 2. Relative to CWD
        if p.exists():
            return str(p.absolute())
            
        # 3. Candidate directories to search
        search_dirs = [
            self.wav_dir,                               # Explicit --wav_dir or manifest_dir/wavs
            self.manifest_path.parent,                  # Same dir as manifest
            self.manifest_path.parent.parent / "wavs",   # One level up / wavs (common)
            self.manifest_path.parent.parent,           # One level up (common)
            Path("finetuning_data/wavs"),               # Specific to the current Kaggle setup
            Path("/kaggle/input/voxcpm-nepali-data/wavs") # Potential Kaggle input path
        ]
        
        for base_dir in search_dirs:
            if not base_dir: continue
            
            # Try as a direct child (filename only)
            cand = base_dir / p.name
            if cand.exists():
                return str(cand.absolute())
            
            # Try as a relative path from this base
            cand = base_dir / p
            if cand.exists():
                return str(cand.absolute())

        # If it has no extension, try adding .wav
        if not p.suffix:
            return self._resolve_path(audio_val + ".wav")

        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item['audio_path']
        text = item.get('text') or item.get('transcript') or ""

        # Do NOT prepend [START] in text since MTLTokenizer adds [ne] as prefix to the text.
        # This will misorder tokens as [ne] [START] instead of inference which uses [START] [ne].
        # Better to pad the tokens manually.
        text_tokens = self.tokenizer.text_to_tokens(text, language_id='ne', lowercase=False).squeeze(0)
        
        # T3 expects [START] and [STOP] tokens (IDs 255 and 0)
        # Pad them manually exactly as inference `mtl_tts.py -> generate()` does
        text_tokens = F.pad(text_tokens, (1, 0), value=255) # start_text_token
        text_tokens = F.pad(text_tokens, (0, 1), value=0)   # stop_text_token

        # Load audio
        wav, _ = librosa.load(audio_path, sr=S3_SR)
        
        # Audio to speech tokens
        with torch.no_grad():
            speech_tokens, _ = self.s3_tokenizer.forward([wav])
            speech_tokens = speech_tokens.squeeze(0)
            
            # Speaker embedding
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
    
    # Pad sequences
    text_tokens_padded = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=705)  # [PAD] token, not [STOP]
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
        wandb.init(project="chatterbox-nepali-finetune")
        wandb.config.update(args)

    # Load pretrained components
    if args.ckpt_dir:
        model_wrapper = ChatterboxMultilingualTTS.from_local(args.ckpt_dir, device)
    else:
        model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)
    
    t3 = model_wrapper.t3
    tokenizer = model_wrapper.tokenizer
    
    # Optional: Resume from an intermediate training checkpoint
    if args.resume_t3_weights:
        print(f"🔄 Resuming training from {args.resume_t3_weights}...")
        resume_state = torch.load(args.resume_t3_weights, map_location="cpu", weights_only=True)
        
        # Clean state dict keys (strip 'patched_model.' and 'model.' if they exist)
        cleaned_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in resume_state.items()}
        
        # Handle vocabulary size potentially being different (e.g. if you added [ne])
        current_vocab_size = t3.hp.text_tokens_dict_size
        ckpt_vocab_size = cleaned_state["text_emb.weight"].shape[0]
        if current_vocab_size != ckpt_vocab_size:
            t3.resize_text_embeddings(ckpt_vocab_size)
            t3.load_state_dict(cleaned_state, strict=False)
            t3.resize_text_embeddings(current_vocab_size)
        else:
            t3.load_state_dict(cleaned_state, strict=False)
        t3.to(device)
    
    # For data loading, we keep feature extractors strictly on CPU
    s3_tokenizer = model_wrapper.s3gen.tokenizer.cpu()
    voice_encoder = model_wrapper.ve.cpu()
    
    # OPTIMIZATION: Initialize [ne] (Nepali) tag from [hi] (Hindi) tag
    # This prevents the initial gibberish by starting with a related language!
    hi_idx = 722
    ne_idx = 2454
    with torch.no_grad():
        if not args.skip_hi_init and t3.text_emb.weight.shape[0] > ne_idx:
            print(f"🎯 Initializing [ne] tag ({ne_idx}) weights from [hi] tag ({hi_idx})...")
            # 1. Update Embedding
            t3.text_emb.weight[ne_idx] = t3.text_emb.weight[hi_idx].clone()
            # 2. Update Prediction Head
            t3.text_head.weight[ne_idx] = t3.text_head.weight[hi_idx].clone()
        elif args.skip_hi_init:
            print(f"❄️ Skipping [hi] tag initialization as requested. [ne] will be learned from scratch.")

    t3.to(device)
    
    if args.distributed:
        t3 = DDP(t3, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    t3.train()
    
    # Dataset
    dataset = NepaliDataset(args.manifest, tokenizer, s3_tokenizer, voice_encoder, device="cpu", wav_dir=args.wav_dir)
    
    sampler = DistributedSampler(dataset) if args.distributed else None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type != "cpu" else False
    )
    
    optimizer = AdamW(t3.parameters(), lr=args.lr)
    
    # Gradient checkpointing: recompute activations during backward to save ~60% memory
    if device.type == "cuda":
        t3_model = t3.module if args.distributed else t3
        t3_model.tfmr.gradient_checkpointing_enable()
        if rank == 0:
            print("🧠 Gradient checkpointing enabled")
    
    # Mixed precision for memory savings (critical for T4 GPUs)
    use_amp = args.fp16 and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None
    if use_amp and rank == 0:
        print("⚡ Mixed precision (float16) enabled")
    
    start_epoch = 0
    if args.resume_t3_weights:
        import re
        match = re.search(r"epoch_(\d+)", args.resume_t3_weights)
        if match:
            start_epoch = int(match.group(1)) + 1
            print(f"⏩ Setting internal loop iterator to start at Epoch {start_epoch}")
            
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
            
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if rank == 0 else dataloader
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            
            text_tokens = batch['text_tokens'].to(device)
            text_token_lens = batch['text_token_lens'].to(device)
            speech_tokens = batch['speech_tokens'].to(device)
            speech_token_lens = batch['speech_token_lens'].to(device)
            speaker_emb = batch['speaker_emb'].to(device)
            
            # Prepare T3Cond
            prompt_len = t3.module.hp.speech_cond_prompt_len if args.distributed else t3.hp.speech_cond_prompt_len
            cond_prompt_tokens = speech_tokens[:, :prompt_len] if speech_tokens.size(1) > prompt_len else None
            
            t3_cond = T3Cond(
                speaker_emb=speaker_emb,
                cond_prompt_speech_tokens=cond_prompt_tokens,
                emotion_adv=0.5 * torch.ones(text_tokens.size(0), 1, 1, device=device)
            )
            
            with autocast("cuda", enabled=use_amp, dtype=torch.float16):
                loss_text, loss_speech = t3.module.loss(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_token_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_token_lens
                ) if args.distributed else t3.loss(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_token_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_token_lens
                )
            
            loss = (loss_text + loss_speech) / args.accum_steps
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(dataloader):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if rank == 0:
                pbar.set_postfix({"loss": loss.item() * args.accum_steps, "loss_speech": loss_speech.item()})
                
                if not args.no_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "loss": loss.item() * args.accum_steps,
                        "loss_text": loss_text.item(),
                        "loss_speech": loss_speech.item(),
                        "lr": optimizer.param_groups[0]['lr']
                    })
            
            # Memory cleanup
            del loss, loss_text, loss_speech, t3_cond
            if i % 10 == 0:
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
            
        # Save checkpoint and generate sample
        if rank == 0 and epoch % args.save_every == 0:
            ckpt_path = f"t3_nepali_epoch_{epoch}.pt"
            model_to_save = t3.module if args.distributed else t3
            torch.save(model_to_save.state_dict(), ckpt_path)
            
            # --- PUSH TO HUB --- 
            if args.push_to_hub:
                try:
                    api = HfApi()
                    token = os.environ.get("HF_TOKEN")
                    print(f"🚀 Pushing checkpoint {ckpt_path} to {args.push_to_hub}...")
                    api.upload_file(
                        path_or_fileobj=ckpt_path,
                        path_in_repo=ckpt_path,
                        repo_id=args.push_to_hub,
                        token=token
                    )
                    print("✅ Checkpoint pushed successfully!")
                except Exception as e:
                    print(f"⚠️ Could not push checkpoint: {e}")
            
            # --- VALIDATION SAMPLE GENERATION ---
            print(f"\n📊 Generating validation sample for epoch {epoch}...")
            model_to_save.eval()
            with torch.no_grad():
                test_text = "नमस्ते, म नेपाली बोलिरहेको छु। यो तालिम कस्तो चलिरहेको छ?"
                
                # Device switch: move encoders back to GPU for inference
                model_wrapper.s3gen.tokenizer.to(device)
                model_wrapper.ve.to(device)
                
                try:
                    # We reuse the model_wrapper's high-level generate logic
                    old_t3 = model_wrapper.t3
                    model_wrapper.t3 = model_to_save
                    
                    val_wav = model_wrapper.generate(
                        test_text, 
                        language_id="ne", 
                        audio_prompt_path=args.eval_audio_path,
                        exaggeration=0.5,
                        temperature=0.8
                    )
                    
                    samples_dir = Path("samples")
                    samples_dir.mkdir(exist_ok=True)
                    output_sample_path = samples_dir / f"sample_epoch_{epoch}.wav"
                    
                    import torchaudio
                    torchaudio.save(str(output_sample_path), val_wav, model_wrapper.sr)
                    print(f"✅ Sample saved to {output_sample_path}")
                    
                    model_wrapper.t3 = old_t3
                except Exception as e:
                    print(f"⚠️ Could not generate sample: {e}")
                finally:
                    # Device switch back: move encoders to CPU for next epoch data loading
                    model_wrapper.s3gen.tokenizer.cpu()
                    model_wrapper.ve.cpu()
            
            model_to_save.train()
            # -----------------------------------
    
    if rank == 0:
        # Safetensors errors out if shared memory pointers exist. 
        # Because 'patched_model' is a duplicated reference specifically made for the generate validation loop, it crashes. We strip it!
        model_to_save = t3.module if args.distributed else t3
        final_sd = {k: v for k, v in model_to_save.state_dict().items() if not k.startswith("patched_model.")}
        save_file(final_sd, "t3_mtl_nepali_final.safetensors")
        print("Training finished. Saved to t3_mtl_nepali_final.safetensors")
        
        if args.push_to_hub:
            try:
                api = HfApi()
                token = os.environ.get("HF_TOKEN")
                print(f"🚀 Pushing final model to {args.push_to_hub}...")
                api.upload_file(
                    path_or_fileobj="t3_mtl_nepali_final.safetensors",
                    path_in_repo="t3_mtl_nepali_final.safetensors",
                    repo_id=args.push_to_hub,
                    token=token
                )
                print("✅ Final model pushed successfully!")
            except Exception as e:
                print(f"⚠️ Could not push final model: {e}")
    
    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    def get_default_device():
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest (jsonl, csv, or tsv)")
    parser.add_argument("--wav_dir", type=str, help="Optional explicit directory for audio files")
    parser.add_argument("--ckpt_dir", type=str, help="Path to base pretrained model dir")
    parser.add_argument("--resume_t3_weights", type=str, help="Path to a t3_nepali_epoch_X.pt file to resume from")
    parser.add_argument("--device", type=str, default=get_default_device(), help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--eval_audio_path", type=str, help="Path to reference audio for validation samples")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (M2 Max can handle 16-32)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps to simulate larger batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU threads for background data loading")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for fine-tuning")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (usually 1e-5 to 5e-5)")
    parser.add_argument("--save_every", type=int, default=5, help="Save interval in epochs")
    parser.add_argument("--skip_hi_init", action="store_true", help="Skip initializing Nepali weights from Hindi (helps with accent if you have enough data)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training (DDP)")
    parser.add_argument("--fp16", action="store_true", help="Enable float16 mixed precision (saves ~40%% GPU memory)")
    parser.add_argument("--push_to_hub", type=str, help="Hugging Face repo ID to push checkpoints to (e.g. officialuser/chatterbox-nepali)")
    
    args = parser.parse_args()
    
    if args.device == "mps":
        print("🚀 Using MPS (Metal Performance Shaders) for Mac acceleration")
    
    train(args)
