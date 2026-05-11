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
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LambdaLR
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


class MultilingualLocalDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        
        print(f"Loading local dataset from {data_path}...")
        # Load all parquet files in the data directory
        data_files = [str(p) for p in Path(data_path).glob("*.parquet")]
        self.ds = load_dataset("parquet", data_files=data_files, split="train")
        
        # Global shuffle — this is only possible with local data!
        # This fixes the sequential language problem.
        print("Performing global shuffle...")
        self.ds = self.ds.shuffle(seed=42)
        
        self.ds = self.ds.cast_column("audio", Audio(sampling_rate=S3_SR, decode=False))
        print(f"Dataset loaded: {len(self.ds)} samples")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        try:
            item = self.ds[idx]
            audio_data = item['audio']
            text = item.get('text') or ""
            lang = item.get('language', 'ne')

            # Process text — must match inference pattern
            text_tokens = self.tokenizer.text_to_tokens(text, language_id=lang).squeeze(0)
            text_tokens = F.pad(text_tokens, (1, 0), value=255)  # [START]
            text_tokens = F.pad(text_tokens, (0, 1), value=0)    # [STOP]

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
            
            if len(wav) > 20 * S3_SR:
                wav = wav[:20 * S3_SR]

            return {
                "text_tokens": text_tokens,
                "wav": wav,
                "lang": lang
            }
        except Exception as e:
            print(f"⚠️ Skipping sample {idx}: {e}")
            # Return a random sample instead of None to keep batch size consistent
            return self.__getitem__(np.random.randint(0, len(self.ds)))

def collate_fn(batch):
    text_tokens = [item['text_tokens'] for item in batch]
    wavs = [item['wav'] for item in batch]
    
    text_token_lens = torch.tensor([len(t) for t in text_tokens])
    text_tokens_padded = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=705)
    
    return {
        "text_tokens": text_tokens_padded,
        "text_token_lens": text_token_lens,
        "wavs": wavs
    }


def freeze_bottom_layers(t3_model, num_freeze):
    """
    Freeze the bottom `num_freeze` transformer layers.
    
    Bottom layers learn universal acoustic patterns (clear speech, natural pauses,
    smooth transitions). Freezing them prevents catastrophic forgetting — the model
    keeps its ability to produce clean audio while the top layers learn new languages.
    
    Trainable after freezing:
      - Top (total - num_freeze) transformer layers
      - text_emb (needed for new language tokens)
      - speech_emb, speech_head (acoustic output)
      - text_head, text_pos_emb, speech_pos_emb
      - cond_enc (conditioning encoder)
      - tfmr.norm (final layer norm)
    
    Frozen:
      - Bottom num_freeze transformer layers (preserves acoustic foundation)
    """
    total_layers = len(t3_model.tfmr.layers)
    if num_freeze > total_layers:
        raise ValueError(f"Cannot freeze {num_freeze} layers, model only has {total_layers}")
    
    frozen_params = 0
    trainable_params = 0
    
    # Freeze the bottom N layers
    for i in range(num_freeze):
        for param in t3_model.tfmr.layers[i].parameters():
            param.requires_grad = False
            frozen_params += param.numel()
    
    # Count trainable params
    for param in t3_model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    
    total_params = frozen_params + trainable_params
    print(f"\n🧊 Layer Freezing Summary:")
    print(f"   Total transformer layers: {total_layers}")
    print(f"   Frozen (bottom):          {num_freeze} layers")
    print(f"   Trainable (top):          {total_layers - num_freeze} layers")
    print(f"   Frozen parameters:        {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"   Trainable parameters:     {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    return num_freeze


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
        wandb.init(project="chatterbox-multilingual-4090-v3")
        wandb.config.update(args)

    model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)
    t3 = model_wrapper.t3
    tokenizer = model_wrapper.tokenizer
    t3.resize_text_embeddings(len(tokenizer.tokenizer.get_vocab()))
    
    t3.to(device)
    
    # ─── Layer Freezing ───
    if args.freeze_layers > 0:
        freeze_bottom_layers(t3, args.freeze_layers)
    else:
        total_params = sum(p.numel() for p in t3.parameters())
        trainable_params = sum(p.numel() for p in t3.parameters() if p.requires_grad)
        print(f"\n⚡ Full fine-tuning (no layers frozen)")
        print(f"   Total parameters:     {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
    
    # ─── Resume from checkpoint ───
    if args.resume_from:
        print(f"\n📂 Resuming from checkpoint: {args.resume_from}")
        state = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        clean_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in state.items()}
        
        # Handle vocab size mismatch
        if "text_emb.weight" in clean_state:
            state_vocab_size = clean_state["text_emb.weight"].shape[0]
            model_vocab_size = t3.hp.text_tokens_dict_size
            if state_vocab_size != model_vocab_size:
                print(f"   Resizing vocabulary from {model_vocab_size} to {state_vocab_size}")
                t3.resize_text_embeddings(state_vocab_size)
        
        t3.load_state_dict(clean_state, strict=False)
        del state, clean_state
        torch.cuda.empty_cache()
        print(f"   ✅ Checkpoint loaded successfully")
    
    # Keep auxiliary models in FP32 for signal processing stability
    s3_tokenizer = model_wrapper.s3gen.tokenizer.to(device)
    voice_encoder = model_wrapper.ve.to(device)
    
    if args.distributed:
        t3 = DDP(t3, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    t3.train()
    
    dataset = MultilingualLocalDataset(
        args.data_path, 
        tokenizer, 
        device="cpu"
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    # ─── Optimizer: only optimize trainable parameters ───
    trainable_params = [p for p in t3.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    if device.type == "cuda":
        (t3.module if args.distributed else t3).tfmr.gradient_checkpointing_enable()
    
    # ─── Learning Rate Scheduler ───
    estimated_steps_per_epoch = args.estimated_steps_per_epoch
    total_training_steps = estimated_steps_per_epoch * args.epochs
    warmup_steps = args.warmup_steps
    
    if args.lr_schedule == "constant":
        # Warmup → Constant LR (more stable for fine-tuning)
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.01,
            end_factor=1.0, 
            total_iters=warmup_steps
        )
        constant_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, constant_scheduler], 
            milestones=[warmup_steps]
        )
    elif args.lr_schedule == "cosine":
        # Warmup → Cosine Decay
        cosine_steps = max(total_training_steps - warmup_steps, 1)
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.01,
            end_factor=1.0, 
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=cosine_steps, 
            eta_min=args.min_lr
        )
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_steps]
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {args.lr_schedule}")
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"🚀 Training Config (V3):")
        print(f"   Epochs:            {args.epochs}")
        print(f"   Batch size:        {args.batch_size}")
        print(f"   Accum steps:       {args.accum_steps}")
        print(f"   Effective batch:   {args.batch_size * args.accum_steps}")
        print(f"   Peak LR:          {args.lr}")
        print(f"   LR schedule:      {args.lr_schedule}")
        print(f"   Min LR:           {args.min_lr}")
        print(f"   Weight decay:     {args.weight_decay}")
        print(f"   Warmup steps:     {warmup_steps}")
        print(f"   Est. total steps: {total_training_steps}")
        print(f"   Grad clip norm:   {args.max_grad_norm}")
        print(f"   Precision:        {'bf16' if args.bf16 else 'fp16' if args.fp16 else 'fp32'}")
        print(f"   Frozen layers:    {args.freeze_layers}")
        print(f"   Speech loss wt:   {args.speech_loss_weight}")
        print(f"   Text loss wt:     {args.text_loss_weight}")
        print(f"   Save every:       {args.save_every} epochs")
        print(f"   Push to hub:      {args.push_to_hub or 'disabled'}")
        if args.resume_from:
            print(f"   Resumed from:     {args.resume_from}")
        print(f"{'='*60}")
    
    # ─── AMP Setup ───
    use_amp = (args.fp16 or args.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = GradScaler("cuda", enabled=args.fp16) if args.fp16 else None
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if rank == 0 else dataloader
        epoch_loss_sum = 0.0
        epoch_loss_text_sum = 0.0
        epoch_loss_speech_sum = 0.0
        epoch_steps = 0
        
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
            
            # ─── Weighted loss: speech loss is weighted higher ───
            # This forces the model to focus on acoustic quality rather than
            # text prediction (which converges much faster and leads to overfitting)
            weighted_loss = (args.text_loss_weight * loss_text + args.speech_loss_weight * loss_speech) / args.accum_steps
            
            if scaler:
                scaler.scale(weighted_loss).backward()
                if (i+1) % args.accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            else:
                weighted_loss.backward()
                if (i+1) % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            
            # Track raw (unweighted) losses for monitoring
            raw_loss = loss_text.item() + loss_speech.item()
            raw_loss_text = loss_text.item()
            raw_loss_speech = loss_speech.item()
            epoch_loss_sum += raw_loss
            epoch_loss_text_sum += raw_loss_text
            epoch_loss_speech_sum += raw_loss_speech
            epoch_steps += 1
            
            if rank == 0: 
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{raw_loss:.4f}",
                    "L_txt": f"{raw_loss_text:.4f}",
                    "L_spch": f"{raw_loss_speech:.4f}",
                    "lr": f"{current_lr:.2e}"
                })
                if not args.no_wandb:
                    wandb.log({
                        "loss": raw_loss,
                        "loss_text": raw_loss_text,
                        "loss_speech": raw_loss_speech,
                        "weighted_loss": weighted_loss.item() * args.accum_steps,
                        "global_step": global_step,
                        "epoch": epoch,
                        "learning_rate": current_lr
                    })
            
            # Periodic cache clearing
            if (i+1) % 10 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        # ─── End of Epoch ───
        if rank == 0 and epoch_steps > 0:
            avg_loss = epoch_loss_sum / epoch_steps
            avg_loss_text = epoch_loss_text_sum / epoch_steps
            avg_loss_speech = epoch_loss_speech_sum / epoch_steps
            
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Avg Loss:        {avg_loss:.4f}")
            print(f"   Avg Loss Text:   {avg_loss_text:.4f}")
            print(f"   Avg Loss Speech: {avg_loss_speech:.4f}")
            print(f"   Learning Rate:   {scheduler.get_last_lr()[0]:.2e}")
            print(f"   Global Steps:    {global_step}")
            
            if not args.no_wandb:
                wandb.log({
                    "epoch_avg_loss": avg_loss,
                    "epoch_avg_loss_text": avg_loss_text,
                    "epoch_avg_loss_speech": avg_loss_speech,
                    "epoch": epoch
                })
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"   ✨ New best loss: {best_loss:.4f}")
        
        # ─── Checkpoint Saving ───
        is_save_epoch = (epoch % args.save_every == 0) and (epoch > 0)
        is_final_epoch = (epoch == args.epochs - 1)
        
        if rank == 0 and (is_save_epoch or is_final_epoch):
            t3_model = t3.module if args.distributed else t3
            ckpt_path = f"t3_v3_epoch_{epoch}.pt"
            torch.save(t3_model.state_dict(), ckpt_path)
            print(f"   💾 Saved checkpoint: {ckpt_path}")
            
            if args.push_to_hub:
                try:
                    HfApi().upload_file(
                        path_or_fileobj=ckpt_path, 
                        path_in_repo=ckpt_path, 
                        repo_id=args.push_to_hub, 
                        token=os.environ.get("HF_TOKEN")
                    )
                    print(f"   ☁️  Pushed to {args.push_to_hub}/{ckpt_path}")
                except Exception as e:
                    print(f"   ⚠️ Push failed: {e}")
            
            # Clean up old local checkpoint
            if is_save_epoch and not is_final_epoch and epoch > args.save_every:
                old_ckpt = f"t3_v3_epoch_{epoch - args.save_every}.pt"
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                    print(f"   🗑️  Cleaned up old local checkpoint: {old_ckpt}")

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"🏁 Training V3 complete!")
        print(f"   Total epochs:    {args.epochs}")
        print(f"   Total steps:     {global_step}")
        print(f"   Best avg loss:   {best_loss:.4f}")
        print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/chatterbox-multilingual-data", help="Path to local dataset directory")
    parser.add_argument("--push_to_hub", type=str, default="Firoj112/chatterbox-multilingual-t3-v2-4090", help="HF Model Repo ID to push checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6, help="Peak learning rate (V3 default: 5e-6)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for cosine decay")
    parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine"],
                        help="LR schedule after warmup: 'constant' (stable) or 'cosine' (decaying)")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Number of linear warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--freeze_layers", type=int, default=15,
                        help="Number of bottom transformer layers to freeze (model has 30). "
                             "Freezing preserves universal acoustic patterns while allowing top layers to learn new languages.")
    parser.add_argument("--speech_loss_weight", type=float, default=3.0,
                        help="Weight for speech loss. Higher = model focuses more on acoustic quality.")
    parser.add_argument("--text_loss_weight", type=float, default=1.0,
                        help="Weight for text loss.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a checkpoint to resume from (e.g. t3_multilingual_epoch_6.pt)")
    parser.add_argument("--estimated_steps_per_epoch", type=int, default=6000)
    parser.add_argument("--save_every", type=int, default=2, help="Save checkpoint every N epochs")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (recommended for RTX 30/40 series)")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)
