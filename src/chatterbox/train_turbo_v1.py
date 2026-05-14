
import sys
from unittest.mock import MagicMock

# --- ENVIRONMENT FIX: Mock torchvision if it's broken ---
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import librosa
import numpy as np
from safetensors.torch import save_file, load_file as load_safetensors
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, PeftModel
import gc
import io
import soundfile as sf
import torchaudio.functional as F_audio

from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import S3_SR
from datasets import load_dataset, Audio

class TurboDataset(Dataset):
    def __init__(self, data_path, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        
        print(f"Loading local dataset from {data_path}...")
        data_files = [str(p) for p in Path(data_path).rglob("*.parquet")]
        if not data_files:
            raise FileNotFoundError(f"No parquet files found in {data_path}")
        
        self.full_ds = load_dataset("parquet", data_files=data_files, split="train")
        self.full_ds = self.full_ds.cast_column("audio", Audio(sampling_rate=S3_SR, decode=False))
        self.full_ds = self.full_ds.shuffle(seed=42)
        print(f"✅ Loaded {len(self.full_ds)} items from parquet files.")

    def __len__(self):
        return len(self.full_ds)

    def __getitem__(self, idx):
        try:
            item = self.full_ds[idx]
            audio_data = item['audio']
            text = item.get('text') or ""
            lang = item.get('language') or item.get('lang') or "ne"

            # Prepend language tag
            lang_tag = f"[{lang.lower()}]"
            full_text = f"{lang_tag} {text}"
            
            # Tokenize using BPE tokenizer
            # T3 expects [START] and [STOP] tokens (IDs 255 and 0 in original, but here we use tokenizer's SOT/EOT)
            # Actually Turbo TTS script says:
            # text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            # It doesn't seem to add 255/0 explicitly in tts_turbo.py, but T3._ensure_BOT_EOT expects them.
            # Let's check T3Config for Turbo.
            
            encoded = self.tokenizer(full_text, return_tensors="pt")
            text_tokens = encoded.input_ids.squeeze(0)
            
            # We must NOT add 255 (START) and 0 (STOP) because the Turbo base model inference
            # does not use them! Adding them corrupts the positional alignment.

            # Decode parquet audio
            if isinstance(audio_data, dict) and 'bytes' in audio_data and audio_data['bytes'] is not None:
                with io.BytesIO(audio_data['bytes']) as f:
                    wav, orig_sr = sf.read(f)
                if orig_sr != S3_SR:
                    wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
                    wav_t = F_audio.resample(wav_t, orig_sr, S3_SR)
                    wav = wav_t.squeeze(0).numpy()
            else:
                wav = audio_data['array']
            
            wav = wav.astype(np.float32)
            if len(wav) > 15 * S3_SR: 
                wav = wav[:int(15 * S3_SR)]

            return {
                "text_tokens": text_tokens,
                "wav": wav
            }
        except Exception as e:
            print(f"⚠️ Error processing sample {idx}: {e}")
            return self.__getitem__(np.random.randint(0, len(self.full_ds)))

def collate_fn(batch):
    text_tokens = [item['text_tokens'] for item in batch]
    wavs = [item['wav'] for item in batch]
    
    text_token_lens = torch.tensor([len(t) for t in text_tokens])
    # Pad with tokenizer's pad token if available, otherwise use a safe value
    padding_value = 0 # stop_text_token is often used as padding too in T3
    text_tokens_padded = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=padding_value)
    
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
    else:
        device = torch.device(args.device)
        rank = 0

    try:
        import setproctitle
        setproctitle.setproctitle("CHatterbox_LORA_turbo_mai_nep")
    except ImportError:
        pass

    if rank == 0 and not args.no_wandb:
        wandb.init(project="chatterbox-turbo-nepali-lora", name="CHatterbox_LORA_turbo_mai_nep")
        wandb.config.update(args)

    print(f"🚀 Loading Turbo model on {device}...")
    model_wrapper = ChatterboxTurboTTS.from_pretrained(device)
    t3 = model_wrapper.t3
    tokenizer = model_wrapper.tokenizer
    
    # ─── ADD LANGUAGE TAGS & RESIZE ───
    new_tokens = ["[ne]", "[mai]"]
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"➕ Added {num_added} new tokens to tokenizer.")
    
    vocab_size = len(tokenizer)
    print(f"📏 Resizing T3 vocabulary to {vocab_size} tokens")
    t3.resize_text_embeddings(vocab_size)
    
    # ─── INJECT LORA ───
    print("💉 Injecting LoRA adapters into T3 Transformer...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn", "c_proj", "c_fc"], # GPT-2 specific
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=None,
    )
    t3.tfmr = get_peft_model(t3.tfmr, lora_config)
    t3.tfmr.print_trainable_parameters()
    
    # ─── RESUME FROM CHECKPOINT ───
    if args.init_lora_path:
        print(f"📂 Resuming LoRA from {args.init_lora_path}...")
        # Load LoRA weights into tfmr
        t3.tfmr.from_pretrained(t3.tfmr.base_model.model, args.init_lora_path)
        # Load text embeddings and head
        if os.path.exists(f"{args.init_lora_path}/text_emb.pt"):
            t3.text_emb.load_state_dict(torch.load(f"{args.init_lora_path}/text_emb.pt", map_location=device))
        if os.path.exists(f"{args.init_lora_path}/text_head.pt"):
            t3.text_head.load_state_dict(torch.load(f"{args.init_lora_path}/text_head.pt", map_location=device))
        print("✅ Weights loaded successfully.")

    # Ensure text embeddings and heads are trainable (essential for new tokens)
    t3.text_emb.requires_grad = True
    t3.text_head.requires_grad = True

    
    # Feature extractors
    s3_tokenizer = model_wrapper.s3gen.tokenizer.to(device)
    voice_encoder = model_wrapper.ve.to(device)
    
    if args.distributed:
        t3 = DDP(t3, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    t3.train()
    
    # Dataset
    dataset = TurboDataset(args.data_path, tokenizer, device="cpu")
    sampler = DistributedSampler(dataset) if args.distributed else None
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    trainable_params = [p for p in t3.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    if device.type == "cuda":
        t3_model = t3.module if args.distributed else t3
        # Patch for GPT2Model which expects .wte for gradient checkpointing
        target = t3_model.tfmr.base_model.model if hasattr(t3_model.tfmr, "base_model") else t3_model.tfmr
        if not hasattr(target, 'wte'):
            target.get_input_embeddings = lambda: t3_model.text_emb
            
        t3_model.tfmr.gradient_checkpointing_enable()
        print("🧠 Gradient checkpointing enabled (with Turbo patch).")
    
    use_amp = (args.fp16 or args.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = GradScaler("cuda", enabled=args.fp16) if args.fp16 else None
    
    # ─── TRAINING LOOP ───
    for epoch in range(args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
            
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if rank == 0 else dataloader
        for i, batch in enumerate(pbar):
            text_tokens = batch['text_tokens'].to(device)
            text_token_lens = batch['text_token_lens'].to(device)
            wavs = batch['wavs']
            
            with torch.no_grad():
                speech_tokens_list = []
                speaker_embs_list = []
                for w in wavs:
                    st, _ = s3_tokenizer.forward([w])
                    speech_tokens_list.append(st.squeeze(0).to(device))
                    ve = torch.from_numpy(voice_encoder.embeds_from_wavs([w], sample_rate=16000)).to(device)
                    speaker_embs_list.append(ve.mean(axis=0, keepdim=True))
                
                speech_token_lens = torch.tensor([len(s) for s in speech_tokens_list], device=device)
                speech_tokens = torch.nn.utils.rnn.pad_sequence(speech_tokens_list, batch_first=True, padding_value=6562)
                speaker_emb = torch.cat(speaker_embs_list, dim=0)

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
                loss = (args.text_loss_weight * loss_text + args.speech_loss_weight * loss_speech) / args.accum_steps
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (i + 1) % args.accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            if rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"L": f"{loss.item()*args.accum_steps:.3f}", "lr": f"{current_lr:.2e}"})
                if not args.no_wandb:
                    wandb.log({
                        "loss": loss.item() * args.accum_steps,
                        "loss_text": loss_text.item(),
                        "loss_speech": loss_speech.item(),
                        "lr": current_lr,
                        "epoch": epoch
                    })
            
            if i % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # ─── SAVE & VALIDATE ───
        if rank == 0 and (epoch + 1) % args.save_every == 0:
            ckpt_dir = f"turbo_lora_epoch_{epoch}"
            os.makedirs(ckpt_dir, exist_ok=True)
            
            t3_model = t3.module if args.distributed else t3
            t3_model.tfmr.save_pretrained(ckpt_dir)
            torch.save(t3_model.text_emb.state_dict(), f"{ckpt_dir}/text_emb.pt")
            torch.save(t3_model.text_head.state_dict(), f"{ckpt_dir}/text_head.pt")
            
            print(f"💾 Saved LoRA adapter to {ckpt_dir}")
            
            # --- HUB PUSH ---
            push_success = False
            if args.hub_repo:

                print(f"🚀 Pushing to Hub: {args.hub_repo}...")
                try:
                    from huggingface_hub import HfApi, create_repo
                    api = HfApi()
                    token = os.environ.get("HF_TOKEN")
                    
                    # Ensure repo exists
                    create_repo(repo_id=args.hub_repo, token=token, exist_ok=True, repo_type="model")
                    
                    # Create a basic model card if it doesn't exist
                    readme_path = os.path.join(ckpt_dir, "README.md")
                    if not os.path.exists(readme_path):
                        with open(readme_path, "w") as f:
                            f.write(f"---\nlibrary_name: peft\ntags:\n- text-to-speech\n- nepali\n- maithili\n- turbo\n---\n# Chatterbox Turbo LoRA - Epoch {epoch}\nFine-tuned for Nepali and Maithili.")

                    api.upload_folder(
                        folder_path=ckpt_dir,
                        repo_id=args.hub_repo,
                        commit_message=f"Epoch {epoch} save",
                        repo_type="model",
                        token=token
                    )
                    print(f"✅ Successfully pushed to Hub.")
                    push_success = True
                except Exception as e:
                    print(f"⚠️ Hub push failed: {e}")
                    push_success = False



            # --- LOCAL CLEANUP ---
            # If we pushed successfully, we can remove the previous local checkpoint to save space
            if push_success:
                # Find other turbo_lora_epoch folders and remove them
                import shutil
                for other_dir in Path(".").glob("turbo_lora_epoch_*"):
                    if other_dir.is_dir() and other_dir.name != ckpt_dir:
                        print(f"🧹 Cleaning up old local checkpoint: {other_dir}")
                        shutil.rmtree(other_dir)
            
            # Simple validation
            t3_model.eval()
            try:
                test_text = "नमस्ते, म नेपाली टर्बो मोडल हुँ।"
                val_wav = model_wrapper.generate(test_text, audio_prompt_path=args.eval_audio_path)
                sf.write(f"samples/turbo_val_epoch_{epoch}.wav", val_wav.squeeze(0).cpu().numpy(), model_wrapper.sr)
                print(f"✅ Validation sample saved.")
            except Exception as e:
                print(f"⚠️ Validation failed: {e}")
            t3_model.train()

    if rank == 0:
        print("🎉 Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/chatterbox-multilingual-data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--text_loss_weight", type=float, default=1.0)
    parser.add_argument("--speech_loss_weight", type=float, default=3.0)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--eval_audio_path", type=str, default="samples/Prakash.mp3")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hub_repo", type=str, default=None, help="Hugging Face repo ID to push to")
    parser.add_argument("--init_lora_path", type=str, default=None, help="Path to existing LoRA checkpoint to resume from")
    args = parser.parse_args()

    
    os.makedirs("samples", exist_ok=True)
    train(args)
