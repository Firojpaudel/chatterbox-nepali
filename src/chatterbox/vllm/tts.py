from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any
import time
import json
import shutil
import os

# Monkey-patch transformers to prevent vLLM from crashing on duplicate 'aimv2' registration
import transformers
from transformers.models.auto.configuration_auto import AutoConfig
_original_register = AutoConfig.register
def _new_register(model_type, config, exist_ok=False):
    return _original_register(model_type, config, exist_ok=True)
AutoConfig.register = _new_register

def _patch_vllm_on_disk():
    """Surgery: Permanently fix the vLLM 'aimv2' bug by patching the source code on disk."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            return
        vllm_dir = os.path.dirname(spec.origin)
        ovis_path = os.path.join(vllm_dir, "transformers_utils", "configs", "ovis.py")
        
        if os.path.exists(ovis_path):
            with open(ovis_path, "r") as f:
                content = f.read()
            
            # The line causing the subprocess to crash
            old_line = 'AutoConfig.register("aimv2", AIMv2Config)'
            new_line = 'AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)'
            
            if old_line in content and new_line not in content:
                print(f"🩹 Applying permanent patch to vLLM at: {ovis_path}")
                with open(ovis_path, "w") as f:
                    f.write(content.replace(old_line, new_line))
    except Exception as e:
        print(f"Warning: Failed to dynamically patch vLLM: {e}")

_patch_vllm_on_disk()

from vllm import LLM, SamplingParams
from functools import lru_cache

# Speech tokens are offset by 2560 to avoid overlap with text tokens (0-2454)
MULTILINGUAL_SPEECH_TOKEN_OFFSET = 2560


import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file

from chatterbox.vllm.models.t3.modules.t3_config import T3Config

from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.voice_encoder import VoiceEncoder
from .models.t3 import SPEECH_TOKEN_OFFSET
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from .text_utils import punc_norm

REPO_ID = "ResembleAI/chatterbox"
NEPALI_REPO_ID = "Firoj112/chatterbox-nepali-runs"

@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    @classmethod
    def load(cls, fpath):
        kwargs = torch.load(fpath, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


# ── Multilingual vLLM config.json for t3-model directory ──────────────────
MULTILINGUAL_VLLM_CONFIG = {
    "architectures": ["ChatterboxT3"],
    "is_multilingual": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_implementation": "sdpa",
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 8192,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 30,
    "num_key_value_heads": 32,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "use_cache": True,    
    "vocab_size": 32000
}


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, target_device: str, max_model_len: int,
                 t3: LLM, t3_config: T3Config, t3_cond_enc: T3CondEnc, 
                 t3_speech_emb: torch.nn.Embedding, t3_speech_pos_emb: LearnedPositionEmbeddings,
                 s3gen: S3Gen, ve: VoiceEncoder, default_conds: Conditionals):
        self.target_device = target_device
        self.max_model_len = max_model_len
        self.t3 = t3
        self.t3_config = t3_config
        self.t3_cond_enc = t3_cond_enc
        self.t3_speech_emb = t3_speech_emb
        self.t3_speech_pos_emb = t3_speech_pos_emb

        self.s3gen = s3gen
        self.ve = ve
        self.default_conds = default_conds

    @property
    def sr(self) -> int:
        """Sample rate of synthesized audio"""
        return S3GEN_SR

    @classmethod
    def _load_cond_components(cls, t3_weights: dict, t3_config: T3Config, target_device: str):
        """Load conditioning encoder, speech embedding, and speech position embedding from weights."""
        
        def expand_weights(weights_dict, target_dim=2048):
            expanded = {}
            for k, v in weights_dict.items():
                # 2D weights: [N, 1024] or [1024, N]
                if v.dim() == 2:
                    new_v = torch.zeros([target_dim if v.shape[0] == 1024 else v.shape[0], target_dim if v.shape[1] == 1024 else v.shape[1]], dtype=v.dtype, device=v.device)
                    if v.shape[1] == 1024 and v.shape[0] == 1024:
                        new_v[:1024, :1024] = v
                        new_v[1024:2048, 1024:2048] = v
                    elif v.shape[1] == 1024:
                        new_v[:, :1024] = v
                        new_v[:, 1024:2048] = v
                    elif v.shape[0] == 1024:
                        new_v[:1024, :] = v
                        new_v[1024:2048, :] = v
                    v = new_v
                # 3D weights: [1, 32, 1024]
                elif v.dim() == 3 and v.shape[2] == 1024:
                    new_v = torch.zeros([v.shape[0], v.shape[1], target_dim], dtype=v.dtype, device=v.device)
                    new_v[:, :, :1024] = v
                    new_v[:, :, 1024:2048] = v
                    v = new_v
                # 1D weights: [1024] (biases/norms)
                elif v.dim() == 1 and v.shape[0] == 1024:
                    new_v = torch.zeros([target_dim], dtype=v.dtype, device=v.device)
                    for i in range(0, target_dim, 1024):
                        new_v[i:i+1024] = v
                    v = new_v
                expanded[k] = v
            return expanded

        t3_enc = T3CondEnc(t3_config)
        enc_weights = {k.replace('cond_enc.', ''): v for k, v in t3_weights.items() if k.startswith('cond_enc.')}
        print(f"DEBUG: _load_cond_components - n_channels: {t3_config.n_channels}, target_device: {target_device}")
        if t3_config.n_channels > 1024:
            enc_weights = expand_weights(enc_weights, target_dim=t3_config.n_channels)
            # Print a few shapes to verify expansion
            for k in list(enc_weights.keys())[:3]:
                print(f"DEBUG: Expanded {k} shape: {list(enc_weights[k].shape)}")
        t3_enc.load_state_dict(enc_weights)
        t3_enc = t3_enc.to(device=target_device).eval()

        t3_speech_emb = torch.nn.Embedding(t3_config.speech_tokens_dict_size, t3_config.n_channels)
        emb_weights = {k.replace('speech_emb.', ''): v for k, v in t3_weights.items() if k.startswith('speech_emb.')}
        if t3_config.n_channels > 1024:
            emb_weights = expand_weights(emb_weights, target_dim=t3_config.n_channels)
        t3_speech_emb.load_state_dict(emb_weights)
        t3_speech_emb = t3_speech_emb.to(device=target_device).eval()

        t3_speech_pos_emb = LearnedPositionEmbeddings(t3_config.max_speech_tokens + 2 + 2, t3_config.n_channels)
        pos_weights = {k.replace('speech_pos_emb.', ''): v for k, v in t3_weights.items() if k.startswith('speech_pos_emb.')}
        if t3_config.n_channels > 1024:
            pos_weights = expand_weights(pos_weights, target_dim=t3_config.n_channels)
        t3_speech_pos_emb.load_state_dict(pos_weights)
        t3_speech_pos_emb = t3_speech_pos_emb.to(device=target_device).eval()

        return t3_enc, t3_speech_emb, t3_speech_pos_emb

    @classmethod
    def from_local(cls, ckpt_dir: str | Path, target_device: str = "cuda", 
                   max_model_len: int = 1000, compile: bool = False,
                   max_batch_size: int = 10,
                   s3gen_use_fp16: bool = False,
                   **kwargs) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)
        t3_config = T3Config()

        t3_weights = load_file(ckpt_dir / "t3_cfg.safetensors")
        t3_enc, t3_speech_emb, t3_speech_pos_emb = cls._load_cond_components(t3_weights, t3_config, target_device)

        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()
        vllm_memory_needed = (1.55*1024*1024*1024) + (max_batch_size * max_model_len * 1024 * 128)
        vllm_memory_percent = vllm_memory_needed / unused_gpu_memory
        print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

        base_vllm_kwargs = {
            "model": "./t3-model",
            "task": "generate",
            "tokenizer": "EnTokenizer",
            "tokenizer_mode": "custom",
            "gpu_memory_utilization": vllm_memory_percent,
            "enforce_eager": not compile,
            "max_model_len": max_model_len,
        }
        t3 = LLM(**{**base_vllm_kwargs, **kwargs})

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve = ve.to(device=target_device).eval()

        s3gen = S3Gen(use_fp16=s3gen_use_fp16)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen = s3gen.to(device=target_device).eval()

        default_conds = Conditionals.load(ckpt_dir / "conds.pt")
        default_conds.to(device=target_device)

        return cls(
            target_device=target_device, max_model_len=max_model_len,
            t3=t3, t3_config=t3_config, t3_cond_enc=t3_enc, t3_speech_emb=t3_speech_emb, t3_speech_pos_emb=t3_speech_pos_emb,
            s3gen=s3gen, ve=ve, default_conds=default_conds,
        )

    @classmethod
    def from_pretrained(cls,
                        repo_id: str = REPO_ID,
                        revision: str = "1b475dffa71fb191cb6d5901215eb6f55635a9b6",
                        *args, **kwargs) -> 'ChatterboxTTS':
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        t3_cfg_path = Path(local_path).parent / "t3_cfg.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path).parent, *args, **kwargs)

    @classmethod
    def from_nepali(cls, model_filename: str,
                    target_device: str = "cuda",
                    max_model_len: int = 1000,
                    compile: bool = False,
                    max_batch_size: int = 10,
                    s3gen_use_fp16: bool = False,
                    **kwargs) -> 'ChatterboxTTS':
        """
        Load a Nepali fine-tuned ChatterboxTTS model for vLLM inference.

        Downloads support files (ve, s3gen, conds, tokenizer) from the
        Firoj112/chatterbox-nepali-runs repo and uses the given fine-tuned
        T3 weights file for the vLLM engine.

        Args:
            model_filename: Absolute path to the Nepali fine-tuned .safetensors
                            weights file (already downloaded by gradio_app).
        """
        model_filename = Path(model_filename).resolve()
        print(f"[from_nepali] Using T3 weights: {model_filename}")

        # ── 1. Download support files from HF ─────────────────────────────
        ckpt_dir = Path(snapshot_download(
            repo_id=NEPALI_REPO_ID,
            repo_type="model",
            revision="main",
            allow_patterns=[
                "ve.safetensors",
                "s3gen.safetensors",
                "conds.pt",
                "grapheme_mtl_merged_expanded_v1.json",
                "Cangjie5_TC.json",
            ],
            token=os.getenv("HF_TOKEN"),
        ))
        print(f"[from_nepali] Support files at: {ckpt_dir}")

        # ── 2. Load conditioning components from the Nepali weights ───────
        t3_config = T3Config.multilingual()
        t3_weights = load_file(str(model_filename))
        t3_enc, t3_speech_emb, t3_speech_pos_emb = cls._load_cond_components(
            t3_weights, t3_config, target_device
        )
        del t3_weights  # free RAM
        torch.cuda.empty_cache()

        # ── 3. Set up t3-model/ directory for vLLM engine ─────────────────
        # Ensure fresh model directory for vLLM
        model_dir = Path("./t3-model")
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
        model_dir.mkdir(exist_ok=True)

        # Symlink model weights
        model_safetensors = model_dir / "model.safetensors"
        model_safetensors.symlink_to(model_filename)

        # Write correct multilingual config.json
        print(f"DEBUG: Writing vLLM config.json to {model_dir / 'config.json'}")
        print(f"DEBUG: Config content: {json.dumps(MULTILINGUAL_VLLM_CONFIG, indent=2)}")
        with open(model_dir / "config.json", "w") as f:
            json.dump(MULTILINGUAL_VLLM_CONFIG, f, indent=2)

        # Copy tokenizer JSON so vLLM can find it next to model weights
        tok_src = ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"
        if tok_src.exists():
            shutil.copy(tok_src, model_dir / "grapheme_mtl_merged_expanded_v1.json")

        cangjie_src = ckpt_dir / "Cangjie5_TC.json"
        if cangjie_src.exists():
            shutil.copy(cangjie_src, model_dir / "Cangjie5_TC.json")

        # ── 4. Allocate GPU memory (multilingual model is bigger) ─────────
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()
        
        # Block-diagonal expansion quadrupled the LLaMA weight parameters!
        # The model weights now take 3.83 GiB alone.
        # We need to allocate at least 5 GiB + KV cache.
        vllm_memory_needed = (5.5 * 1024**3) + (max_batch_size * max_model_len * 2048 * 128)
        
        vllm_memory_percent = vllm_memory_needed / unused_gpu_memory
        # Cap at 90% just to be safe
        vllm_memory_percent = min(vllm_memory_percent, 0.90)
        
        print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

        # ── 5. Start vLLM engine ──────────────────────────────────────────
        base_vllm_kwargs = {
            "model": "./t3-model",
            "task": "generate",
            "tokenizer": "MtlTokenizer",
            "tokenizer_mode": "custom",
            "gpu_memory_utilization": vllm_memory_percent,
            "enforce_eager": not compile,
            "max_model_len": max_model_len,
            "swap_space": 0, # Disable the 4 GiB CPU Swap Space allocation to save System RAM
        }
        t3 = LLM(**{**base_vllm_kwargs, **kwargs})

        # ── 6. Load voice encoder & S3Gen from HF cache ──────────────────
        ve = VoiceEncoder()
        ve.load_state_dict(load_file(str(ckpt_dir / "ve.safetensors")))
        ve = ve.to(device=target_device).eval()

        s3gen = S3Gen(use_fp16=s3gen_use_fp16)
        s3gen.load_state_dict(load_file(str(ckpt_dir / "s3gen.safetensors")), strict=False)
        s3gen = s3gen.to(device=target_device).eval()

        default_conds = Conditionals.load(ckpt_dir / "conds.pt")
        default_conds.to(device=target_device)

        print("✅ Nepali vLLM model fully loaded!")
        return cls(
            target_device=target_device, max_model_len=max_model_len,
            t3=t3, t3_config=t3_config, t3_cond_enc=t3_enc,
            t3_speech_emb=t3_speech_emb, t3_speech_pos_emb=t3_speech_pos_emb,
            s3gen=s3gen, ve=ve, default_conds=default_conds,
        )

    @lru_cache(maxsize=10)
    def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Tuple[dict[str, Any], torch.Tensor]:
        if wav_fpath is None:
            s3gen_ref_dict = self.default_conds.gen
            t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
            ve_embed = self.default_conds.t3.speaker_emb
        else:
            ## Load reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

            # Speech cond prompt tokens
            if self.t3_config.speech_cond_prompt_len:
                s3_tokzr = self.s3gen.tokenizer
                t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=self.t3_config.speech_cond_prompt_len)
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)
            else:
                t3_cond_prompt_tokens = None

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        cond_prompt_speech_emb = None
        if t3_cond_prompt_tokens is not None:
            cond_prompt_speech_emb = self.t3_speech_emb(t3_cond_prompt_tokens)[0] + self.t3_speech_pos_emb(t3_cond_prompt_tokens)

        cond_emb = self.t3_cond_enc(T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            cond_prompt_speech_emb=cond_prompt_speech_emb,
            emotion_adv=0.5 * torch.ones(1, 1)
        ).to(device=self.target_device)).to(device="cpu")  # Conditionals need to be given to VLLM in CPU

        return s3gen_ref_dict, cond_emb

    def update_exaggeration(self, cond_emb: torch.Tensor, exaggeration: float) -> torch.Tensor:
        if exaggeration == 0.5:
            return cond_emb

        new_cond_emb = cond_emb.clone()
        new_cond_emb[-1] = self.t3_cond_enc.emotion_adv_fc(
            (exaggeration * torch.ones(1, 1)).to(self.target_device)
        ).to('cpu')
        return new_cond_emb

    def generate(
        self,
        prompts: Union[str, list[str]],
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[Union[str, list[str]]] = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        max_tokens=1000,
        top_p=0.8,
        repetition_penalty=2.0,
        *args, **kwargs,
    ) -> list[any]:
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)

        return self.generate_with_conds(
            prompts=prompts,
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb,
            language_id=language_id,
            temperature=temperature,
            exaggeration=exaggeration,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            *args, **kwargs
        )

    def generate_with_conds(
        self,
        prompts: Union[str, list[str]],
        s3gen_ref: dict[str, Any],
        cond_emb: torch.Tensor,
        language_id: Optional[Union[str, list[str]]] = None,
        temperature: float = 0.8,
        exaggeration: float = 0.5,
        max_tokens=1000,
        diffusion_steps: int = 10,
        top_p=0.8,
        repetition_penalty=2.0,
        *args, **kwargs,
    ) -> list[any]:
        if isinstance(prompts, str):
            prompts = [prompts]

        # Handle language_id
        if language_id is None:
            language_ids = [None] * len(prompts)
        elif isinstance(language_id, str):
            language_ids = [language_id] * len(prompts)
        else:
            language_ids = language_id

        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        # Norm and tokenize text with optional language tags
        prepared_prompts = []
        for p, lang_id in zip(prompts, language_ids):
            normalized = punc_norm(p)
            if lang_id:
                text = f"[START][{lang_id.lower()}]{normalized}[STOP]"
            else:
                text = f"[START]{normalized}[STOP]"
            prepared_prompts.append(text)

        with torch.inference_mode():
            start_time = time.time()
            batch_results = self.t3.generate(
                [
                    {
                        "prompt": text,
                        "multi_modal_data": {
                            "conditionals": [cond_emb],
                        },
                    }
                    for text in prepared_prompts
                ],
                sampling_params=SamplingParams(
                    temperature=temperature,
                    stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
                    max_tokens=min(max_tokens, self.max_model_len),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    *args, **kwargs,
                )
            )
            t3_gen_time = time.time() - start_time
            print(f"[T3] Speech Token Generation time: {t3_gen_time:.2f}s")

            # run torch gc
            torch.cuda.empty_cache()

            start_time = time.time()
            results = []
            for i, batch_result in enumerate(batch_results):
                for output in batch_result.outputs:
                    if i % 5 == 0:
                        print(f"[S3] Processing prompt {i} of {len(batch_results)}")

                    # Run gc every 10 prompts
                    if i % 10 == 0:
                        torch.cuda.empty_cache()

                    # Offset the tokens back to 0-8191 for the S3 synthesizer
                    # IMPORTANT: We MUST filter out the STOP token (8192+) which vLLM might include.
                    speech_tokens = [t - MULTILINGUAL_SPEECH_TOKEN_OFFSET for t in output.token_ids 
                                     if MULTILINGUAL_SPEECH_TOKEN_OFFSET <= t < MULTILINGUAL_SPEECH_TOKEN_OFFSET + 8192]
                    speech_tokens = torch.tensor(speech_tokens, device="cuda")
                    
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < self.t3_config.stop_speech_token]

                    wav, _ = self.s3gen.inference(
                        speech_tokens=speech_tokens,
                        ref_dict=s3gen_ref,
                        n_timesteps=diffusion_steps,
                    )
                    results.append(wav.cpu())
            s3gen_gen_time = time.time() - start_time
            print(f"[S3Gen] Wavform Generation time: {s3gen_gen_time:.2f}s")

            return results
        
    def shutdown(self):
        del self.t3
        torch.cuda.empty_cache()
