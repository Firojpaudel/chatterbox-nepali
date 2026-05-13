"""
ULTIMATE V2V Streamer (Robust Edition).
Optimized for RTX 4090 with Parallel Vocoding.
"""
import time
import torch
import numpy as np
import logging
import queue
import threading
import copy
import gc
from typing import Generator
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.s3tokenizer import drop_invalid_tokens
from chatterbox.models.s3gen.const import S3GEN_SR
from chatterbox.utils.sanitizer import sanitize_text, smart_chunk
from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from transformers.generation.logits_process import (
    TopPLogitsWarper,
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
)

logger = logging.getLogger("Streamer")

class StreamingChatterbox:
    def __init__(self, model: ChatterboxMultilingualTTS, device: str = "cuda"):
        self.model = model
        self.device = device
        self.sr = S3GEN_SR

        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True

        if hasattr(self.model.t3.tfmr, "merge_and_unload"):
            try:
                self.model.t3.tfmr = self.model.t3.tfmr.merge_and_unload()
                logger.info("🚀 LoRA Merged.")
            except Exception as e:
                logger.warning(f"⚠️ LoRA Merge Error: {e}")

        self.model.t3.to(self.device).eval()
        self.model.s3gen.to(self.device).eval()
        self.samples_per_token = 960 # 24000Hz / 25Hz
        self._warmup()

    def _warmup(self):
        logger.info("🔥 Priming CUDA kernels...")
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            dummy = torch.zeros((10,), dtype=torch.long, device=self.device)
            self.model.s3gen.inference(dummy, ref_dict=self.model.conds.gen)
        logger.info("✅ Warmup Complete.")

    @torch.inference_mode()
    def stream_inference(
        self,
        text: str,
        language_id: str,
        exaggeration: float = 0.0,
        cfg_weight: float = 0.8,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        top_p: float = 1.0,
        min_p: float = 0.05,
        max_new_tokens: int = 1000,
        chunk_size_tokens: int = 35,
        context_window: int = 40,
    ) -> Generator[dict, None, None]:
        start_time = time.time()
        full_text = sanitize_text(text, lang=language_id)
        text_chunks = smart_chunk(full_text)
        
        logger.info(f"⚡ ULTIMATE V2V: chunks={len(text_chunks)}, cfg={cfg_weight}")

        # Parallel Queues (Long-lived for the whole request)
        token_queue = queue.Queue()
        audio_queue = queue.Queue()
        stop_signal = threading.Event()

        def vocoder_worker():
            all_tokens = torch.tensor([], dtype=torch.long, device=self.device)
            last_chunk_time_worker = start_time
            chunk_idx = 0
            while not stop_signal.is_set() or not token_queue.empty():
                try:
                    chunk = token_queue.get(timeout=0.1)
                    if chunk is None: break
                    
                    chunk_clean = drop_invalid_tokens(chunk).to(self.device)
                    if chunk_clean.numel() == 0:
                        token_queue.task_done()
                        continue

                    ctx_len = min(context_window, all_tokens.size(-1))
                    tokens_to_proc = torch.cat([all_tokens[-ctx_len:], chunk_clean], dim=-1) if ctx_len > 0 else chunk_clean
                    
                    if tokens_to_proc.numel() > 0:
                        with torch.inference_mode(), torch.amp.autocast('cuda'):
                            wav, _ = self.model.s3gen.inference(
                                tokens_to_proc, 
                                ref_dict=self.model.conds.gen,
                                n_cfm_timesteps=6
                            )
                        wav = wav.squeeze(0).cpu().numpy()
                        
                        if ctx_len > 0:
                            wav = wav[int(ctx_len * self.samples_per_token):]
                        
                        fade = int(0.01 * self.sr)
                        if 0 < fade < len(wav): wav[:fade] *= np.linspace(0.0, 1.0, fade)
                        
                        mv = np.abs(wav).max()
                        if mv > 0.0:
                            target = 0.8
                            wav = wav * (target / max(mv, 1e-6))
                        
                        now = time.time()
                        audio_queue.put({
                            'audio': wav, 'delta': (now - last_chunk_time_worker)*1000,
                            'is_final': False, 'ttfb': (now - start_time)*1000 if chunk_idx == 0 else None
                        })
                        last_chunk_time_worker = now
                        chunk_idx += 1
                    
                    all_tokens = torch.cat([all_tokens, chunk_clean], dim=-1)
                    token_queue.task_done()
                except queue.Empty: continue

        vocoder_thread = threading.Thread(target=vocoder_worker)
        vocoder_thread.start()

        # Shared backend objects
        alignment_analyzer = None
        patched_model = None
        
        try:
            global_chunk_idx = 0
            for text_chunk in text_chunks:
                logger.info(f"🎙️ Processing Text Chunk: '{text_chunk[:50]}...'")
                
                # 1. Prepare T3 for this chunk
                text_tokens = self.model.tokenizer.text_to_tokens(text_chunk, language_id=language_id.lower()).to(self.device)
                text_tokens = torch.cat([text_tokens, text_tokens], dim=0) # Batch-2
                
                sot, eot = self.model.t3.hp.start_text_token, self.model.t3.hp.stop_text_token
                text_tokens = torch.nn.functional.pad(text_tokens, (1, 1), value=sot)
                text_tokens[:, -1] = eot
                
                initial_speech_tokens = self.model.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
                embeds, len_cond = self.model.t3.prepare_input_embeds(
                    t3_cond=self.model.conds.t3, text_tokens=text_tokens,
                    speech_tokens=initial_speech_tokens, cfg_weight=cfg_weight,
                )

                slice_indices = (len_cond, len_cond + text_tokens.size(-1))
                if alignment_analyzer is None:
                    alignment_analyzer = AlignmentStreamAnalyzer(
                        self.model.t3.tfmr, None, text_tokens_slice=slice_indices,
                        eos_idx=self.model.t3.hp.stop_speech_token, lang=language_id,
                    )
                else:
                    alignment_analyzer.reset(slice_indices)

                if patched_model is None:
                    patched_model = T3HuggingfaceBackend(
                        config=self.model.t3.cfg, llama=self.model.t3.tfmr,
                        speech_enc=self.model.t3.speech_emb, speech_head=self.model.t3.speech_head,
                        alignment_stream_analyzer=alignment_analyzer,
                    )
                else:
                    patched_model.alignment_stream_analyzer = alignment_analyzer

                # 2. Start generation for this chunk
                bos_token = torch.tensor([[self.model.t3.hp.start_speech_token]], dtype=torch.long, device=self.device)
                bos_embed = patched_model.speech_enc(bos_token) + self.model.t3.speech_pos_emb.get_fixed_embedding(0)
                bos_embed = torch.cat([bos_embed, bos_embed])
                
                inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
                generated_ids = bos_token.clone()
                
                # Sampling setup
                top_p_warper = TopPLogitsWarper(top_p=top_p)
                min_p_warper = MinPLogitsWarper(min_p=min_p)
                rep_penalty = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

                output = patched_model(
                    inputs_embeds=inputs_embeds, past_key_values=None,
                    use_cache=True, output_attentions=False, output_hidden_states=False, return_dict=True,
                )
                past, chunk_buffer, local_chunk_idx = output.past_key_values, [], 0

                for i in range(max_new_tokens):
                    logits = output.logits[:, -1, :]
                    if cfg_weight > 0:
                        cond_logits, uncond_logits = logits[0:1, :], logits[1:2, :]
                        logits = cond_logits + cfg_weight * (cond_logits - uncond_logits)
                    
                    last_token = generated_ids[0, -1].item()
                    logits = alignment_analyzer.step(logits, next_token=last_token)
                    logits = rep_penalty(generated_ids, logits)
                    
                    if temperature != 1.0: logits = logits / temperature
                    logits = top_p_warper(generated_ids, min_p_warper(generated_ids, logits))
                    
                    next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                    chunk_buffer.append(next_token)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    if next_token.item() == self.model.t3.hp.stop_speech_token:
                        if chunk_buffer: token_queue.put(torch.cat(chunk_buffer, dim=1)[0])
                        break
                    
                    current_chunk_size = 15 if (global_chunk_idx == 0 and local_chunk_idx == 0) else chunk_size_tokens
                    if len(chunk_buffer) >= current_chunk_size:
                        token_queue.put(torch.cat(chunk_buffer, dim=1)[0])
                        chunk_buffer = []
                        local_chunk_idx += 1

                    while not audio_queue.empty(): yield audio_queue.get()

                    next_embed = patched_model.speech_enc(next_token) + self.model.t3.speech_pos_emb.get_fixed_embedding(i + 1)
                    next_embed = torch.cat([next_embed, next_embed])
                    output = patched_model(
                        inputs_embeds=next_embed, past_key_values=past,
                        use_cache=True, output_attentions=True, output_hidden_states=False, return_dict=True,
                    )
                    past = output.past_key_values
                
                global_chunk_idx += 1

            token_queue.put(None)
            vocoder_thread.join()
            while not audio_queue.empty(): yield audio_queue.get()
        finally:
            stop_signal.set()
            if alignment_analyzer: alignment_analyzer.close()
            if vocoder_thread.is_alive(): token_queue.put(None); vocoder_thread.join()
            
            # Explicitly clear cache and collect garbage to prevent slowing down over time
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"✅ ULTIMATE V2V Complete")
