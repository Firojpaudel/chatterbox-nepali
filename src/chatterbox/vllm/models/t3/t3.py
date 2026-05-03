from typing import Iterable, Mapping, Optional, Sequence, Union
import os

import torch
import torch.nn as nn
import random
from transformers.feature_extraction_utils import BatchFeature

from vllm.config import VllmConfig, ModelConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, MultiModalKwargsItem, MultiModalBatchedField
from vllm.multimodal.parse import MultiModalDataParser, ModalityDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalDataDict,
    MultiModalDataItems,
    MultiModalFieldConfig,
    PromptUpdate,
    MultiModalInputs,
    PlaceholderRange,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from chatterbox.vllm.models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from chatterbox.vllm.models.t3.modules.t3_config import T3Config
from .modules.cond_enc import T3Cond, T3CondEnc


PREFILL_COND_START_TOKEN = 695  # [PLACEHOLDER55]; Marks the first token of the conditionals
PREFILL_COND_END_TOKEN = 696  # [PLACEHOLDER56]; Marks the last token of the conditionals
PREFILL_END_TOKEN = 697  # [PLACEHOLDER57]; Marks the end of the prefill block. This corresponds to the start of speech token.

CONDITIONING_SIZE = 34 # 1 for speaker_emb, 0 for clap_emb, 32 for cond_prompt_speech_emb, 1 for emotion_adv

# HACK: We need to be able to distinguish between the prefill tokens and the decode tokens.
# We'll do this by offsetting the speech tokens (only within vLLM) so they don't overlap with the
# normal speech tokens. This way, any token < SPEECH_TOKEN_OFFSET is a prefill token, and any token
# >= SPEECH_TOKEN_OFFSET is a decode token. This will only affect the logits and the encoding logic.
# No effect on the hidden states or the actual Llama model itself.
SPEECH_TOKEN_OFFSET = 2560


class T3ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"conditionals": 1}


class T3MultiModalDummyInputsBuilder(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "[START]Hello, world![STOP]"

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        return { "conditionals": [torch.zeros(CONDITIONING_SIZE, 2048)] * mm_counts["conditionals"] }


class T3MultiModalDataParser(MultiModalDataParser):
    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        conditionals: Optional[torch.Tensor] = mm_data.get("conditionals", None)
        if conditionals is None:
            return MultiModalDataItems({})

        return MultiModalDataItems({
            "conditionals": ConditionalsEmbeddingItems(conditionals)
        })


class ConditionalsEmbeddingItems(ModalityDataItems[torch.Tensor, torch.Tensor]):
    def __init__(self, data: torch.Tensor) -> None:
        super().__init__(data, "conditionals")

    def get_count(self) -> int:
        return 1

    def get(self, index: int) -> torch.Tensor:
        assert index == 0, index
        return self.data

    def get_processor_data(self) -> Mapping[str, torch.Tensor]:
        return {}

    def get_passthrough_data(self) -> Mapping[str, torch.Tensor]:
        return {"conditionals": self.data}


def create_triangular_matrix(m, n):
    # Create row indices and column indices
    row_indices = torch.arange(m).unsqueeze(1)  # Shape: (m, 1)
    col_indices = torch.arange(n).unsqueeze(0)  # Shape: (1, n)

    # Create the triangular mask
    matrix = (col_indices <= row_indices).float()

    return matrix


class T3MultiModalProcessor(BaseMultiModalProcessor[T3ProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return T3MultiModalDataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            conditionals=MultiModalFieldConfig.batched("conditionals")
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        # Bypassed via `apply` method.
        return []

    def _call_hf_processor(
        self,
        prompt: str,
        # Not to be confused with `mm_data` in `self.apply`.
        # This refers to the data to be passed to HF processor.
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        processed_outputs = tokenizer(prompt, return_tensors="pt")
        processed_outputs['conditionals'] = mm_data.get('conditionals', None)
        if processed_outputs['conditionals'] is not None:
            print("processed_outputs", processed_outputs['conditionals'].shape)
        return processed_outputs

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        """
        Process multi-modal inputs to be used in vLLM.

        The main steps are:

        1. Apply HF Processor on prompt text and multi-modal data together,
           outputting token IDs and processed tensors.
        2. Find and update sequences in the token IDs with placeholder tokens.
           The number of placeholder tokens equals the feature size of the
           multi-modal data outputted by the multi-modal encoder.
           (SKIPPED for T3 conditioning)
        3. Extract information about the placeholder tokens from the
           processed token IDs.
           (Stubbed for T3 conditioning)
        """
        mm_items = self._to_mm_items(mm_data)

        (
            prompt_ids,
            mm_kwargs,
            mm_hashes,
            is_update_applied,
        ) = self._apply_hf_processor(
            prompt,
            mm_items,
            hf_processor_mm_kwargs,
            tokenization_kwargs,

            # Skip prompt caching calculation for now
            return_mm_hashes=False,
        )

        # We are going to apply custom logic to squish the embeddings in the right format.
        # The final embedding will look like <| cond | text | speech |>
        #
        # For prompt IDs, we're going to replace the input tokens that match the conditionals with a
        # sequence of tokens that won't normally appear in the text prompt. This will help us unbatch
        # batched inputs.
        final_prompt_ids = [
            # Conditionals (totaling CONDITIONING_SIZE tokens)
            PREFILL_COND_START_TOKEN,
            *([prompt_ids[0]] * (CONDITIONING_SIZE-2)),
            PREFILL_COND_END_TOKEN,

            # Text prompt,
            *prompt_ids,

            # Start of speech token / End of prefill block
            PREFILL_END_TOKEN,
        ]

        # HACK: Because vLLM can split the prefill across multiple batches, we need some way to
        # remember the offset of each text token.
        # We'll do this by extending the 32x1024 embedding to <len(final_prompt_ids)>x1024, filling in
        # the first 32x1024 with the original conditionals, and the rest with a triangular matrix of 1s
        # which will encode the offset of each text token.
        conditionals = mm_data.get("conditionals", None)
        assert conditionals is not None and len(conditionals) > 0, "Conditionals are required for prefill"
        assert len(conditionals) == 1, "Only one conditional embedding is supported for prefill"
        assert conditionals[0].shape[0] == CONDITIONING_SIZE, "Conditionals must be CONDITIONING_SIZE tokens long"

        new_conditionals = conditionals[0]
        
        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=final_prompt_ids,
            mm_kwargs=MultiModalKwargs({"conditionals": [new_conditionals]}),
            mm_hashes={
                # Assign a random hash for now, because we're not actually hashing the multimodal data.
                "conditionals": [str(random.random())],
            },
            mm_placeholders={
                # HACK: Tell vLLM that the conditionals modify the entire prompt. This will cause our hacked embeddings
                #       to be injected into the entire prompt, rather than just the conditioning portion.
                "conditionals": [PlaceholderRange(offset=0, length=len(final_prompt_ids), is_embed=None)]
            },
        )


@MULTIMODAL_REGISTRY.register_processor(T3MultiModalProcessor,
                                        info=T3ProcessingInfo,
                                        dummy_inputs=T3MultiModalDummyInputsBuilder)
class T3VllmModel(nn.Module, VllmModelForTextGeneration, SupportsMultiModal):
    """Native vLLM implementation of the Chatterbox T3 """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        # The model is initialized with hidden_size=2048 natively.
        # We will dynamically expand the 1024-dim weights to 2048-dim block-diagonal
        # matrices in load_weights. This allows the 32 heads to process the cond
        # and uncond sequences independently in a single native vLLM engine pass!
        self.vllm_config = vllm_config
        self.cfg: ModelConfig = vllm_config.model_config

        # Initialize LLaMA backbone
        self.tfmr = LlamaModel(vllm_config=vllm_config, prefix=prefix + ".tfmr")

        # Hot-patch vLLM's RMSNorm to prevent channel mixing!
        # Standard RMSNorm computes the mean across the full 2048 dimension, which breaks our
        # block-diagonal CFG by allowing the cond and uncond streams to scale each other.
        # This custom module computes the RMS across the 1024-dim streams completely independently.
        class BlockDiagonalRMSNorm(torch.nn.Module):
            def __init__(self, vllm_rmsnorm):
                super().__init__()
                self.weight = vllm_rmsnorm.weight
                self.variance_epsilon = vllm_rmsnorm.variance_epsilon
                self.dim = self.weight.shape[0] // 2
            
            def forward(self, x, residual=None):
                if residual is not None:
                    x = x + residual
                    residual = x
                
                # Cast to float32 for precision, split into the two streams
                x_fp32 = x.to(torch.float32)
                x1, x2 = x_fp32.split([self.dim, self.dim], dim=-1)
                w1, w2 = self.weight.split([self.dim, self.dim], dim=-1)
                
                # DEBUG: Monitor stream isolation and stats
                if getattr(self, '_log_count', 0) < 3:
                    c_mean, c_std = x1.mean().item(), x1.std().item()
                    u_mean, u_std = x2.mean().item(), x2.std().item()
                    if torch.isnan(x1).any() or torch.isnan(x2).any():
                        print(f"CRITICAL: NaN detected in RMSNorm input! Cond NaN: {torch.isnan(x1).any()}, Uncond NaN: {torch.isnan(x2).any()}")
                    # Only print if they are significantly different or it's the first few layers
                    print(f"DEBUG: RMSNorm Stream Stats - Cond: mean={c_mean:.4f}, std={c_std:.4f} | Uncond: mean={u_mean:.4f}, std={u_std:.4f}")
                    self._log_count = getattr(self, '_log_count', 0) + 1

                var1 = x1.pow(2).mean(-1, keepdim=True)
                x1_norm = x1 * torch.rsqrt(var1 + self.variance_epsilon)
                
                var2 = x2.pow(2).mean(-1, keepdim=True)
                x2_norm = x2 * torch.rsqrt(var2 + self.variance_epsilon)
                
                x_norm_fp16 = torch.cat([x1_norm, x2_norm], dim=-1).to(self.weight.dtype)
                out = x_norm_fp16 * self.weight
                
                if residual is not None:
                    return out, residual
                return out

        # Apply the patch to all layers and the final output norm
        for layer in self.tfmr.layers:
            layer.input_layernorm = BlockDiagonalRMSNorm(layer.input_layernorm)
            layer.post_attention_layernorm = BlockDiagonalRMSNorm(layer.post_attention_layernorm)
        self.tfmr.norm = BlockDiagonalRMSNorm(self.tfmr.norm)


        # Initialize custom components
        hf_cfg = vllm_config.model_config.hf_config
        print(f"DEBUG: vLLM Internal Config - hidden_size: {hf_cfg.hidden_size}, num_heads: {hf_cfg.num_attention_heads}")
        
        is_multilingual = getattr(hf_cfg, 'is_multilingual', False)

        self.t3conf = T3Config.multilingual() if is_multilingual else T3Config()
        self.dim = 1024
        self.cond_enc = T3CondEnc(self.t3conf)
        self.text_emb = nn.Embedding(self.t3conf.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(self.t3conf.speech_tokens_dict_size, self.dim)

        # custom position embedding
        max_text_seq_len = self.t3conf.max_text_tokens + 2
        self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

        max_mel_seq_len = self.t3conf.max_speech_tokens + 2 + 2
        self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        # self.text_head = nn.Linear(self.dim, self.t3conf.text_tokens_dict_size, bias=False)
        self.speech_head = ParallelLMHead(
            num_embeddings=self.t3conf.speech_tokens_dict_size,
            embedding_dim=self.dim,
            padding_size=1,
            prefix=prefix + ".speech_head",
        )
        self.logits_processor = LogitsProcessor(self.t3conf.speech_tokens_dict_size)

        self.cfg_scale = float(os.environ.get("CHATTERBOX_CFG_SCALE", "1.5"))
        print("Applying CFG scale:", self.cfg_scale)
        self.prompt_len_cache = {}


    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params: set[str] = set()
        state_dicts = {}
        hf_llama_weights = {}
        for name, weight in weights:
            # Strip prefixes from Nepali fine-tuned checkpoints
            if name.startswith("patched_model."):
                name = name[len("patched_model."):]
            if name.startswith("model."):
                name = name[len("model."):]

            # Llama weights need to be passed through vllm's load_weights rather than load_state_dict
            if name.startswith("tfmr."):
                subname = name[5:]
                # Store original unexpanded weights to save System RAM
                hf_llama_weights[subname] = weight
                continue
            loaded_params.add(name)
            attr, subname = name.split('.', 1)
            state_dict = state_dicts.get(attr, {})
            state_dict[subname] = weight
            state_dicts[attr] = state_dict

        for attr, state_dict in state_dicts.items():
            if hasattr(self, attr):
                print(f"Loading vLLM weights: {attr} ({list(state_dict.keys())})")
                getattr(self, attr).load_state_dict(state_dict, strict=False)

        # Use a generator to expand weights just-in-time to prevent System RAM OOM kills
        def generate_expanded_llama_weights():
            # Pop items from the dictionary to aggressively free System RAM during iteration!
            while hf_llama_weights:
                subname, weight = hf_llama_weights.popitem()
                if any(x in subname for x in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]):
                    dim0, dim1 = weight.shape
                    new_weight = torch.zeros((dim0 * 2, dim1 * 2), dtype=weight.dtype, device=weight.device)
                    new_weight[:dim0, :dim1] = weight
                    new_weight[dim0:, dim1:] = weight
                    yield subname, new_weight
                    del new_weight
                elif any(x in subname for x in ["input_layernorm.weight", "post_attention_layernorm.weight", "norm.weight", "norm.bias"]):
                    new_weight = torch.cat([weight, weight], dim=0)
                    yield subname, new_weight
                    del new_weight
                elif "o_proj" in subname or "down_proj" in subname:
                    # BLOCK-DIAGONAL FIX: Perfect isolation barrier
                    new_weight = torch.zeros((weight.shape[0] * 2, weight.shape[1] * 2), dtype=weight.dtype, device=weight.device)
                    new_weight[:weight.shape[0], :weight.shape[1]] = weight
                    new_weight[weight.shape[0]:, weight.shape[1]:] = weight
                    yield subname, new_weight
                    del new_weight
                elif "embed_tokens.weight" in subname:
                    # weight is [8, 1024]. Expand to [8, 2048]
                    expanded_weight = torch.cat([weight, weight], dim=1)
                    # Pad to the full 32000 vocab size we defined in config
                    final_weight = torch.zeros((32000, expanded_weight.shape[1]), 
                                              device=expanded_weight.device, 
                                              dtype=expanded_weight.dtype)
                    final_weight[:expanded_weight.shape[0]] = expanded_weight
                    yield subname, final_weight
                    del final_weight, expanded_weight
                else:
                    yield subname, weight
                
                # Delete original weight reference so Python GC reclaims it immediately
                del weight

        llama_loaded_params = self.tfmr.load_weights(generate_expanded_llama_weights())
        loaded_params.update('tfmr.' + i for i in llama_loaded_params)

        # Precompute text positional embeddings
        text_position_ids = torch.arange(self.t3conf.max_text_tokens + 2, device=self.text_pos_emb.emb.weight.device)
        self.precomputed_text_pos_emb = self.text_pos_emb.get_fixed_embedding(text_position_ids)[0]

        # Precompute speech positional embeddings
        speech_position_ids = torch.arange(self.t3conf.max_speech_tokens + 2 + 2, device=self.speech_pos_emb.emb.weight.device)
        self.precomputed_speech_pos_emb = self.speech_pos_emb.get_fixed_embedding(speech_position_ids)[0]

        return loaded_params


    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        conditionals: Optional[list[list[T3Cond]]] = kwargs.get("conditionals", [])
        return [batch[0] for batch in conditionals]


    def split_prefill_decode(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[torch.Tensor],
        positions: torch.Tensor,
    ) -> list[tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
        if len(input_ids) == 0:
            return []
        
        remaining_multimodal_embeddings = torch.cat(multimodal_embeddings, dim=0) if multimodal_embeddings else None

        in_prefill_block = input_ids[0] < SPEECH_TOKEN_OFFSET
        output = []
        id_buffer = []
        pos_buffer = []

        for input_id, pos in zip(input_ids, positions):
            if (in_prefill_block != (input_id < SPEECH_TOKEN_OFFSET)) or (input_id == PREFILL_COND_START_TOKEN):
                if id_buffer:
                    cur_ids = torch.stack(id_buffer).to(input_ids.device)
                    cur_pos = torch.stack(pos_buffer).to(input_ids.device)
                    if in_prefill_block and remaining_multimodal_embeddings is not None:
                        # CONDITIONING: Only take the 34 embeddings we provided
                        take_n = min(len(remaining_multimodal_embeddings), CONDITIONING_SIZE)
                        mme = remaining_multimodal_embeddings[:take_n]
                        remaining_multimodal_embeddings = remaining_multimodal_embeddings[take_n:]
                        output.append((cur_ids, mme, cur_pos))
                    else:
                        output.append((cur_ids, None, cur_pos))

                id_buffer = []
                pos_buffer = []
                in_prefill_block = (input_id < SPEECH_TOKEN_OFFSET)

            id_buffer.append(input_id)
            pos_buffer.append(pos)

        if id_buffer:
            cur_ids = torch.stack(id_buffer).to(input_ids.device)
            cur_pos = torch.stack(pos_buffer).to(input_ids.device)
            if in_prefill_block and remaining_multimodal_embeddings is not None:
                take_n = min(len(remaining_multimodal_embeddings), CONDITIONING_SIZE)
                mme = remaining_multimodal_embeddings[:take_n]
                remaining_multimodal_embeddings = remaining_multimodal_embeddings[take_n:]
                output.append((cur_ids, mme, cur_pos))
            else:
                output.append((cur_ids, None, cur_pos))

        return output


    def get_input_embeddings(self, input_ids: torch.Tensor, multimodal_embeddings: list[MultiModalEmbeddings], positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        # HACK: vLLM dummy runs pass token IDs of all 0s for profiling.
        if input_ids is not None and torch.all(input_ids == 0):
             return torch.zeros((len(input_ids), self.dim * 2), 
                                dtype=self.speech_head.weight.dtype, 
                                device=input_ids.device)

        # Get the unique ID for this request's conditioning data
        cond_tensor = multimodal_embeddings[0] if (multimodal_embeddings and len(multimodal_embeddings) > 0) else None
        cond_id = id(cond_tensor) if cond_tensor is not None else None

        if input_ids is not None and (input_ids >= SPEECH_TOKEN_OFFSET).any():
            speech_ids = torch.clamp(input_ids - SPEECH_TOKEN_OFFSET, min=0)
            embeds = self.speech_emb(speech_ids)
            
            # Use cached prompt length to calculate relative speech position
            prompt_len = 36 + 20 # Conservative fallback
            if cond_id in self.prompt_len_cache:
                prompt_len = self.prompt_len_cache[cond_id]
            
            rel_pos = (positions - prompt_len).clamp(min=0)
            embeds = embeds + self.precomputed_speech_pos_emb[rel_pos]
            
            out = torch.cat([embeds, embeds], dim=1)
            return out
        else:
            out = []
            for ids, multimodal_embedding, block_positions in self.split_prefill_decode(input_ids, multimodal_embeddings, positions):
                if multimodal_embedding is None:
                    if (ids < SPEECH_TOKEN_OFFSET).any():
                        text_ids = ids
                        pos_indices = block_positions.clamp(0, self.t3conf.max_text_tokens - 1)
                        text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[pos_indices]
                        
                        cond_embeds = text_emb
                        uncond_embeds = torch.zeros_like(text_emb)
                        
                        final_embeds = torch.cat([cond_embeds, uncond_embeds], dim=1)
                        out.append(final_embeds)
                        continue
                    else:
                        speech_ids = torch.clamp(ids - SPEECH_TOKEN_OFFSET, min=0)
                        embeds = self.speech_emb(speech_ids)
                        pos_indices = block_positions.clamp(0, self.t3conf.max_speech_tokens + 3)
                        embeds = embeds + self.precomputed_speech_pos_emb[pos_indices]
                        final_embeds = torch.cat([embeds, embeds], dim=1)
                        out.append(final_embeds)
                        continue

                # We're in the prefill stage, and need to wrangle the multimodal embeddings into the right format.
                # Embeddings are in the format of <| cond | text | speech |>
                #
                # However, due to vLLM batching, we may only get the first half or the last half of the prefill block.
                #
                # We're going to assume that the prefill block only span at most two batches - i.e. we'll always have
                # at least the start token or the end token. More is theorically possible, but not an edge case I'm going
                # to handle here.
                #
                # Note that we may have as little as a single token from the block.

                # To ease the implementation logic, we're going to implement each case separately.

                if ids[0] == PREFILL_COND_START_TOKEN and ids[-1] == PREFILL_END_TOKEN:
                    # We have the full prefill block.

                    # The first 34 tokens are the cond portion. The remainder, except for the last token are the text
                    # portion. The last token is a placeholder for the start of speech token.
                    # Alignment fix: Start token [0], Cond [1..34], Text [35..-1], Stop [-1]
                    start_token_emb = self.text_emb(ids[0:1])
                    text_ids = ids[CONDITIONING_SIZE+1:-1]
                    text_token_emb = self.text_emb(text_ids)
                    text_pos_emb = self.precomputed_text_pos_emb[0:len(text_ids)]
                    text_emb = text_token_emb + text_pos_emb

                    start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
                    start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0] + self.precomputed_speech_pos_emb[0:1]

                    # Generate version with both text and no-text embeddings for CFG
                    conditioning_emb = multimodal_embedding[0:CONDITIONING_SIZE]
                    conditioning_emb = conditioning_emb.to(start_token_emb.device, dtype=start_token_emb.dtype)
                    cond_embeds = torch.cat([start_token_emb, conditioning_emb, text_emb, start_of_speech_emb], dim=0)
                    # Unconditional stream MUST have zeroed text but RETAINED positions!
                    uncond_text_emb = self.precomputed_text_pos_emb[0:len(text_ids)]
                    uncond_embeds = torch.cat([start_token_emb, conditioning_emb, uncond_text_emb, start_of_speech_emb], dim=0)

                    # Concatenate into one giant tensor, which will be split in the forward pass
                    final_embeds = torch.cat([cond_embeds, uncond_embeds], dim=1)
                    
                    # Store prompt length for decoding stage positional alignment
                    if cond_id is not None:
                         self.prompt_len_cache[cond_id] = ids.shape[0]
                         
                    out.append(final_embeds)
                elif ids[0] == PREFILL_COND_START_TOKEN:
                    # We have the start of the prefill block.
                    # The only thing we an assume here is that we don't have the end token, so we can skip the start of speech token.
                    # print("t3/get_input_embeddings/start of prefill block")

                    # The first 34 tokens are the cond portion. The remainder are the text portion.
                    # This logic should correctly handle:
                    #  - We don't have any text tokens in this batch
                    #  - We only have part of the conditioning tokens in this batch
                    text_ids = ids[CONDITIONING_SIZE:]
                    text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)]

                    # Generate version with both text and no-text embeddings for CFG
                    conditioning_emb = multimodal_embedding[0:min(CONDITIONING_SIZE, len(multimodal_embedding))]
                    cond_embeds = torch.cat([conditioning_emb, text_emb], dim=0)
                    uncond_embeds = torch.cat([conditioning_emb, torch.zeros_like(text_emb)], dim=0)

                    # Concatenate into one giant tensor, which will be split in the forward pass
                    final_embeds = torch.cat([cond_embeds, uncond_embeds], dim=1)
                    assert len(final_embeds) == len(ids), "Number of output elements does not match number of input elements"
                    out.append(final_embeds)
                elif ids[-1] == PREFILL_END_TOKEN:
                    # We have the end of the prefill block.
                    # The only thing we an assume here is that we have the start of speech token,
                    # and that our conditioning embeddings will at minimum be truncated. We can't
                    # assume anything about the text portion.

                    # Check if the end-of-conditioning token is present. If it is, we can assume that
                    # we have the full text block. If it's not, we can assume that there's no conditioning
                    # portion.
                    indices = torch.where(ids == PREFILL_COND_END_TOKEN)[0]
                    if len(indices) > 0:
                        # print("t3/get_input_embeddings/end of prefill block, has conditioning")
                        
                        # We have the full text input, and it's from indices[0]+1 to the end of the input.
                        # (indices[0] is the end of the conditioning)
                        text_ids = ids[indices[0]+1:-1]
                        text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)]
                        
                        start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
                        start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0] + self.precomputed_speech_pos_emb[0:1]

                        conditioning_emb = multimodal_embedding[:indices[0]+1]
                        
                        cond_embeds = torch.cat([conditioning_emb, text_emb, start_of_speech_emb], dim=0)
                        
                        # Unconditional text stream needs positional embeddings, but token embeddings are zeroed!
                        uncond_text_emb = self.precomputed_text_pos_emb[0:len(text_ids)]
                        uncond_embeds = torch.cat([conditioning_emb, uncond_text_emb, start_of_speech_emb], dim=0)

                        final_embeds = torch.cat([cond_embeds, uncond_embeds], dim=1)
                        # assert len(final_embeds) == len(ids), "Number of output elements does not match number of input elements"
                        out.append(final_embeds)
                    else:
                        # We don't have the conditioning portion, and we may only have part of the text portion.
                        # print("t3/get_input_embeddings/end of prefill block, no conditioning")

                        # conditioning_emb is the first 34 tokens of the sequence.
                        conditioning_emb = multimodal_embedding # (34, 1024)
                        
                        text_ids = ids[1:-1]
                        
                        # DEBUG: Log prefill text length
                        if getattr(self, '_log_prefill', True):
                            print(f"DEBUG: get_input_embeddings (Prefill) - text_len: {len(text_ids)}")
                            self._log_prefill = False

                        # Standard text embeddings + positional embeddings
                        text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)]

                        start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
                        # The first speech token (start of speech) is always at position 0 in the speech positional embeddings
                        start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0]  + self.precomputed_speech_pos_emb[0:1]
                        
                        # CORRECT ORDER: [Conditioning (34), Text (N), Start-of-Speech (1)]
                        cond_embeds = torch.cat([conditioning_emb, text_emb, start_of_speech_emb], dim=0)
                        
                        # DIAGNOSTIC TEST: Force the unconditional stream to be ZEROS (except start-of-speech)
                        # This should make the uncond stream completely different.
                        # If the output is still identical, the backbone is mixing them.
                        uncond_text_emb = torch.zeros_like(text_emb)
                        uncond_embeds = torch.cat([torch.zeros_like(conditioning_emb), uncond_text_emb, start_of_speech_emb], dim=0)
                        
                        final_embeds = torch.cat([cond_embeds, uncond_embeds], dim=1)
                        assert len(final_embeds) == len(ids), f"Length mismatch: {len(final_embeds)} vs {len(ids)}"
                        out.append(final_embeds)

                else:
                    # Something else - we don't know what to do with this.
                    print("t3/get_input_embeddings/ERROR: prefill block contains neither start nor end. Please report this issue.")
                    print("t3/get_input_embeddings/ids", ids.shape, ids.dtype, ids)
                    print("t3/get_input_embeddings/multimodal_embedding", multimodal_embedding.shape if multimodal_embedding is not None else None)
                    raise ValueError(f"Unknown prefill block: {ids}")

            output = torch.cat(out, dim=0)

            # if len(output) != len(input_ids):
            #     print("t3/get_input_embeddings/output", output.shape, output.dtype)
            #     print("t3/get_input_embeddings/input_ids", input_ids.shape, input_ids.dtype)
            #     print("t3/get_input_embeddings/multimodal_embeddings", len(multimodal_embeddings))
            return output


    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        # Split the hidden state vector into the three parts
        cond_hidden_states, uncond_hidden_states = hidden_states.split([self.dim, self.dim], dim=1)
        
        # DEBUG: Log raw hidden states for exact comparison
        if cond_hidden_states.numel() > 0:
            diff = (cond_hidden_states - uncond_hidden_states).abs().max().item()
            print(f"DEBUG: compute_logits - MAX STREAM DIFF: {diff:.8f}")
            print(f"DEBUG: RAW Cond Hidden[0, :5]: {cond_hidden_states[0, :5].detach().cpu().tolist()}")
            print(f"DEBUG: RAW Uncond Hidden[0, :5]: {uncond_hidden_states[0, :5].detach().cpu().tolist()}")

        # DIAGNOSTIC LOGGING
        if getattr(self, '_log_count', 0) < 5:
            c_mean, c_std = cond_hidden_states.mean().item(), cond_hidden_states.std().item()
            u_mean, u_std = uncond_hidden_states.mean().item(), uncond_hidden_states.std().item()
            print(f"DEBUG: compute_logits - Cond Hidden: mean={c_mean:.4f}, std={c_std:.4f} | Uncond Hidden: mean={u_mean:.4f}, std={u_std:.4f}")
            self._log_count = getattr(self, '_log_count', 0) + 1
    
        cond_logits = self.logits_processor(self.speech_head, cond_hidden_states, sampling_metadata)
        uncond_logits = self.logits_processor(self.speech_head, uncond_hidden_states, sampling_metadata)

        if getattr(self, '_log_logits', 0) < 5:
            c_l_mean, c_l_std = cond_logits.mean().item(), cond_logits.std().item()
            u_l_mean, u_l_std = uncond_logits.mean().item(), uncond_logits.std().item()
            print(f"DEBUG: compute_logits - Cond Logits: mean={c_l_mean:.4f}, std={c_l_std:.4f} | Uncond Logits: mean={u_l_mean:.4f}, std={u_l_std:.4f}")
            self._log_logits = getattr(self, '_log_logits', 0) + 1
    
        # 4. Final Combination
        logits = cond_logits + self.cfg_scale * (cond_logits - uncond_logits)
        
        # 5. Safety: Logit Masking
        # We MUST ensure the model only samples valid speech tokens (IDs 2560 to 2560+8192)
        # Any token outside this range will be forced to -inf.
        # This prevents "leaking" into text tokens which would cause garbage audio.
        mask = torch.ones_like(logits, dtype=torch.bool)
        # 8192 is the number of speech tokens. 
        # We also allow the stop token (8194 relative to our speech_head) if it exists in the tensor.
        mask[:, :min(8192, logits.shape[1])] = False 
        if logits.shape[1] > 8194:
            mask[:, 8194] = False
        else:
            # If the head is smaller than 8195, allow the last few tokens as a fallback for STOP
            mask[:, -2:] = False 
        
        logits.masked_fill_(mask, -float('inf'))

        # 6. Diagnostics
        if getattr(self, '_log_final_logits', 0) < 5:
             f_mean, f_std = logits.mean().item(), logits.std().item()
             print(f"DEBUG: compute_logits - Final Combined Logits (Masked): mean={f_mean:.4f}, std={f_std:.4f}")
             self._log_final_logits = getattr(self, '_log_final_logits', 0) + 1
    
        # 7. Offset the logits so the resulting speech token is +SPEECH_TOKEN_OFFSET 
        padding = torch.full((logits.shape[0], SPEECH_TOKEN_OFFSET), float('-inf'), 
                             dtype=logits.dtype, device=logits.device)
        logits = torch.cat([padding, logits], dim=1)
        
        # 8. Pad up to the full 32000 vocab size expected by vLLM config
        if logits.shape[1] < 32000:
            extra_padding = torch.full((logits.shape[0], 32000 - logits.shape[1]), float('-inf'), 
                                      dtype=logits.dtype, device=logits.device)
            logits = torch.cat([logits, extra_padding], dim=1)
        elif logits.shape[1] > 32000:
            logits = logits[:, :32000]
            
        return logits


    def forward(
        self,
        input_ids: Optional[torch.Tensor],  # Almost always None
        positions: torch.Tensor,  # Position IDs since start of the context (i.e. since the first conditional token)
        intermediate_tensors: Optional[IntermediateTensors],  # Almost always None
        inputs_embeds: Optional[torch.Tensor] = None,  # The actual inputs to the model
        **kwargs: object,
    ) -> torch.Tensor:
        # print("t3/inputs_embeds", inputs_embeds.shape, inputs_embeds.dtype)
        # print("t3/positions", positions.shape, positions.dtype)

        # These are usually NULL:
        # print("t3/intermediate_tensors", intermediate_tensors)
        # print("t3/input_ids", input_ids)
        # print("t3/kwargs", kwargs)

        # If inputs_embeds is not provided (e.g. during dummy runs), we compute them ourselves
        if inputs_embeds is None:
            # We use an empty list for multimodal_embeddings as it's not used during decode
            inputs_embeds = self.get_input_embeddings(input_ids, [], positions=positions)

        # inputs_embeds is ALREADY concatenated along the hidden_size dimension (dim=-1)
        # because get_input_embeddings does: torch.cat([cond_embeds, uncond_embeds], dim=1)
        # where dim=1 is the hidden dimension. Thus, its shape is [seq_len, 2048].
        
        # Remove multimodal-specific arguments before passing to the backbone transformer
        # as LlamaModel.forward does not expect them.
        kwargs.pop("conditionals", None)
        kwargs.pop("multimodal_embeddings", None)

        # We process the 2048-dim embeddings through our expanded 2048-dim backbone natively!
        hidden_states = self.tfmr(
            input_ids=None,
            positions=positions, # Native sequence length, no batch duplication hack!
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds, # [N, 2048]
            **kwargs
        )
        
        # Return the 2048-dim hidden states. They will be split in compute_logits.
        return hidden_states

    def get_language_model(self) -> torch.nn.Module:
        return self.tfmr