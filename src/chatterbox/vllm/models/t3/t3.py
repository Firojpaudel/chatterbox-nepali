from typing import Iterable, Mapping, Optional, Sequence, Union, Tuple, Any
import os
import torch
import torch.nn as nn
import random
from transformers.feature_extraction_utils import BatchFeature

from vllm.config import VllmConfig, ModelConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
try:
    from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, supports_vllm_gen
except ImportError:
    from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal
    def supports_vllm_gen(cls): return cls
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
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from chatterbox.vllm.models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from chatterbox.vllm.models.t3.modules.t3_config import T3Config
from .modules.cond_enc import T3Cond, T3CondEnc


PREFILL_COND_START_TOKEN = 695
PREFILL_COND_END_TOKEN = 696
PREFILL_END_TOKEN = 697

SPEECH_TOKEN_OFFSET = 2560
CONDITIONING_SIZE = 34
FIXED_TEXT_LEN = 200 
SPEECH_START_POS = CONDITIONING_SIZE + FIXED_TEXT_LEN + 1 


class T3ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"conditionals": 1}


class T3MultiModalDummyInputsBuilder(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "[START]Hello![STOP]"

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        return { "conditionals": [torch.zeros(CONDITIONING_SIZE, 1024)] * mm_counts["conditionals"] }


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


class T3MultiModalProcessor(BaseMultiModalProcessor[T3ProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return T3MultiModalDataParser()

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs) -> Mapping[str, MultiModalFieldConfig]:
        return dict(conditionals=MultiModalFieldConfig.batched("conditionals"))

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs) -> Sequence[PromptUpdate]:
        return []

    def _call_hf_processor(self, prompt: str, mm_data, mm_kwargs, tok_kwargs) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        processed_outputs = tokenizer(prompt, return_tensors="pt")
        processed_outputs['conditionals'] = mm_data.get('conditionals', None)
        return processed_outputs

    def apply(self, prompt: Union[str, list[int]], mm_data: MultiModalDataDict, 
             hf_processor_mm_kwargs, tokenization_kwargs=None, return_mm_hashes=False) -> MultiModalInputs:
        mm_items = self._to_mm_items(mm_data)
        prompt_ids, mm_kwargs, mm_hashes, _ = self._apply_hf_processor(
            prompt, mm_items, hf_processor_mm_kwargs, tokenization_kwargs, return_mm_hashes=False
        )

        actual_prompt_ids = prompt_ids[:FIXED_TEXT_LEN]
        padding_len = FIXED_TEXT_LEN - len(actual_prompt_ids)
        
        final_prompt_ids = [
            PREFILL_COND_START_TOKEN,
            *([prompt_ids[0]] * (CONDITIONING_SIZE - 2)),
            PREFILL_COND_END_TOKEN,
            *actual_prompt_ids,
            *([0] * padding_len),
            PREFILL_END_TOKEN,
        ]

        conditionals = mm_data.get("conditionals", None)
        assert conditionals is not None and len(conditionals) == 1
        
        new_conditionals = torch.cat([
            conditionals[0],
            torch.zeros(FIXED_TEXT_LEN + 1, conditionals[0].shape[1]).to(conditionals[0].device),
        ], dim=0)

        new_mm_kwargs = MultiModalKwargs.from_items([
            MultiModalKwargsItem.from_elems(
                MultiModalBatchedField().build_elems(
                    modality="conditionals", key="conditionals", data=[new_conditionals],
                )
            )
        ])

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=final_prompt_ids,
            mm_kwargs=new_mm_kwargs,
            mm_hashes={"conditionals": [str(random.random())]},
            mm_placeholders={
                "conditionals": [PlaceholderRange(offset=0, length=len(final_prompt_ids), is_embed=None)]
            },
        )


@MULTIMODAL_REGISTRY.register_processor(T3MultiModalProcessor, info=T3ProcessingInfo, dummy_inputs=T3MultiModalDummyInputsBuilder)
@supports_vllm_gen
class T3VllmModel(nn.Module, VllmModelForTextGeneration, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        # Set hidden size for Llama back-end to 1024
        vllm_config.model_config.hf_config.hidden_size = 1024
        self.vllm_config = vllm_config
        self.cfg: ModelConfig = vllm_config.model_config

        self.tfmr = LlamaModel(vllm_config=vllm_config, prefix=prefix + ".tfmr")
        text_tokens_dict_size = 704 if self.cfg.tokenizer == "EnTokenizer" else 2455

        self.t3conf = T3Config()
        self.dim = self.t3conf.n_channels # 1024
        self.cond_enc = T3CondEnc(self.t3conf)
        self.text_emb = nn.Embedding(text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(self.t3conf.speech_tokens_dict_size, self.dim)

        self.text_pos_emb = LearnedPositionEmbeddings(self.t3conf.max_text_tokens + 2, self.dim)
        self.speech_pos_emb = LearnedPositionEmbeddings(self.t3conf.max_speech_tokens + 4, self.dim)

        self.speech_head = ParallelLMHead(
            num_embeddings=self.t3conf.speech_tokens_dict_size,
            embedding_dim=self.dim,
            padding_size=1,
            prefix=prefix + ".speech_head",
        )
        self.logits_processor = LogitsProcessor(self.t3conf.speech_tokens_dict_size)
        self.cfg_scale = float(os.environ.get("CHATTERBOX_CFG_SCALE", "0.5"))
        # Default to safe mode for stability
        self.SAFE_MODE = os.environ.get("CHATTERBOX_SAFE_MODE", "True") == "True"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params: set[str] = set()
        state_dicts = {}
        hf_llama_weights = {}
        counts = {"tfmr": 0, "cond_enc": 0, "text_emb": 0, "speech_emb": 0, "other": 0}
        
        print("--- T3 Weight Loading Started ---")
        for raw_name, weight in weights:
            name = raw_name
            is_backbone = False
            subname = ""
            for marker in ["layers.", "embed_tokens.", "norm.", "tfmr."]:
                if marker in name:
                    is_backbone = True
                    subname = name[name.find(marker):]
                    if marker == "tfmr.": subname = subname[5:]
                    break
            
            if is_backbone:
                if subname == "embed_tokens.weight" and weight.shape[0] < 32000:
                    padding = torch.zeros((32000 - weight.shape[0], weight.shape[1]), dtype=weight.dtype, device=weight.device)
                    weight = torch.cat([weight, padding], dim=0)
                hf_llama_weights[subname] = weight
                counts["tfmr"] += 1
                continue
            
            loaded_params.add(raw_name)
            if '.' not in name:
                counts["other"] += 1
                continue
            attr, subname = name.split('.', 1)
            if attr in counts: counts[attr] += 1
            else: counts["other"] += 1
            state_dict = state_dicts.get(attr, {})
            state_dict[subname] = weight
            state_dicts[attr] = state_dict

        for attr, state_dict in state_dicts.items():
            if hasattr(self, attr): getattr(self, attr).load_state_dict(state_dict, strict=False)

        llama_loaded = self.tfmr.load_weights(hf_llama_weights.items())
        loaded_params.update('tfmr.' + i for i in llama_loaded)

        print(f"--- T3 Weight Load Summary ---")
        print(f"  Backbone (tfmr): {counts['tfmr']} tensors loaded.")
        
        # Precompute positional embeddings
        dev = self.speech_emb.weight.device
        self.precomputed_text_pos_emb = self.text_pos_emb.get_fixed_embedding(torch.arange(self.t3conf.max_text_tokens + 2, device=dev))[0]
        self.precomputed_speech_pos_emb = self.speech_pos_emb.get_fixed_embedding(torch.arange(self.t3conf.max_speech_tokens + 4, device=dev))[0]
        return loaded_params

    def get_input_embeddings(self, input_ids: torch.Tensor, multimodal_embeddings: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            is_decoding = (input_ids >= SPEECH_TOKEN_OFFSET).all()
            return self.speech_emb(input_ids - SPEECH_TOKEN_OFFSET) if is_decoding else self.text_emb(input_ids)
            
        out = []
        for i, ids in enumerate(input_ids):
            mme = multimodal_embeddings[i]
            cond_emb = mme[0:CONDITIONING_SIZE]
            text_ids = ids[CONDITIONING_SIZE:CONDITIONING_SIZE+FIXED_TEXT_LEN]
            pad_mask = (text_ids != 0)
            actual_text_len = pad_mask.sum().item()
            
            text_emb = self.text_emb(text_ids)
            if actual_text_len > 0:
                text_emb[:actual_text_len] += self.precomputed_text_pos_emb[:actual_text_len].to(text_emb.device)
            
            start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
            start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0] + self.precomputed_speech_pos_emb[0:1].to(ids.device)
            
            cond_embeds = torch.cat([cond_emb, text_emb, start_of_speech_emb], dim=0)
            if not self.SAFE_MODE:
                uncond_embeds = torch.cat([cond_emb, torch.zeros_like(text_emb), start_of_speech_emb], dim=0)
                # Concatenate along embedding dimension (dim=1) for CFG splitting in forward()
                cond_embeds = torch.cat([cond_embeds, uncond_embeds], dim=1)
            out.append(cond_embeds)
        return torch.cat(out, dim=0)

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        if self.SAFE_MODE:
            logits = self.logits_processor(self.speech_head, hidden_states, sampling_metadata)
        else:
            # Doubled batch mode: split on batch dimension (dim=0)
            cond_hs, uncond_hs = hidden_states.chunk(2, dim=0)
            cond_logits = self.logits_processor(self.speech_head, cond_hs, sampling_metadata)
            uncond_logits = self.logits_processor(self.speech_head, uncond_hs, sampling_metadata)
            logits = cond_logits + self.cfg_scale * (cond_logits - uncond_logits)

        target_vocab = 32000
        if logits.shape[1] < target_vocab:
            padding = torch.full((logits.shape[0], target_vocab - logits.shape[1]), float('-inf'), device=logits.device, dtype=logits.dtype)
            logits = torch.cat([logits, padding], dim=-1)
            
        full_logits = torch.full((logits.shape[0], target_vocab + SPEECH_TOKEN_OFFSET), float('-inf'), device=logits.device, dtype=logits.dtype)
        full_logits[:, SPEECH_TOKEN_OFFSET:SPEECH_TOKEN_OFFSET+target_vocab] = logits
        return full_logits

    def forward(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors=None, inputs_embeds=None):
        if inputs_embeds is None and input_ids is not None:
            is_decoding = (input_ids >= SPEECH_TOKEN_OFFSET).all()
            if is_decoding:
                rel_pos = torch.clamp(positions - SPEECH_START_POS + 1, 0, len(self.precomputed_speech_pos_emb)-1)
                inputs_embeds = self.speech_emb(input_ids - SPEECH_TOKEN_OFFSET) + self.precomputed_speech_pos_emb[rel_pos].to(input_ids.device)
            else:
                inputs_embeds = self.text_emb(input_ids)

        if not self.SAFE_MODE and inputs_embeds is not None and inputs_embeds.shape[-1] == self.dim * 2:
            # Handle CFG by doubling batch and halving hidden size
            c, u = inputs_embeds.split([self.dim, self.dim], dim=-1)
            inputs_embeds = torch.cat([c, u], dim=0)
            positions = torch.cat([positions, positions], dim=0)
            if input_ids is not None:
                input_ids = torch.cat([input_ids, input_ids], dim=0)
        
        return self.tfmr(input_ids=input_ids, positions=positions, kv_caches=kv_caches, 
                        attn_metadata=attn_metadata, intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds)

    def get_language_model(self) -> torch.nn.Module: return self.tfmr