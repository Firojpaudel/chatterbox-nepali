from typing import Iterable, Mapping, Optional, Sequence, Union, Tuple, Any
import torch
import torch.nn as nn
import random
import os
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

def create_triangular_matrix(m, n, dtype=torch.float32):
    row_indices = torch.arange(m).unsqueeze(1)
    col_indices = torch.arange(n).unsqueeze(0)
    return (col_indices <= row_indices).to(dtype)


class T3ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"conditionals": 1}


class T3MultiModalDummyInputsBuilder(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "[START]Hello![STOP]"

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        return { "conditionals": [torch.zeros(CONDITIONING_SIZE, 1024, dtype=torch.float16)] * mm_counts["conditionals"] }


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

        final_prompt_ids = [
            PREFILL_COND_START_TOKEN,
            *([prompt_ids[0]] * (CONDITIONING_SIZE - 2)),
            PREFILL_COND_END_TOKEN,
            *prompt_ids,
            PREFILL_END_TOKEN,
        ]

        conditionals = mm_data.get("conditionals", None)
        assert conditionals is not None and len(conditionals) == 1
        
        dev = conditionals[0].device
        dtype = conditionals[0].dtype
        new_conditionals = torch.cat([
            conditionals[0],
            create_triangular_matrix(len(prompt_ids), conditionals[0].shape[1], dtype=dtype).to(dev),
            torch.zeros(1, conditionals[0].shape[1], dtype=dtype).to(dev),
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
        vllm_config.model_config.hf_config.hidden_size = 1024
        self.vllm_config = vllm_config
        self.cfg: ModelConfig = vllm_config.model_config

        self.tfmr = LlamaModel(vllm_config=vllm_config, prefix=prefix + ".tfmr")
        
        is_multilingual = getattr(self.cfg.hf_config, 'is_multilingual', True)
        self.t3conf = T3Config.multilingual() if is_multilingual else T3Config()
        
        self.dim = self.t3conf.n_channels
        self.cond_enc = T3CondEnc(self.t3conf)
        self.text_emb = nn.Embedding(self.t3conf.text_tokens_dict_size, self.dim)
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
        self.cfg_scale = float(os.environ.get("CHATTERBOX_CFG_SCALE", "0.8"))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params: set[str] = set()
        state_dicts = {}
        hf_llama_weights = {}
        
        valid_llama_suffixes = {
            "embed_tokens.weight", "norm.weight", "layers.",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
            "input_layernorm.weight", "post_attention_layernorm.weight"
        }

        for raw_name, weight in weights:
            name = raw_name
            if "tfmr." in name:
                subname = name[name.find("tfmr.")+5:]
                if any(suffix in subname for suffix in valid_llama_suffixes):
                    if subname == "embed_tokens.weight" and weight.shape[0] < 32000:
                        padding = torch.zeros((32000 - weight.shape[0], weight.shape[1]), dtype=weight.dtype, device=weight.device)
                        weight = torch.cat([weight, padding], dim=0)
                    hf_llama_weights[subname] = weight
                continue
            
            loaded_params.add(raw_name)
            if '.' not in name: continue
            attr, subname = name.split('.', 1)
            state_dict = state_dicts.get(attr, {})
            state_dict[subname] = weight
            state_dicts[attr] = state_dict

        for attr, state_dict in state_dicts.items():
            if hasattr(self, attr): getattr(self, attr).load_state_dict(state_dict, strict=False)

        llama_loaded = self.tfmr.load_weights(hf_llama_weights.items())
        loaded_params.update('tfmr.' + i for i in llama_loaded)

        dev = self.speech_emb.weight.device
        dtype = self.speech_emb.weight.dtype
        self.precomputed_text_pos_emb = self.text_pos_emb.get_fixed_embedding(torch.arange(self.t3conf.max_text_tokens + 2, device=dev))[0].to(dtype)
        self.precomputed_speech_pos_emb = self.speech_pos_emb.get_fixed_embedding(torch.arange(self.t3conf.max_speech_tokens + 4, device=dev))[0].to(dtype)
        return loaded_params

    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        conditionals = kwargs.get("conditionals", [])
        if conditionals is None or (isinstance(conditionals, list) and len(conditionals) == 0):
            return None
        return [batch[0] for batch in conditionals]

    def split_prefill_decode(self, input_ids: torch.Tensor, multimodal_embeddings: list[torch.Tensor]) -> list[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if len(input_ids) == 0: return []
        remaining_mm = torch.cat(multimodal_embeddings, dim=0)
        output = []
        buffer = []
        in_prefill = input_ids[0] < SPEECH_TOKEN_OFFSET
        for iid in input_ids:
            is_p = iid < SPEECH_TOKEN_OFFSET
            if (in_prefill != is_p) or (iid == PREFILL_COND_START_TOKEN):
                if buffer:
                    buf_t = torch.tensor(buffer, device=input_ids.device)
                    if in_prefill:
                        sz = min(len(buffer), len(remaining_mm))
                        mme, remaining_mm = remaining_mm.split([sz, len(remaining_mm) - sz], dim=0)
                        output.append((buf_t, mme))
                    else:
                        output.append((buf_t, None))
                buffer = [iid]
                in_prefill = is_p
            else:
                buffer.append(iid)
        if buffer:
            buf_t = torch.tensor(buffer, device=input_ids.device)
            if in_prefill:
                sz = min(len(buffer), len(remaining_mm))
                mme, _ = remaining_mm.split([sz, len(remaining_mm) - sz], dim=0)
                output.append((buf_t, mme))
            else:
                output.append((buf_t, None))
        return output

    def get_input_embeddings(self, input_ids: torch.Tensor, multimodal_embeddings: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        dtype = self.speech_emb.weight.dtype
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            embeds = self.speech_emb(input_ids - SPEECH_TOKEN_OFFSET)
            return torch.cat([embeds, embeds], dim=1).contiguous()
        
        mm_casted = [mm.to(dtype) for mm in multimodal_embeddings]
            
        out = []
        for ids, mme in self.split_prefill_decode(input_ids, mm_casted):
            if mme is None:
                embeds = self.speech_emb(ids - SPEECH_TOKEN_OFFSET)
                out.append(torch.cat([embeds, embeds], dim=1))
                continue
            
            if ids[0] == PREFILL_COND_START_TOKEN and ids[-1] == PREFILL_END_TOKEN:
                text_ids = ids[CONDITIONING_SIZE:-1]
                text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)].to(ids.device)
                start_of_speech_token = torch.tensor([self.t3conf.start_speech_token], device=ids.device)
                start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0] + self.precomputed_speech_pos_emb[0:1].to(ids.device)
                conditioning_emb = mme[0:CONDITIONING_SIZE]
                cond_embeds = torch.cat([conditioning_emb, text_emb, start_of_speech_emb], dim=0)
                uncond_embeds = torch.cat([conditioning_emb, torch.zeros_like(text_emb), start_of_speech_emb], dim=0)
                out.append(torch.cat([cond_embeds, uncond_embeds], dim=1))
            elif ids[0] == PREFILL_COND_START_TOKEN:
                text_ids = ids[CONDITIONING_SIZE:]
                text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)].to(ids.device)
                conditioning_emb = mme[0:min(CONDITIONING_SIZE, len(mme))]
                cond_embeds = torch.cat([conditioning_emb, text_emb], dim=0)
                uncond_embeds = torch.cat([conditioning_emb, torch.zeros_like(text_emb)], dim=0)
                out.append(torch.cat([cond_embeds, uncond_embeds], dim=1))
            elif ids[-1] == PREFILL_END_TOKEN:
                indices = torch.where(ids == PREFILL_COND_END_TOKEN)[0]
                if len(indices) > 0:
                    text_ids = ids[indices[0]+1:-1]
                    text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[0:len(text_ids)].to(ids.device)
                    start_of_speech_token = torch.tensor([self.t3conf.start_speech_token], device=ids.device)
                    start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0] + self.precomputed_speech_pos_emb[0:1].to(ids.device)
                    conditioning_emb = mme[:indices[0]+1]
                    cond_embeds = torch.cat([conditioning_emb, text_emb, start_of_speech_emb], dim=0)
                    uncond_embeds = torch.cat([conditioning_emb, torch.zeros_like(text_emb), start_of_speech_emb], dim=0)
                    out.append(torch.cat([cond_embeds, uncond_embeds], dim=1))
                else:
                    text_ids = ids[:-1]
                    text_pos = torch.sum(mme[0:len(text_ids)], dim=1).long() - 1
                    text_emb = self.text_emb(text_ids) + self.precomputed_text_pos_emb[text_pos].to(ids.device)
                    start_of_speech_token = torch.tensor([self.t3conf.start_speech_token], device=ids.device)
                    start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0] + self.precomputed_speech_pos_emb[0:1].to(ids.device)
                    cond_embeds = torch.cat([text_emb, start_of_speech_emb], dim=0)
                    uncond_embeds = torch.cat([torch.zeros_like(text_emb), start_of_speech_emb], dim=0)
                    out.append(torch.cat([cond_embeds, uncond_embeds], dim=1))
            else:
                out.append(torch.zeros(len(ids), self.dim * 2, dtype=dtype, device=ids.device))
        return torch.cat(out, dim=0).contiguous()

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        cond_hs, uncond_hs = hidden_states.split([self.dim, self.dim], dim=1)
        cond_logits = self.logits_processor(self.speech_head, cond_hs, sampling_metadata)
        uncond_logits = self.logits_processor(self.speech_head, uncond_hs, sampling_metadata)
        logits = cond_logits + self.cfg_scale * (cond_logits - uncond_logits)
        padding = torch.full((logits.shape[0], SPEECH_TOKEN_OFFSET), float('-inf'), dtype=logits.dtype, device=logits.device)
        return torch.cat([padding, logits], dim=1)

    def forward(self, input_ids: Optional[torch.Tensor] = None, positions: Optional[torch.Tensor] = None, 
                intermediate_tensors: Optional[IntermediateTensors] = None, inputs_embeds: Optional[torch.Tensor] = None, **kwargs):
        if inputs_embeds is None:
            mm_data = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, mm_data)

        cond_embeds, uncond_embeds = inputs_embeds.split([self.dim, self.dim], dim=1)
        
        # Double positions and embeds, ensuring strict contiguity for CUDA kernels
        doubled_positions = torch.cat([positions, positions], dim=0).contiguous()
        doubled_embeds = torch.cat([cond_embeds, uncond_embeds], dim=0).contiguous()

        hidden_states = self.tfmr(
            input_ids=None,
            positions=doubled_positions,
            intermediate_tensors=intermediate_tensors, 
            inputs_embeds=doubled_embeds
        )
        
        h1, h2 = hidden_states.split([len(cond_embeds), len(uncond_embeds)], dim=0)
        return torch.cat([h1, h2], dim=1).contiguous()

    def get_language_model(self) -> torch.nn.Module: return self.tfmr