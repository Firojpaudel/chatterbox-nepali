from typing import Optional

T3_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_attention_heads": 32,
    "num_hidden_layers": 30,
    "num_key_value_heads": 32,
    "rms_norm_eps": 1e-05,
    "vocab_size": 32000,
    "rope_theta": 10000.0,
    "max_position_embeddings": 1000,
}

class T3Config:
    def __init__(self, **kwargs):
        # Core Transformer dimensions
        self.n_channels = kwargs.get("hidden_size", 2048)
        self.intermediate_size = kwargs.get("intermediate_size", 8192)
        self.n_heads = kwargs.get("num_attention_heads", 32)
        self.n_layers = kwargs.get("num_hidden_layers", 30)
        self.n_kv_heads = kwargs.get("num_key_value_heads", 32)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-05)
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 1000)
        
        # Conditioning Encoder required fields
        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.emotion_adv = True
        self.use_perceiver_resampler = True
        self.n_conditioning_tokens = 34
        self.conditioning_dim = 1024
        self.n_audio_tokens = 1
        self.hf_config = None

    @classmethod
    def multilingual(cls):
        return cls()

    @classmethod
    def english_only(cls):
        return cls()

    @property
    def is_multilingual(self) -> bool:
        return True
