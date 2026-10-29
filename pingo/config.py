"""PingoConfig — hiperparâmetros do modelo (Passo 5).

Herda de `PretrainedConfig` só como scaffolding do HF (`save_pretrained` /
`from_pretrained`); a arquitetura em si está em `pingo/model.py`.

Default = **Pingo-120M**: 12 camadas, d_model 896, 14 heads (head_dim 64),
SwiGLU 2304, ctx 2048, vocab 16.384, RoPE θ=10k, embeddings tied.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class PingoConfig(PretrainedConfig):
    model_type = "pingo"

    def __init__(
        self,
        vocab_size: int = 16_384,
        hidden_size: int = 896,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 14,
        head_dim: int | None = None,          # default: hidden_size // heads
        intermediate_size: int = 2_304,       # SwiGLU (≈ 8/3 · d_model)
        max_position_embeddings: int = 2_048,
        rope_theta: float = 10_000.0,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 2,
        **kwargs,
    ):
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} não é divisível por {num_attention_heads} heads"
            )
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range

        kwargs.setdefault("tie_word_embeddings", True)  # embeddings tied
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
