"""Arquitetura do Pingo (Passo 5) — decoder-only próprio, estilo moderno simples:

- embedding de tokens + **tied embeddings** (sem embedding de posição)
- pré-norm **RMSNorm** em cada bloco
- **MHA causal** (head_dim 64) via `F.scaled_dot_product_attention`, **RoPE** (θ=10k) em Q/K
- MLP **SwiGLU**
- lineares sem viés; init `normal(0, 0.02)` com escala `1/√(2L)` nas projeções
  residuais (o_proj e down_proj)
- `PingoLM(PreTrainedModel)`: aceita `cache=PingoCache(...)` para geração rápida;
  no treino `cache=None` (atenção cheia sobre a janela de contexto)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel

from .cache import PingoCache
from .config import PingoConfig


class RMSNorm(nn.Module):
    """RMSNorm (pré-norm); estatística em fp32 para estabilidade sob bf16."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


class RoPE(nn.Module):
    """RoPE clássico: freqs = pos · inv_freq; Q/K rotacionados por (cos, sin).

    `inv_freq` é calculado sob demanda por device (sem buffer): buffers
    non-persistent voltam uninitialized do `from_pretrained` do HF v5, e
    persistentes poluiriam o checkpoint — o custo é trivial (arange + pow).
    """

    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = float(theta)
        self._inv_freq_cache: dict[torch.device, torch.Tensor] = {}

    def inv_freq(self, device: torch.device) -> torch.Tensor:
        if device not in self._inv_freq_cache:
            e = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim
            self._inv_freq_cache[device] = self.theta ** (-e)
        return self._inv_freq_cache[device]

    def tables(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids [B,T] → cos/sin [B,T,head_dim] (broadcast p/ heads no apply)."""
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq(position_ids.device)  # [B,T,D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B,T,D]
        return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B,H,T,D]; cos/sin [B,T,D] → unsqueeze(1) broadcast p/ H."""
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


class Attention(nn.Module):
    """MHA causal com suporte opcional a PingoCache (Q/K/V/O sem viés)."""

    def __init__(self, config: PingoConfig, layer_idx: int):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        qkv_dim = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, qkv_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, qkv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, qkv_dim, bias=False)
        self.o_proj = nn.Linear(qkv_dim, config.hidden_size, bias=False)
        self.o_proj._residual_proj = True  # escala 1/√(2L) no init
        self.layer_idx = layer_idx

    def _split(self, proj: torch.Tensor) -> torch.Tensor:
        b, t, _ = proj.shape
        return proj.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,D]

    def forward(self, x, cos, sin, cache: PingoCache | None = None,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """`attn_mask` bool [B,1,T,T] (True = pode atender) p/ batches com padding.

        Exige `is_causal=False` no SDPA: a causalidade já vem embutida na máscara montada
        pelo `PingoLM` (passar as duas coisas ao SDPA deixa o comportamento ambíguo).
        """
        b, t, _ = x.shape
        q = apply_rope(self._split(self.q_proj(x)), cos, sin)
        k = apply_rope(self._split(self.k_proj(x)), cos, sin)
        v = self._split(self.v_proj(x))

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)
            kv_len = k.shape[2]
            if t == kv_len:
                is_causal = True                      # prefill em cache vazio
            elif t == 1:
                is_causal = False                     # decode: query vê todo o passado
            else:
                raise ValueError(
                    "PingoCache só suporta prefill (cache vazio) ou step de 1 token; "
                    f"recebeu t={t} com cache já em {kv_len - t}"
                )
        else:
            is_causal = attn_mask is None  # com máscara explícita, a causalidade vem nela

        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    """MLP SwiGLU: down(silu(gate(x)) * up(x)); down_proj é projeção residual."""

    def __init__(self, config: PingoConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.down_proj._residual_proj = True  # escala 1/√(2L) no init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PingoBlock(nn.Module):
    """Pré-norm: x += Attn(RMSNorm(x)); x += SwiGLU(RMSNorm(x))."""

    def __init__(self, config: PingoConfig, layer_idx: int):
        super().__init__()
        self.norm_attn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config, layer_idx)
        self.norm_mlp = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, cache: PingoCache | None = None,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin, cache, attn_mask)
        return x + self.mlp(self.norm_mlp(x))


class PingoLM(PreTrainedModel):
    config_class = PingoConfig
    _no_split_modules = ["PingoBlock"]
    _tied_weights_keys = {"lm_head.weight": "embed.weight"}  # HF v5: declaração do tie
    main_input_name = "input_ids"

    def __init__(self, config: PingoConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RoPE(config.head_dim, theta=config.rope_theta)
        self.blocks = nn.ModuleList(
            PingoBlock(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied embeddings

        self.post_init()

    # HF chama _init_weights em cada módulo (via post_init → initialize_weights)
    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if getattr(module, "_residual_proj", False):
                module.weight.data.mul_(1.0 / math.sqrt(2 * self.config.num_hidden_layers))
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        cache: PingoCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> CausalLMOutputWithPast:
        """`attention_mask` [B,T] com 1 = token real, 0 = padding (convenção HF).

        Padding é **à direita** (pad no fim): assim as posições dos tokens reais continuam
        0..n-1 e o RoPE não precisa de deslocamento. `labels` usa `-100` nas posições que não
        devem contribuir para a loss (prompt e padding) — default do `cross_entropy`.
        """
        b, t = input_ids.shape
        if position_ids is None:
            if cache is not None:
                position_ids = torch.arange(
                    cache.cur_len, cache.cur_len + t, device=input_ids.device
                ).unsqueeze(0).expand(b, t)
            else:
                position_ids = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)

        attn_mask = None
        if attention_mask is not None:
            if cache is not None:
                raise ValueError("attention_mask não é suportado junto com PingoCache")
            keep = attention_mask.bool()[:, None, None, :]           # [B,1,1,T]
            causal = torch.ones(t, t, dtype=torch.bool,
                                device=input_ids.device).tril()       # [T,T]
            attn_mask = keep & causal                                 # [B,1,T,T]
            # toda linha precisa de pelo menos uma chave (token 0 enxerga a si mesmo)
            attn_mask[:, :, :, 0] = True

        x = self.embed(input_ids)
        cos, sin = self.rope.tables(position_ids)
        for blk in self.blocks:
            x = blk(x, cos, sin, cache, attn_mask)
        if cache is not None:
            cache.commit()
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits)
