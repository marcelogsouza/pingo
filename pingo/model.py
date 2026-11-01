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

    def forward(self, x: torch.Tensor, capture: dict | None = None,
                chave: str | None = None) -> torch.Tensor:
        """`capture`+`chave` (modo didático, passo ④): guarda o vetor **antes do γ** —
        a lição é que o RMSNorm puro deixa todo vetor com norma √d, e o γ aprende por cima."""
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        puro = (x.float() * norm).to(x.dtype)
        if capture is not None and chave is not None:
            capture[chave] = puro.detach()
        return puro * self.weight


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
                attn_mask: torch.Tensor | None = None,
                capture: dict | None = None) -> torch.Tensor:
        """`attn_mask` bool [B,1,T,T] (True = pode atender) p/ batches com padding.

        Exige `is_causal=False` no SDPA: a causalidade já vem embutida na máscara montada
        pelo `PingoLM` (passar as duas coisas ao SDPA deixa o comportamento ambíguo).

        `capture` (modo didático): dict vazio que sai preenchido com q_lin/q, k_lin/k
        ([B,H,T,D], pré/pós-RoPE) e `w` — as probabilidades de atenção [B,H,T,T] refeitas
        à mão (o SDPA não devolve pesos). Só no prefill sem padding; o caminho normal não
        muda uma linha: o SDPA continua sendo o produtor do `y`.
        """
        b, t, _ = x.shape
        q_lin = self._split(self.q_proj(x))
        k_lin = self._split(self.k_proj(x))
        q = apply_rope(q_lin, cos, sin)
        k = apply_rope(k_lin, cos, sin)
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
        if capture is not None:
            if cache is not None:
                raise ValueError("captura de atenção só no prefill (sem PingoCache)")
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
            permitido = causal if attn_mask is None else (causal & attn_mask[:, 0])
            neg = ~causal.unsqueeze(0).unsqueeze(0) if attn_mask is None else ~permitido.unsqueeze(1)
            logits = logits.masked_fill(neg, torch.finfo(q.dtype).min)
            w = torch.softmax(logits, dim=-1)
            capture.update(q_lin=q_lin.detach(), q=q.detach(), k_lin=k_lin.detach(),
                           k=k.detach(), w=w.detach())
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

    def forward(self, x: torch.Tensor, capture: dict | None = None) -> torch.Tensor:
        """`capture` (modo didático, passo ④): sai com `g/u/s` — o gate, o up e o produto
        silu(gate)·up, o vetor escondido de `intermediate_size` números. A conta é a mesma
        de sempre: o capture só segura os intermediários no caminho."""
        g = self.gate_proj(x)
        u = self.up_proj(x)
        s = F.silu(g) * u
        if capture is not None:
            capture.update(g=g.detach(), u=u.detach(), s=s.detach())
        return self.down_proj(s)


class PingoBlock(nn.Module):
    """Pré-norm: x += Attn(RMSNorm(x)); x += SwiGLU(RMSNorm(x)).

    `capture` (modo didático): o mesmo dict que a atenção recebe sai também com o bloco
    inteiro — `x0/n1p/n1/a/x1/n2p/n2` (fluxo residual e as duas normas; `n*p` = antes do γ)
    e `g/u/s/m/x2` (SwiGLU e a soma final). Sem `capture` o caminho não muda uma linha.
    """

    def __init__(self, config: PingoConfig, layer_idx: int):
        super().__init__()
        self.norm_attn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config, layer_idx)
        self.norm_mlp = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, cache: PingoCache | None = None,
                attn_mask: torch.Tensor | None = None,
                capture: dict | None = None) -> torch.Tensor:
        if capture is not None:
            capture["x0"] = x.detach()
            h = self.norm_attn(x, capture, "n1p")
            capture["n1"] = h.detach()
            a = self.attn(h, cos, sin, cache, attn_mask, capture)
            x = x + a
            capture["a"] = a.detach()
            capture["x1"] = x.detach()
            h = self.norm_mlp(x, capture, "n2p")
            capture["n2"] = h.detach()
            m = self.mlp(h, capture)
            capture["m"] = m.detach()
            capture["x2"] = (x + m).detach()
            return x + m
        x = x + self.attn(self.norm_attn(x), cos, sin, cache, attn_mask, capture)
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
        capture: list[dict] | None = None,
    ) -> CausalLMOutputWithPast:
        """`attention_mask` [B,T] com 1 = token real, 0 = padding (convenção HF).

        Padding é **à direita** (pad no fim): assim as posições dos tokens reais continuam
        0..n-1 e o RoPE não precisa de deslocamento. `labels` usa `-100` nas posições que não
        devem contribuir para a loss (prompt e padding) — default do `cross_entropy`.

        `capture` (modo didático, passos ③–④): lista de um dict vazio por bloco; cada um sai
        com `q_lin/q/k_lin/k` e `w` (as probabilidades de atenção do prefill) e com o bloco
        inteiro — `x0/n1p/n1/a/x1/n2p/n2/g/u/s/m/x2`. Exige B=1, sem padding e sem cache —
        o forward normal (treino/geração) não toca nisso.
        """
        b, t = input_ids.shape
        if capture is not None:
            if b != 1 or attention_mask is not None or cache is not None:
                raise ValueError("capture: só prefill de sequência única, sem padding/cache")
            if len(capture) != self.config.num_hidden_layers:
                raise ValueError(f"capture precisa de {self.config.num_hidden_layers} dicts")
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
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, cache, attn_mask, None if capture is None else capture[i])
        if cache is not None:
            cache.commit()
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits)
