"""Geração própria com KV cache (Passo 5): greedy / temperature / top-k / top-p.

`generate()` prefilla o prompt com um `PingoCache` e depois decodifica 1 token
por vez — cada step custa O(ctx) em vez de O(ctx²) do re-forward completo.
"""

from __future__ import annotations

import torch

from .cache import PingoCache


def sample_next(logits: torch.Tensor, temperature: float = 1.0,
                top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Amostra 1 id por linha de `logits` [B,V]. temperature<=0 → greedy (argmax)."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    x = logits.float() / temperature
    if top_k:
        k = min(top_k, x.shape[-1])
        kth = x.topk(k, dim=-1).values[:, -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    if top_p and top_p < 1.0:
        # nucleus de verdade: o acumulado é sobre as PROBABILIDADES (cumsum em cima de
        # logits era o jeito antigo, e cortava tudo quando o logit era positivo)
        sorted_x, order = x.sort(dim=-1, descending=True)
        p_ord = sorted_x.softmax(dim=-1)
        keep = p_ord.cumsum(dim=-1).sub(p_ord) < top_p  # sempre mantém o melhor
        mask = torch.full_like(keep, True).scatter_(1, order, keep)
        x = x.masked_fill(~mask, float("-inf"))
    return torch.multinomial(x.softmax(dim=-1), num_samples=1)


def apply_repetition_penalty(logits: torch.Tensor, seen_ids: torch.Tensor,
                             penalty: float) -> torch.Tensor:
    """Penaliza logits de tokens já presentes em `seen_ids` (semântica do HF).

    score > 0 → divide por `penalty`; score < 0 → multiplica. `penalty=1.0` é no-op.
    É o antídoto do loop de repetição do greedy em modelo pequeno.
    """
    if penalty == 1.0:
        return logits
    out = logits.clone()
    sel = out[0, seen_ids]
    out[0, seen_ids] = torch.where(sel > 0, sel / penalty, sel * penalty)
    return out


def _prompt_tensor(prompt_ids: list[int] | torch.Tensor, model) -> torch.Tensor:
    """Normaliza o prompt para `[1, n]` no device do modelo (batch=1 por ora)."""
    if isinstance(prompt_ids, list):
        prompt_ids = torch.tensor([prompt_ids], dtype=torch.long)
    elif prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.shape[0] != 1:
        raise ValueError("generate() é batch=1 (batching fica p/ depois)")
    prompt_ids = prompt_ids.to(next(model.parameters()).device)
    if prompt_ids.shape[1] >= model.config.max_position_embeddings:
        raise ValueError("prompt não cabe no contexto do modelo")
    return prompt_ids


@torch.no_grad()
def generate_stream(model, prompt_ids: list[int] | torch.Tensor, max_new_tokens: int = 128,
                    temperature: float = 1.0, top_k: int | None = None,
                    top_p: float | None = None, stop_token: int | None = None,
                    seed: int | None = None, repetition_penalty: float = 1.0):
    """**Streaming**: igual a `generate`, mas emite cada token novo (`int`) na hora.

    O token de parada é emitido e fecha o gerador — exatamente como em `generate`, cuja
    saída já inclui o stop. Serve para a WebUI (SSE): o chamador decodifica um token por
    vez em vez de esperar a resposta inteira.
    """
    prompt_ids = _prompt_tensor(prompt_ids, model)
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    cache = PingoCache(model.config, batch_size=1)
    ids = prompt_ids
    logits = model(ids, cache=cache).logits[:, -1]

    for _ in range(max_new_tokens):
        if cache.cur_len >= model.config.max_position_embeddings:
            break  # preencheu o contexto (prefill já entrou no cache)
        step_logits = logits
        if repetition_penalty != 1.0:
            step_logits = apply_repetition_penalty(logits, torch.unique(ids[0]), repetition_penalty)
        nxt = sample_next(step_logits, temperature=temperature, top_k=top_k, top_p=top_p)
        ids = torch.cat((ids, nxt), dim=-1)
        yield int(nxt)
        if stop_token is not None and int(nxt) == stop_token:
            break
        logits = model(nxt, cache=cache).logits[:, -1]


@torch.no_grad()
def generate(model, prompt_ids: list[int] | torch.Tensor, max_new_tokens: int = 128,
             temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
             stop_token: int | None = None, seed: int | None = None,
             repetition_penalty: float = 1.0) -> torch.Tensor:
    """Retorna `[1, prompt+gerado]` (para no máximo `max_new_tokens` novos).

    Para no contexto do modelo (cache pré-alocado com `max_position_embeddings`).
    `repetition_penalty > 1` pune tokens já usados no prompt+geração (ver
    `apply_repetition_penalty`); é o que evita o loop do greedy em 120M.

    É um invólucro de `generate_stream`: mesma amostragem, mesma ordem de RNG.
    """
    prompt_ids = _prompt_tensor(prompt_ids, model)
    novos = list(generate_stream(model, prompt_ids, max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_k=top_k, top_p=top_p,
                                 stop_token=stop_token, seed=seed,
                                 repetition_penalty=repetition_penalty))
    if not novos:
        return prompt_ids
    extra = torch.tensor([novos], dtype=torch.long, device=prompt_ids.device)
    return torch.cat((prompt_ids, extra), dim=-1)
