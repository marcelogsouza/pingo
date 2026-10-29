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
        sorted_x, order = x.sort(dim=-1, descending=True)
        keep = sorted_x.cumsum(dim=-1).sub(sorted_x) < top_p  # sempre mantém o melhor
        mask = torch.full_like(keep, True).scatter_(1, order, keep)
        x = x.masked_fill(~mask, float("-inf"))
    return torch.multinomial(x.softmax(dim=-1), num_samples=1)


@torch.no_grad()
def generate(model, prompt_ids: list[int] | torch.Tensor, max_new_tokens: int = 128,
             temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
             stop_token: int | None = None, seed: int | None = None) -> torch.Tensor:
    """Retorna `[1, prompt+gerado]` (para no máximo `max_new_tokens` novos).

    Para no contexto do modelo (cache pré-alocado com `max_position_embeddings`).
    """
    if isinstance(prompt_ids, list):
        prompt_ids = torch.tensor([prompt_ids], dtype=torch.long)
    elif prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.shape[0] != 1:
        raise ValueError("generate() é batch=1 (batching fica p/ depois)")
    prompt_ids = prompt_ids.to(next(model.parameters()).device)
    if prompt_ids.shape[1] >= model.config.max_position_embeddings:
        raise ValueError("prompt não cabe no contexto do modelo")

    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    cache = PingoCache(model.config, batch_size=1)
    ids = prompt_ids
    logits = model(ids, cache=cache).logits[:, -1]

    for _ in range(max_new_tokens):
        if cache.cur_len >= model.config.max_position_embeddings:
            break  # preencheu o contexto (prefill já entrou no cache)
        nxt = sample_next(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        ids = torch.cat((ids, nxt), dim=-1)
        if stop_token is not None and int(nxt) == stop_token:
            break
        logits = model(nxt, cache=cache).logits[:, -1]

    return ids
