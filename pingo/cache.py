"""PingoCache — KV cache com buffers pré-alocados por camada (Passo 5).

Um par de tensors `[B, H, max_seq, D]` (K e V) por camada, alocado no primeiro
`update()` (quando sabemos device/dtype/batch). A escrita é incremental na
fatia `[cur_len : cur_len+T]`; a leitura devolve a visão `[:, :, :cur_len+T]`.

Uso (feito por `PingoLM.forward`): cada attention chama `update(layer_idx, k, v)`
durante o forward e, ao final de todas as camadas, `commit(T)` avança a posição.

Restrições deliberadas (geração é inferência, sem grad):
- uso sem `torch.no_grad()` não propaga gradiente pelos valores cacheados;
- após o prefill, só aceita step de 1 token por vez (decode); prefill multi-token
  só com cache vazio.
"""

from __future__ import annotations

import torch


class PingoCache:
    def __init__(self, config, batch_size: int | None = None, max_seq: int | None = None):
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.max_seq = max_seq or config.max_position_embeddings
        self.batch_size = batch_size
        self._k: list[torch.Tensor | None] = [None] * self.n_layers
        self._v: list[torch.Tensor | None] = [None] * self.n_layers
        self._cur_len = 0
        self._pending = 0  # tokens escritos neste forward, aguardando commit()

    @property
    def cur_len(self) -> int:
        """Posições já ocupadas no cache (antes do próximo update)."""
        return self._cur_len

    def reset(self) -> None:
        for i in range(self.n_layers):
            if self._k[i] is not None:
                self._k[i] = None
                self._v[i] = None
        self._cur_len = 0
        self._pending = 0

    def _ensure(self, layer_idx: int, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._k[layer_idx] is None:
            buf = (k.shape[0], self.n_heads, self.max_seq, self.head_dim)
            self._k[layer_idx] = torch.zeros(buf, dtype=k.dtype, device=k.device)
            self._v[layer_idx] = torch.zeros(buf, dtype=k.dtype, device=k.device)
            if self.batch_size is None:
                self.batch_size = k.shape[0]
            elif self.batch_size != k.shape[0]:
                raise ValueError(f"batch {k.shape[0]} ≠ batch alocado {self.batch_size}")
        return self._k[layer_idx], self._v[layer_idx]

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Escreve K/V novos na posição atual e devolve as visões completas p/ atenção.

        Todas as camadas de um mesmo forward escrevem na MESMA posição `cur_len`
        (o avanço só acontece em `commit()`, chamado pelo modelo no fim do forward).
        """
        t = k.shape[2]
        start = self._cur_len
        if start + t > self.max_seq:
            raise ValueError(f"cache cheio: {start + t} tokens > max_seq {self.max_seq}")
        buf_k, buf_v = self._ensure(layer_idx, k)
        buf_k[:, :, start:start + t] = k
        buf_v[:, :, start:start + t] = v
        self._pending = t
        return buf_k[:, :, :start + t], buf_v[:, :, :start + t]

    def commit(self) -> None:
        """Avança a posição após um forward completo (chamado por PingoLM.forward)."""
        self._cur_len += self._pending
        self._pending = 0
