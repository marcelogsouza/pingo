"""Loader memmap para os shards `.bin` uint16 do Passo 4 (estilo nanoGPT).

Shards esperados em `data/tokens/`: `train-NNNN.bin`, `val-NNNN.bin` + `meta.json`
(escritos por `scripts/prepare_tokens.py`). Batches têm forma `[B, ctx]` com o alvo
deslocado de 1 (`y[t] == x[t+1]`), prontos p/ cross-entropy de próximo token.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DTYPE = np.uint16


@dataclass
class TokenData:
    train: list[np.memmap]
    val: list[np.memmap]
    meta: dict = field(default_factory=dict)

    def get_batch(self, split: str, batch_size: int, ctx: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        shards = self.train if split == "train" else self.val
        return get_batch(shards, batch_size, ctx, rng)


def load_tokens(tokens_dir: Path) -> TokenData:
    tokens_dir = Path(tokens_dir)
    meta_path = tokens_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} ausente — rode scripts/prepare_tokens.py antes")
    train = [np.memmap(p, dtype=DTYPE, mode="r") for p in sorted(tokens_dir.glob("train-*.bin"))]
    val = [np.memmap(p, dtype=DTYPE, mode="r") for p in sorted(tokens_dir.glob("val-*.bin"))]
    if not train or not val:
        raise FileNotFoundError(f"shards train-*.bin/val-*.bin ausentes em {tokens_dir}")
    meta = json.loads(meta_path.read_text())
    return TokenData(train, val, meta)


def get_batch(shards: list[np.memmap], batch_size: int, ctx: int,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Amostra `batch_size` janelas de `ctx` tokens uniformemente entre os shards."""
    usable = [s for s in shards if len(s) > ctx + 1]
    if not usable:
        raise ValueError(f"nenhum shard com mais de {ctx + 1} tokens")
    lens = np.array([len(s) for s in usable])
    ends = lens.cumsum()
    x = np.empty((batch_size, ctx), dtype=np.int64)
    y = np.empty((batch_size, ctx), dtype=np.int64)
    for b in range(batch_size):
        r = rng.integers(int(ends[-1]))
        i = int(np.searchsorted(ends, r, side="right"))
        off = int(rng.integers(0, lens[i] - ctx))  # cabe off+ctx+1 no shard
        s = usable[i]
        x[b] = s[off:off + ctx]
        y[b] = s[off + 1:off + 1 + ctx]
    return x, y
