#!/usr/bin/env python
"""Passo 4 — binariza os parquets em shards `.bin` uint16 (estilo nanoGPT).

Pipeline do plano: tokenizer treinado (Passo 3) aplicado aos parquets → shards
`train-NNNN.bin` de ~1M tokens (docs separados por `<|eos|>`) + `val.bin` com o
`split=val` COMPLETO (perplexidade estável). Como temos ~15B tokens e a meta é
~2,5–3B, o train é **amostrado**: arquivos em ordem embaralhada (seed) até
atingir `--train-tokens`. `meta.json` registra contagens e proveniência.

Uso:
  uv run python scripts/prepare_tokens.py                    # completo (~2,75B train + val todo)
  uv run python scripts/prepare_tokens.py --train-tokens 10_000_000 --out data/tokens_smoke  # teste
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tokenizer import iter_docs  # reaproveita o gerador por row-group (RAM limitada)


class BinShards:
    """Escreve um stream de tokens uint16 em shards de exatamente `cap` tokens
    (só o último, em `close()`, pode ser parcial)."""

    def __init__(self, out_dir: Path, prefix: str, cap: int):
        self.out_dir, self.prefix, self.cap = out_dir, prefix, cap
        self.buf = np.empty(cap, dtype=np.uint16)
        self.pos = self.idx = self.total = 0

    def add(self, ids: np.ndarray) -> None:
        i = 0
        while i < len(ids):
            take = min(self.cap - self.pos, len(ids) - i)
            self.buf[self.pos:self.pos + take] = ids[i:i + take]
            self.pos += take
            i += take
            if self.pos == self.cap:
                self._flush()
        self.total += len(ids)

    def _flush(self) -> None:
        if self.pos == 0:
            return
        self.buf[:self.pos].tofile(self.out_dir / f"{self.prefix}-{self.idx:04d}.bin")
        self.idx += 1
        self.pos = 0

    def close(self) -> tuple[int, int]:
        self._flush()
        return self.idx, self.total


def encode_batch(tok, docs: list[str], n_process: int) -> np.ndarray:
    """Tokeniza um lote de docs e concatena tudo com `<|eos|>` após cada doc."""
    with warnings.catch_warnings():  # doc > model_max_length não é erro: nada é truncado aqui
        warnings.simplefilter("ignore")
        batches = tok(docs, add_special_tokens=False, return_attention_mask=False,
                      n_process=n_process)["input_ids"]
    eos = tok.eos_token_id
    flat = [t for ids in batches for t in (*ids, eos)]
    return np.fromiter(flat, dtype=np.uint16, count=len(flat))


def binarize(tok, raw_dir: Path, split: str, writer: BinShards, seed: int,
             token_budget: int, flush_chars: int, n_process: int) -> dict:
    docs, chars, n_docs = [], 0, 0
    t0 = time.time()
    for content in iter_docs(raw_dir, split, seed):
        if writer.total >= token_budget:
            break
        docs.append(content)
        chars += len(content)
        if chars >= flush_chars:
            writer.add(encode_batch(tok, docs, n_process))
            n_docs += len(docs)
            docs, chars = [], 0
            print(f"  {split}: {writer.total/1e6:.1f}M tokens, {n_docs} docs "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    if docs:
        writer.add(encode_batch(tok, docs, n_process))
        n_docs += len(docs)
    return {"docs": n_docs, "tokens": writer.total, "min": round((time.time() - t0) / 60, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "tokens")
    ap.add_argument("--train-tokens", type=int, default=2_750_000_000,
                    help="meta de tokens de treino (temos ~15B no corpus; 0 = tudo)")
    ap.add_argument("--val-tokens", type=int, default=0, help="0 = split val completo")
    ap.add_argument("--shard-tokens", type=int, default=1_000_000)
    ap.add_argument("--flush-chars", type=int, default=10_000_000,
                    help="chars acumulados por chamada de tokenização em lote")
    ap.add_argument("--n-process", type=int, default=8, help="processos na tokenização em lote")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if not (args.tokenizer / "tokenizer.json").exists():
        sys.exit(f"tokenizer não encontrado em {args.tokenizer} — rode train_tokenizer.py antes")
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"binarizando train (meta {args.train_tokens/1e9:.2f}B tokens, seed {args.seed}) ...", flush=True)
    train_w = BinShards(args.out, "train", args.shard_tokens)
    train_stats = binarize(tok, args.raw_dir, "train", train_w,
                           args.seed, args.train_tokens, args.flush_chars, args.n_process)
    n_train_shards, n_train_tokens = train_w.close()

    print("binarizando val (split=val completo) ...", flush=True)
    val_w = BinShards(args.out, "val", max(args.shard_tokens, 50_000_000))
    val_stats = binarize(tok, args.raw_dir, "val", val_w,
                         args.seed, args.val_tokens or 2**62, args.flush_chars, args.n_process)
    n_val_shards, n_val_tokens = val_w.close()

    meta = {
        "tokenizer": str(args.tokenizer.relative_to(ROOT)) if args.tokenizer.is_relative_to(ROOT) else str(args.tokenizer),
        "vocab_size": tok.vocab_size,
        "eos_id": tok.eos_token_id,
        "dtype": "uint16",
        "shard_tokens": args.shard_tokens,
        "train_shards": n_train_shards,
        "train_tokens": n_train_tokens,
        "val_shards": n_val_shards,
        "val_tokens": n_val_tokens,
        "seed": args.seed,
        "train_stats": train_stats,
        "val_stats": val_stats,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("resumo:", json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
