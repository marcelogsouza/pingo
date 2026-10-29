#!/usr/bin/env python
"""Passo 3 — treina o tokenizer BPE byte-level (vocab 16k) do Pingo.

Pipeline do plano: amostra ~1 GB dos parquets (`split=train`!) → treina BPE
byte-level com `tokenizers` (GPT-2 style regex + byte_fallback) com os especiais
`<|bos|> <|eos|> <|pad|>` → salva em `tokenizer/` e avalia taxa chars/token no
`split=val` comparando com o tokenizer GPT-2 no mesmo texto.

Uso:
  uv run python scripts/train_tokenizer.py                    # run completo (16k, ~1 GB)
  uv run python scripts/train_tokenizer.py --vocab 2000 --sample-chars 50_000_000  # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections.abc import Iterator
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]

SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>"]
SAMPLE_FILE_CHARS = 128_000_000  # tamanho de cada arquivo-texto da amostra de treino


def iter_docs(raw_dir: Path, split: str, seed: int) -> Iterator[str]:
    """Gera `content` dos parquets em ordem de arquivo embaralhada, sem carregar tudo na RAM."""
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        sys.exit(f"nenhum parquet em {raw_dir} — rode process_data.py antes")
    random.Random(seed).shuffle(files)
    for path in files:
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            tbl = pf.read_row_group(i, columns=["content", "split"])
            mask = [s == split for s in tbl.column("split").to_pylist()]
            for content, keep in zip(tbl.column("content").to_pylist(), mask):
                if keep:
                    yield content


def build_sample(raw_dir: Path, sample_dir: Path, budget_chars: int, seed: int) -> dict:
    """Escreve a amostra de treino como arquivos-texto (~128 MB cada), docs separados por \\n."""
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True)

    n_docs = n_chars = 0
    part = 0
    fh = (sample_dir / f"sample_{part:03d}.txt").open("w", encoding="utf-8")
    written = 0
    for content in iter_docs(raw_dir, "train", seed):
        if n_chars >= budget_chars:
            break
        fh.write(content)
        fh.write("\n")
        n_docs += 1
        n_chars += len(content) + 1
        written += len(content) + 1
        if written >= SAMPLE_FILE_CHARS:
            fh.close()
            part += 1
            written = 0
            fh = (sample_dir / f"sample_{part:03d}.txt").open("w", encoding="utf-8")
    fh.close()
    if part == 0 and not (sample_dir / "sample_000.txt").read_bytes():
        sys.exit("amostra vazia — confira os parquets e o filtro split")
    return {"sample_docs": n_docs, "sample_chars": n_chars, "sample_files": part + 1}


def train_tokenizer(sample_files: list[Path], vocab: int, min_freq: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                       trim_offsets=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab,
                                  special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  min_frequency=min_freq,
                                  show_progress=True,
                                  limit_alphabet=vocab)  # bytes nunca estouram o vocab, mas trava o alfabeto inicial
    tokenizer.train([str(p) for p in sample_files], trainer)
    return tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "tokenizer")
    ap.add_argument("--sample-chars", type=int, default=1_000_000_000)
    ap.add_argument("--vocab", type=int, default=16_384)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--eval-chars", type=int, default=5_000_000,
                    help="chars do split=val usados na avaliação chars/token")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    sample_dir = args.out / "sample"
    print(f"1/3 montando amostra de treino (~{args.sample_chars/1e9:.1f} GB, split=train) ...", flush=True)
    sample_stats = build_sample(args.raw_dir, sample_dir, args.sample_chars, args.seed)
    print("   ", json.dumps(sample_stats), flush=True)

    print(f"2/3 treinando BPE byte-level vocab={args.vocab} ...", flush=True)
    tok = train_tokenizer(sorted(sample_dir.glob("*.txt")), args.vocab, args.min_frequency)

    print(f"3/3 avaliando chars/token no split=val (vs GPT-2) ...", flush=True)
    eval_chars = 0
    chunks: list[str] = []
    for content in iter_docs(args.raw_dir, "val", args.seed):
        chunks.append(content)
        eval_chars += len(content)
        if eval_chars >= args.eval_chars:
            break
    eval_text = "\n".join(chunks)

    our_ids = tok.encode(eval_text, add_special_tokens=False).ids
    from transformers import AutoTokenizer
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    gpt2_ids = gpt2.encode(eval_text)

    freq = Counter(our_ids)
    top20 = [[tok.id_to_token(t), c] for t, c in freq.most_common(20)]
    stats = {
        **sample_stats,
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "eval_split": "val",
        "eval_chars": len(eval_text),
        "our_tokens": len(our_ids),
        "our_chars_per_token": round(len(eval_text) / len(our_ids), 3),
        "gpt2_tokens": len(gpt2_ids),
        "gpt2_chars_per_token": round(len(eval_text) / len(gpt2_ids), 3),
        "gain_vs_gpt2_pct": round(100 * (1 - len(our_ids) / len(gpt2_ids)), 2),
        "top20_tokens_eval": top20,
    }

    # salvar no formato HF (tokenizer.json + tokenizer_config.json + vocab/merges)
    from transformers import PreTrainedTokenizerFast
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok,
                                     bos_token=SPECIAL_TOKENS[0], eos_token=SPECIAL_TOKENS[1],
                                     pad_token=SPECIAL_TOKENS[2], model_max_length=2048)
    hf_tok.save_pretrained(args.out)
    stats_path = args.out / "vocab_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    shutil.rmtree(sample_dir)  # amostra bruta não precisa ficar no disco
    print(json.dumps({k: v for k, v in stats.items() if k != "top20_tokens_eval"}, indent=2, ensure_ascii=False))
    print(f"tokenizer salvo em {args.out} (stats em vocab_stats.json)")


if __name__ == "__main__":
    main()
