#!/usr/bin/env python
"""Passo 3 — treina o tokenizer BPE byte-level do Pingo (treino-5: por experimento).

Pipeline do plano: amostra ~1 GB do corpus (`split=train`!) → treina BPE byte-level
com `tokenizers` (GPT-2 style regex + byte_fallback) com os especiais configuráveis
(`<|bos|> <|eos|> <|pad|>` default; dá para embutir `<|user|> <|assistant|> <|end|>`
desde a origem) → salva em `tokenizer/<id>/` e avalia chars/token no val comparando
com GPT-2 no mesmo texto. O default histórico é `tokenizer/` (16.384, congelado —
quem passa `--id` escreve em `tokenizer/<id>/` e não toca no de sempre).

Uso:
  uv run python scripts/train_tokenizer.py                    # default: 16k em tokenizer/
  uv run python scripts/train_tokenizer.py --vocab 2000 --sample-chars 50_000_000  # smoke
  uv run python scripts/train_tokenizer.py --id wiki-32k --vocab 32768 \
      --specials '<|bos|>,<|eos|>,<|pad|>,<|user|>,<|assistant|>,<|end|>' \
      --jsonl data/wiki/pt.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections.abc import Iterator
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]

SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>"]
CHAT_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]   # opcionais na origem (treino-5)
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


def iter_docs_jsonl(path: Path) -> Iterator[str]:
    """`content` de um JSONL (contrato da QA do treino-2), em ordem de arquivo."""
    with Path(path).open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                content = json.loads(line)["content"]
                assert isinstance(content, str) and content.strip()
            except (json.JSONDecodeError, KeyError, AssertionError):
                continue
            yield content


def build_sample(raw_dir: Path, sample_dir: Path, budget_chars: int, seed: int,
                 jsonl: Path | None = None) -> dict:
    """Escreve a amostra de treino como arquivos-texto (~128 MB cada), docs separados por \\n."""
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True)

    if jsonl is not None:
        docs = iter_docs_jsonl(jsonl)
    else:
        docs = iter_docs(raw_dir, "train", seed)

    n_docs = n_chars = 0
    part = 0
    fh = (sample_dir / f"sample_{part:03d}.txt").open("w", encoding="utf-8")
    written = 0
    for content in docs:
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
            fh = (sample_dir / f"sample_{part:03d}.txt").open("w", encoding="utf-8")
    fh.close()
    if part == 0 and not (sample_dir / "sample_000.txt").read_bytes():
        sys.exit("amostra vazia — confira o corpus e o filtro split")
    return {"sample_docs": n_docs, "sample_chars": n_chars, "sample_files": part + 1}


def train_tokenizer(sample_files: list[Path], vocab: int, min_freq: int,
                    specials: list[str]) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                       trim_offsets=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab,
                                  special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  min_frequency=min_freq,
                                  show_progress=True,
                                  limit_alphabet=vocab)  # bytes nunca estouram o vocab, mas trava o alfabeto inicial
    tokenizer.train([str(p) for p in sample_files], trainer)
    return tokenizer


def parse_specials(txt: str) -> list[str]:
    """`--specials` aceita lista separada por vírgula/espaço; valida o tripé obrigatório."""
    toks = [t for t in txt.replace(",", " ").split() if t]
    faltando = [t for t in ("<|bos|>", "<|eos|>", "<|pad|>") if t not in toks]
    if faltando:
        sys.exit(f"--specials precisa de <|bos|> <|eos|> <|pad|> (faltam: {', '.join(faltando)})")
    return toks


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default=None,
                    help="nome do experimento → tokenizer/<id>/ (omitido: escreve em tokenizer/)")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="corpus p/ amostra é um JSONL `content` (contrato da QA) em vez dos parquets")
    ap.add_argument("--out", type=Path, default=None, help="override da pasta de saída")
    ap.add_argument("--sample-chars", type=int, default=1_000_000_000)
    ap.add_argument("--vocab", type=int, default=16_384)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--specials", default=",".join(SPECIAL_TOKENS),
                    help="tokens especiais; bos/eos/pad obrigatórios, chat opcional (treino-5)")
    ap.add_argument("--val-chars", type=int, default=5_000_000,
                    help="chars de texto de val usados na avaliação chars/token")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    specials = parse_specials(args.specials)
    if args.out is not None:
        out = args.out
    elif args.id:
        out = ROOT / "tokenizer" / args.id
        if (out / "tokenizer.json").exists():
            sys.exit(f"tokenizer/{args.id} já existe — escolha outro --id (um escritor por saída)")
    else:
        out = ROOT / "tokenizer"
    if args.jsonl is not None and not args.jsonl.is_file():
        sys.exit(f"--jsonl não é arquivo: {args.jsonl}")

    sample_dir = out / "sample"
    t0 = time.time()
    print(f"1/3 montando amostra de treino (~{args.sample_chars/1e9:.1f} GB) ...", flush=True)
    sample_stats = build_sample(args.raw_dir, sample_dir, args.sample_chars, args.seed,
                                jsonl=args.jsonl)
    print("   ", json.dumps(sample_stats), flush=True)

    print(f"2/3 treinando BPE byte-level vocab={args.vocab} especiais={specials} ...", flush=True)
    tok = train_tokenizer(sorted(sample_dir.glob("*.txt")), args.vocab,
                          args.min_frequency, specials)

    print(f"3/3 avaliando chars/token no val (vs GPT-2) ...", flush=True)
    # texto de avaliação: JSONL (amostra embaralhada) ou split=val dos parquets
    eval_chars, chunks = 0, []
    if args.jsonl is not None:
        ordem = list(iter_docs_jsonl(args.jsonl))
        random.Random(args.seed).shuffle(ordem)
        for content in ordem:
            chunks.append(content)
            eval_chars += len(content)
            if eval_chars >= args.val_chars:
                break
    else:
        for content in iter_docs(args.raw_dir, "val", args.seed):
            chunks.append(content)
            eval_chars += len(content)
            if eval_chars >= args.val_chars:
                break
    eval_text = "\n".join(chunks)

    our_ids = tok.encode(eval_text, add_special_tokens=False).ids
    try:  # o comparativo GPT-2 é opcional: sem cache HF/rede, o resto da ficha vale
        from transformers import AutoTokenizer
        gpt2_ids = AutoTokenizer.from_pretrained("gpt2").encode(eval_text)
        gpt2_tokens = len(gpt2_ids)
        gpt2_cpt = round(len(eval_text) / len(gpt2_ids), 3)
        ganho = round(100 * (1 - len(our_ids) / len(gpt2_ids)), 2)
    except Exception:
        gpt2_tokens = gpt2_cpt = ganho = None

    freq = Counter(our_ids)
    top20 = [[tok.id_to_token(t), c] for t, c in freq.most_common(20)]
    stats = {
        "id": args.id or out.name,
        **sample_stats,
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": specials,
        "chat_tokens_embutidos": [t for t in CHAT_TOKENS if t in specials],
        "corpus": str(args.jsonl) if args.jsonl else str(args.raw_dir),
        "eval_split": "val" if args.jsonl is None else "jsonl-embaralhado",
        "eval_chars": len(eval_text),
        "our_tokens": len(our_ids),
        "our_chars_per_token": round(len(eval_text) / len(our_ids), 3),
        "gpt2_tokens": gpt2_tokens,
        "gpt2_chars_per_token": gpt2_cpt,
        "gain_vs_gpt2_pct": ganho,
        "top20_tokens_eval": top20,
        "minutos": round((time.time() - t0) / 60, 2),
    }

    # salvar no formato HF (tokenizer.json + tokenizer_config.json + vocab/merges)
    from transformers import PreTrainedTokenizerFast
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok,
                                     bos_token="<|bos|>", eos_token="<|eos|>",
                                     pad_token="<|pad|>", model_max_length=2048)
    hf_tok.save_pretrained(out)
    stats_path = out / "vocab_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    shutil.rmtree(sample_dir)  # amostra bruta não precisa ficar no disco
    print(json.dumps({k: v for k, v in stats.items() if k != "top20_tokens_eval"},
                     indent=2, ensure_ascii=False))
    print(f"tokenizer salvo em {out} (stats em vocab_stats.json)")


if __name__ == "__main__":
    main()
