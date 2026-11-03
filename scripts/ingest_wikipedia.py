#!/usr/bin/env python
"""Ingest Wikipedia (Passo 14b) — traz um dataset HF de wiki pt para o contrato do /treino.

Fluxo: `datasets.load_dataset` (ou parquets do Hub) → heurísticas do `process_data.py`
(comprimento, linhas curtas, símbolos, dedup blake2b) + split val/train pelo MESMO hash →
`data/wiki/<id>.jsonl` (lado TRAIN) e `<id>-val.jsonl` (o val) com `content` — o que a QA e o
`prepare_tokens --input jsonl` esperam. O JSONL principal JÁ EXCLUI os docs do val: quem
binarizar re-fatiar esse arquivo por hash fica com val vazio (aceite do passo, verificado). A wiki já vem limpa de HTML, mas o embrulho de scraper (título
repetido, "Editar", seções órfãs) é filtrado aqui nas pontas, não dentro do texto.

Uso:
  uv run python scripts/ingest_wikipedia.py                      # TucanoBR/wikipedia-PT → data/wiki/pt.jsonl
  uv run python scripts/ingest_wikipedia.py --limit 5000 --out data/wiki/pt-smoke.jsonl   # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_data import SHORT_LINE_CHARS, SYMBOLS, norm_key  # as MESMAS réguas do filtro OSCAR

MIN_CHARS = 300
MAX_CHARS = 100_000
SHORT_LINE_RATIO = 0.6
SYMBOL_RATIO = 0.08
VAL_PCT = 2          # % de docs p/ val (por hash — o mesmo do OSCAR)


def passes(content: str, short_line_ratio: float = SHORT_LINE_RATIO) -> tuple[bool, str]:
    n = len(content)
    if n < MIN_CHARS or n > MAX_CHARS:
        return False, "len"
    linhas = content.split("\n")
    if sum(1 for ln in linhas if len(ln.strip()) < SHORT_LINE_CHARS) > short_line_ratio * len(linhas):
        return False, "short_lines"
    if sum(content.count(c) for c in SYMBOLS) > SYMBOL_RATIO * n:
        return False, "symbols"
    return True, ""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="TucanoBR/wikipedia-PT")
    ap.add_argument("--campo", default="text", help="campo de texto no HF (vira `content`)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "wiki" / "pt.jsonl")
    ap.add_argument("--out-val", type=Path, default=None, help="default: <out>-val.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="só os N primeiros (smoke)")
    ap.add_argument("--short-line-ratio", type=float, default=0.9,
                    help="0.9 p/ wiki (títulos de seção são linhas curtas legítimas; 0.6 do "
                         "OSCAR mataria ~64% do corpus — medido 2026-11-03)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    out, out_val = args.out, args.out_val or args.out.with_name(args.out.stem + "-val.jsonl")
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} já existe — use --overwrite (um escritor por saída)")
    out.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    print(f"carregando {args.dataset} ...", flush=True)
    ds = load_dataset(args.dataset, split="train")

    t0 = time.time()
    n = aprovados = duplicadas = n_val = 0
    descarte: dict[str, int] = {}
    vistos: set[bytes] = set()
    fv = out_val.open("w", encoding="utf-8")
    ft = out.open("w", encoding="utf-8")
    for rec in ds:
        if args.limit and n >= args.limit:
            break
        n += 1
        content = (rec.get(args.campo) or "").strip()
        ok, motivo = passes(content, args.short_line_ratio)
        if not ok:
            descarte[motivo] = descarte.get(motivo, 0) + 1
            continue
        key = norm_key(content)
        if key in vistos:
            duplicadas += 1
            continue
        vistos.add(key)
        aprovados += 1
        lado = "val" if int.from_bytes(key, "big") % 100 < VAL_PCT else "train"
        n_val += lado == "val"
        (fv if lado == "val" else ft).write(
            json.dumps({"content": content}, ensure_ascii=False) + "\n")
        if aprovados % 100_000 == 0:
            print(f"  {aprovados:,} aprovados de {n:,} ({time.time()-t0:.0f}s)", flush=True)
    fv.close(); ft.close()

    ficha = {
        "fonte": args.dataset, "campo_origem": args.campo, "docs_vistos": n,
        "aprovados": aprovados, "docs_train": aprovados - n_val, "docs_val": n_val,
        "duplicatas": duplicadas, "descarte": descarte,
        "val_pct": VAL_PCT, "minutos": round((time.time() - t0) / 60, 2),
        "regras": {"min_chars": MIN_CHARS, "max_chars": MAX_CHARS,
                   "short_line_ratio": args.short_line_ratio, "symbol_ratio": SYMBOL_RATIO,
                   "hash": "blake2b norm_key (process_data)", "split": "hash % 100 < val_pct"},
    }
    (out.parent / (out.stem + ".meta.json")).write_text(json.dumps(ficha, indent=2, ensure_ascii=False))
    print(json.dumps(ficha, indent=2, ensure_ascii=False))
    print(f"dataset: {out} | val: {out_val} — rode a QA do /treino e depois prepare_tokens --input jsonl")


if __name__ == "__main__":
    main()
