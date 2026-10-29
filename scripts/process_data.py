#!/usr/bin/env python
"""Passo 2 — converte shards `.jsonl.zst` do Community OSCAR em parquet filtrado.

Pipeline do plano: extrai o campo `content` → filtro de qualidade local
(comprimento de doc, ratio de linhas curtas, ratio de símbolos, dedup exato por
hash de doc) → parquet em `data/raw/`. ~2% dos docs ficam marcados `split=val`
(determinístico por hash), segurados para validação.

Uso:
  uv run python scripts/process_data.py                 # todos os dumps locais
  uv run python scripts/process_data.py --dumps 2024-10 --limit-shards 2  # teste
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]

SYMBOLS = "#%&<>=|\\{}[]*~^_=+"
SHORT_LINE_CHARS = 30


def part_of(name: str) -> int:
    return int(name.split("pt_part_")[1].split(".")[0])


def norm_key(content: str) -> bytes:
    """Chave de dedup: whitespace colapsado → blake2b de 8 bytes (como int)."""
    return hashlib.blake2b(" ".join(content.split()).encode(), digest_size=8).digest()


def passes(content: str, args) -> tuple[bool, str]:
    n = len(content)
    if n < args.min_chars or n > args.max_chars:
        return False, "len"
    lines = content.split("\n")
    if sum(1 for ln in lines if len(ln.strip()) < SHORT_LINE_CHARS) > args.short_line_ratio * len(lines):
        return False, "short_lines"
    nsym = sum(content.count(c) for c in SYMBOLS)
    if nsym > args.symbol_ratio * n:
        return False, "symbols"
    return True, ""


def process_shard(zst_path: Path, out_dir: Path, seen: set[bytes], args, stats: dict) -> None:
    stem = zst_path.name.removesuffix(".jsonl.zst")  # ex.: pt_part_8
    out_path = out_dir / (zst_path.parent.parent.name + "-" + stem + ".parquet")
    if out_path.exists() and not args.overwrite:
        print(f"  pulando {out_path.name} (já existe; use --overwrite)", flush=True)
        return

    schema = pa.schema([
        ("id", pa.string()), ("content", pa.string()), ("url", pa.string()),
        ("dump", pa.string()), ("part", pa.int32()), ("split", pa.string()),
    ])
    partnum = part_of(zst_path.name)
    dump = zst_path.parent.parent.name

    rows: dict[str, list] = {c: [] for c in schema.names}

    def flush(writer: pq.ParquetWriter | None) -> pq.ParquetWriter:
        w = writer or pq.ParquetWriter(str(out_path), schema, compression="zstd")
        w.write_table(pa.Table.from_pydict(rows, schema=schema))
        for v in rows.values():
            v.clear()
        return w

    writer = None
    with zstd.open(zst_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            stats["docs_in"] += 1
            try:
                rec = json.loads(line)
                content = rec["content"]
            except (json.JSONDecodeError, KeyError):
                stats["bad_json"] += 1
                continue
            ok, why = passes(content, args)
            if not ok:
                stats[f"drop_{why}"] += 1
                continue
            key = norm_key(content)
            if key in seen:
                stats["drop_dup"] += 1
                continue
            seen.add(key)
            headers = rec.get("warc_headers") or {}
            rows["id"].append(key.hex())
            rows["content"].append(content)
            rows["url"].append(headers.get("warc-target-uri"))
            rows["dump"].append(dump)
            rows["part"].append(partnum)
            rows["split"].append("val" if int.from_bytes(key, "big") % 100 < args.val_pct else "train")
            stats["kept"] += 1
            stats["chars"] += len(content)
            if len(rows["id"]) >= args.row_group_docs:
                writer = flush(writer)

    if rows["id"]:
        writer = flush(writer)
    if writer:
        writer.close()
        print(f"  {out_path.name}: {stats['kept']} docs", flush=True)
    else:
        print(f"  {zst_path.name}: 0 docs aprovados", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", nargs="+", default=["2024-10"])
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=100_000)
    ap.add_argument("--short-line-ratio", type=float, default=0.6)
    ap.add_argument("--symbol-ratio", type=float, default=0.08)
    ap.add_argument("--val-pct", type=int, default=2)
    ap.add_argument("--row-group-docs", type=int, default=50_000)
    ap.add_argument("--limit-shards", type=int, default=0, help="só as N primeiras parts (teste)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    shards: list[Path] = []
    for dump in args.dumps:
        shards += sorted((args.data_dir / dump / "pt").glob("pt_part_*.jsonl.zst"),
                         key=lambda p: part_of(p.name))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    if not shards:
        sys.exit("nenhum shard .jsonl.zst encontrado — rode download_data.py antes")

    args.out.mkdir(parents=True, exist_ok=True)
    seen: set[bytes] = set()
    stats = dict.fromkeys(["docs_in", "kept", "bad_json", "drop_len",
                           "drop_short_lines", "drop_symbols", "drop_dup", "chars"], 0)
    for shard in shards:
        before = dict(stats)
        print(f"processando {shard.relative_to(ROOT)} ...", flush=True)
        process_shard(shard, args.out, seen, args, stats)
        shard_stats = {k: stats[k] - before[k] for k in stats}
        print(f"  → in={shard_stats['docs_in']} kept={shard_stats['kept']} "
              f"len={shard_stats['drop_len']} short={shard_stats['drop_short_lines']} "
              f"sym={shard_stats['drop_symbols']} dup={shard_stats['drop_dup']}", flush=True)

    summary = {
        "shards": len(shards),
        **stats,
        "docs_unicos": len(seen),
        "chars_total": stats["chars"],
        "aproveitamento": round(stats["kept"] / max(stats["docs_in"], 1), 4),
        "tokens_est_4chars_por_token": round(stats["chars"] / 4),
    }
    (args.out / "stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("resumo:", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
