#!/usr/bin/env python
"""Passo 2 — baixa shards `pt` do Community OSCAR para `data/<dump>/pt/`.

Retomável: partes completas (tamanho confere com o repo) são puladas;
downloads parciais continuam via HTTP Range. Paralelizável e respeita um
orçamento em GB (o plano pede ~40–50 GB baixados).

Uso:
  uv run python scripts/download_data.py                 # dump mais recente com pt (2024-10), até 45 GB
  uv run python scripts/download_data.py --dumps 2023-40 # completo: 2023-40 (~42 GB)
  uv run python scripts/download_data.py --dry-run       # só lista o plano de download
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

REPO = "oscar-corpus/community-oscar"
ROOT = Path(__file__).resolve().parents[1]


def partnum(path: str) -> int:
    return int(path.split("pt_part_")[1].split(".")[0])


def list_parts(dump: str) -> list[tuple[str, int]]:
    """Parte (path_no_repo, bytes) de cada shard pt do dump, ordenado por part."""
    api = HfApi()
    entries = api.list_repo_tree(
        REPO, repo_type="dataset", path_in_repo=f"data/{dump}/pt"
    )
    files = [(e.path, e.size) for e in entries if e.path.endswith(".jsonl.zst")]
    if any(s is None for _, s in files):
        sys.exit(f"erro: metadados de tamanho ausentes em {dump}")
    return sorted(files, key=lambda x: partnum(x[0]))


def download(url: str, dest: Path, expected: int, token: str, retries: int = 5) -> int:
    """Baixa (ou retoma) um shard; retorna bytes baixados nesta chamada."""
    part = dest.with_suffix(dest.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}
    downloaded_now = 0

    for attempt in range(1, retries + 1):
        resume_from = part.stat().st_size if part.exists() else 0
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=120) as resp, \
                    open(part, "ab" if resp.status == 206 else "wb") as fh:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"HTTP {resp.status}")
                resume_from = part.stat().st_size if resp.status == 206 else 0
                shutil.copyfileobj(resp, fh, length=8 * 2**20)
            if part.stat().st_size == expected:
                part.rename(dest)
                return downloaded_now
            # tamanho errado: descarta e tenta do zero
            part.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 — rede: retry com backoff
            if attempt == retries:
                raise
            wait = 2**attempt
            print(f"  [retry {attempt}/{retries}] {dest.name}: {exc} "
                  f"({part.stat().st_size if part.exists() else 0} bytes salvos, "
                  f"espera {wait}s)", flush=True)
            time.sleep(wait)
    return downloaded_now


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", nargs="+", default=["2024-10"],
                    help="dumps do Community OSCAR com pt (ex.: 2024-10 2023-40)")
    ap.add_argument("--max-gb", type=float, default=45.0,
                    help="orçamento total de download (partes já locais contam)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = get_token()
    if not token:
        sys.exit("erro: sem token HF (rode `uv run hf auth login` e aceite o gate)")

    budget = args.max_gb * 1e9
    plan: list[tuple[str, str, int, Path]] = []
    spent = 0.0
    orcado = False
    for dump in args.dumps:
        if orcado:
            break
        for path, size in list_parts(dump):
            if spent + size > budget and plan:
                print(f"orçamento de {args.max_gb} GB atingido — parando em {path}")
                orcado = True
                break
            dest = args.out / path.removeprefix("data/")  # data/<dump>/pt/pt_part_N.jsonl.zst
            done = dest.exists() and dest.stat().st_size == size
            if not done:  # já baixado não gasta orçamento novo
                spent += size
            plan.append(("ok" if done else "baixar", path, size, dest))

    to_get = [p for s, p, _, _ in plan if s == "baixar"]
    print(f"plano: {len(plan)} parts pt ({spent / 1e9:.1f} GB a baixar) nos dumps "
          f"{args.dumps}; {len(plan) - len(to_get)} já locais")
    if args.dry_run or not to_get:
        return

    t0 = time.time()
    got = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(download,
                        f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}",
                        dest, size, token, args.retries): path
            for status, path, size, dest in plan
            if status == "baixar"
        }
        for i, fut in enumerate(as_completed(futs), 1):
            path = futs[fut]
            got += fut.result()
            el = time.time() - t0
            print(f"[{i}/{len(futs)}] {path} — {got / 1e9:.1f} GB, "
                  f"{got / el / 1e6:.0f} MB/s médios", flush=True)

    stats = {
        "repo": REPO,
        "dumps": args.dumps,
        "parts": len(plan),
        "bytes_baixados_essa_execucao": got,
        "segundos": round(time.time() - t0, 1),
    }
    out_stats = args.out / "download_stats.json"
    out_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"concluído: {stats}")


if __name__ == "__main__":
    main()
