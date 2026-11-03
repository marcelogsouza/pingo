#!/usr/bin/env python
"""QA de dataset (treino-2) — o contrato do "dataset qualquer" da interface /treino.

Contrato: **JSONL, uma linha = um objeto JSON com `content` string não-vazia**. Todo
ingest novo (Wikipedia XML, CSV, HF) termina num JSONL assim; a QA é o portão antes de
gastar minutos de tokenização e horas de treino em dado torto.

O relatório é estruturado (a WebUI mostra card a card; a CLI imprime legível):
linhas boas/ruins, `content` ausente (com o campo candidato mais provável), duplicatas
exatas (o MESMO hash blake2b do `process_data.py`), distribuição de comprimento e os
avisos de qualidade com as réguas do `process_data.py` (300+ chars, linhas curtas,
símbolos). Duplicata/comprimento são AVISOS — o usuário decide; JSON quebrado e
`content` faltando são reprovação, porque ali o contrato não vale.

Uso:
  uv run python scripts/qa_dataset.py data/sft/final/train.jsonl        # legível
  uv run python scripts/qa_dataset.py dataset.jsonl --json              # JSON cru (a UI usa a biblioteca)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_data import SHORT_LINE_CHARS, SYMBOLS, norm_key  # as MESMAS réguas do filtro

CAMPO = "content"
MIN_CHARS = 300            # default do --min-chars do process_data.py
SHORT_LINE_RATIO = 0.6     # idem
SYMBOL_RATIO = 0.08        # idem
MAX_EXEMPLOS = 5           # quantos exemplos de cada problema mostrar
MAX_AMOSTRAS = 5


def _percentil(ordenado: list[int], p: float) -> int:
    if not ordenado:
        return 0
    return ordenado[min(len(ordenado) - 1, int(p * len(ordenado)))]


def analisar(path: Path, sample: int = MAX_AMOSTRAS, seed: int = 13,
             short_line_ratio: float = SHORT_LINE_RATIO) -> dict:
    """Varre o JSONL (streaming, memória O(n_docs) só p/ hashes) e devolve o relatório."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    rng = random.Random(seed)
    total = vazias = 0
    content_ok = content_ausente = content_vazio = content_nao_string = 0
    json_erros: list[dict] = []
    ausentes: list[dict] = []
    candidatos: dict[str, int] = {}
    curtos = muitos_simbolos = muitas_linhas_curtas = 0
    duplicatas = 0
    vistos: set[bytes] = set()
    comprimentos: list[int] = []
    amostras: list[dict] = []          # amostragem reservatória
    n_amostras_vistas = 0

    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, start=1):
            total += 1
            if not line.strip():
                vazias += 1
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                if len(json_erros) < MAX_EXEMPLOS:
                    json_erros.append({"linha": i, "motivo": str(exc),
                                       "trecho": line[:120].strip()})
                continue
            if not isinstance(rec, dict):
                if len(json_erros) < MAX_EXEMPLOS:
                    json_erros.append({"linha": i, "motivo": "linha não é um objeto JSON",
                                       "trecho": line[:120].strip()})
                continue

            if CAMPO not in rec:
                content_ausente += 1
                if len(ausentes) < MAX_EXEMPLOS:
                    ausentes.append({"linha": i, "chaves": list(rec)[:12]})
                for k, v in rec.items():          # sugere o campo que parece ser o texto
                    if isinstance(v, str) and len(v) >= MIN_CHARS:
                        candidatos[k] = candidatos.get(k, 0) + 1
                continue
            content = rec[CAMPO]
            if not isinstance(content, str):
                content_nao_string += 1
                continue
            if not content.strip():
                content_vazio += 1
                continue

            content_ok += 1
            n = len(content)
            comprimentos.append(n)
            if n < MIN_CHARS:
                curtos += 1
            linhas = content.split("\n")
            if sum(1 for ln in linhas if len(ln.strip()) < SHORT_LINE_CHARS) > short_line_ratio * len(linhas):
                muitas_linhas_curtas += 1
            if sum(content.count(c) for c in SYMBOLS) > SYMBOL_RATIO * n:
                muitos_simbolos += 1
            key = norm_key(content)
            if key in vistos:
                duplicatas += 1
            else:
                vistos.add(key)
            n_amostras_vistas += 1                # reservoir sampling: uniforme sem saber o total
            if len(amostras) < sample:
                amostras.append({"linha": i, "chars": n,
                                 "inicio": content[:160], "fim": content[-80:]})
            else:
                j = rng.randrange(n_amostras_vistas)
                if j < sample:
                    amostras[j] = {"linha": i, "chars": n,
                                   "inicio": content[:160], "fim": content[-80:]}

    comprimentos.sort()
    problemas, avisos = [], []
    if vazias:
        problemas.append(f"{vazias} linha(s) vazia(s)")
    if json_erros:
        problemas.append(f"{total - content_ok} linha(s) fora do formato JSON (as 5 primeiras listadas)")
    if content_ausente:
        sug = max(candidatos, key=candidatos.get) if candidatos else None
        problemas.append(f"{content_ausente} linha(s) sem o campo `{CAMPO}`"
                         + (f" — o campo `{sug}` parece ser o texto" if sug else ""))
    if content_nao_string:
        problemas.append(f"{content_nao_string} linha(s) com `{CAMPO}` não-string")
    if content_vazio:
        problemas.append(f"{content_vazio} linha(s) com `{CAMPO}` vazio")
    if duplicatas:
        avisos.append(f"{duplicatas} documento(s) repetido(s) exatamente (o process_data.dedup jogaria fora)")
    if curtos:
        avisos.append(f"{curtos} documento(s) com menos de {MIN_CHARS} chars")
    if muitas_linhas_curtas:
        avisos.append(f"{muitas_linhas_curtas} documento(s) com maioria de linhas curtas")
    if muitos_simbolos:
        avisos.append(f"{muitos_simbolos} documento(s) com excesso de símbolos")

    return {
        "arquivo": str(path) if not path.is_relative_to(ROOT) else str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "linhas": total,
        "content_ok": content_ok,
        "problemas": problemas,
        "avisos": avisos,
        "aprovado": not problemas,
        "detalhe": {
            "vazias": vazias, "json_erros": json_erros, "content_ausente": ausentes,
            "content_nao_string": content_nao_string, "content_vazio": content_vazio,
            "campo_candidato": max(candidatos, key=candidatos.get) if candidatos else None,
        },
        "comprimento": {
            "min": comprimentos[0] if comprimentos else 0,
            "p50": _percentil(comprimentos, 0.50),
            "p90": _percentil(comprimentos, 0.90),
            "max": comprimentos[-1] if comprimentos else 0,
            "total_chars": sum(comprimentos),
            "media_chars": round(sum(comprimentos) / len(comprimentos)) if comprimentos else 0,
        },
        "duplicatas": duplicatas,
        "amostras": amostras,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arquivo", type=Path)
    ap.add_argument("--json", action="store_true", help="devolve o relatório cru (a UI usa assim)")
    ap.add_argument("--sample", type=int, default=MAX_AMOSTRAS)
    ap.add_argument("--short-line-ratio", type=float, default=SHORT_LINE_RATIO,
                    help="0.9 p/ wiki (títulos de seção são linhas curtas legítimas); 0.6 é a "
                         "régua do OSCAR")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    try:
        rel = analisar(args.arquivo, sample=args.sample, seed=args.seed,
                       short_line_ratio=args.short_line_ratio)
    except FileNotFoundError:
        sys.exit(f"arquivo não encontrado: {args.arquivo}")

    if args.json:
        print(json.dumps(rel, ensure_ascii=False, indent=2))
    else:
        c = rel["comprimento"]
        print(f"{rel['arquivo']} — {rel['linhas']} linhas, {rel['bytes']:,} bytes")
        print(f"  content ok: {rel['content_ok']} | duplicatas: {rel['duplicatas']}")
        print(f"  chars/doc: min {c['min']:,} | p50 {c['p50']:,} | p90 {c['p90']:,} | "
              f"max {c['max']:,} | média {c['media_chars']:,}")
        for p in rel["problemas"]:
            print(f"  ✖ {p}")
        for a in rel["avisos"]:
            print(f"  ⚠ {a}")
        if rel["detalhe"]["json_erros"]:
            for e in rel["detalhe"]["json_erros"]:
                print(f"    linha {e['linha']}: {e['motivo']}")
        print(f"  {'APROVADO' if rel['aprovado'] else 'REPROVADO — arrume antes de tokenizar'}")
    sys.exit(0 if rel["aprovado"] else 1)


if __name__ == "__main__":
    main()
