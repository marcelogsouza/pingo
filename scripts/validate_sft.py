#!/usr/bin/env python
"""SFT-2 — valida, deduplica e separa o dataset de instruções.

Entra o JSONL cru do gerador (`scripts/gen_sft.py`); saem três arquivos prontos para o
treino — `train.jsonl`, `val.jsonl`, `heldout.jsonl` — mais `stats.json` e `rejects.jsonl`.

O que faz, nesta ordem:

1. **Revalidação** de cada linha contra as regras compartilhadas (`pingo/sft.py`): campos,
   `split=train`, tamanho do trecho, pergunta/resposta, pt, recusa no formato do modo e —
   o mais importante — **evidência literal** dentro do `chunk_text`. Não confia em quem
   escreveu o arquivo.
2. **Dedup**: (a) `(doc_id, chunk_idx)` repetido (artefato de `--resume`); (b) perguntas
   quase idênticas; (c) trechos quase idênticos (sites espelho republicando o mesmo texto).
   Usa **MinHash + LSH** (Jaccard de shingles de palavras) — O(n), sem dependência extra;
   candidatos do LSH são confirmados com Jaccard exato.
3. **Split por DOCUMENTO** (hash determinístico de `doc_id`): todas as linhas de um doc caem
   no mesmo split, então não há vazamento entre treino e avaliação. `val` e `heldout` são
   disjuntos e o script **verifica** isso antes de gravar.

`heldout` é o conjunto que fica intocado até a avaliação final do modelo SFT; `val` serve
para acompanhar a curva durante o treino.

Uso:
  uv run python scripts/validate_sft.py                                  # valida o sft-v1
  uv run python scripts/validate_sft.py --limit 3000 --out-dir /tmp/x    # amostra rápida
  uv run python scripts/validate_sft.py --val-frac 0.04 --heldout-frac 0.04
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pingo.sft import (  # noqa: E402
    TIPOS, example_reject_reason, normalize_ws,
)

# ---------------------------------------------------------------- MinHash / LSH

# p primo de Mersenne 2^61-1: base do hashing universal (a*h + b) % p
_P = (1 << 61) - 1


def _hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def _words(text: str) -> list[str]:
    out = []
    for w in normalize_ws(text).lower().split():
        w = "".join(c for c in w if c.isalnum() or c in "áàâãéêíóôõúüç")
        if w:
            out.append(w)
    return out


def shingles(text: str, k: int = 3, max_shingles: int = 0) -> set[int]:
    """Shingles de `k` palavras, já hasheados.

    `max_shingles` limita o custo em trechos longos (~500 palavras) **amostrando por hash**
    (`h % step == 0`) e não por posição: amostragem posicional com passo proporcional ao
    tamanho desalinha os dois textos e o Jaccard de um espelho de 2 palavras trocadas cai
    para 0,38. Por hash, os mesmos shingles são selecionados nos dois textos e a similaridade
    medida reflete a real.
    """
    words = _words(text)
    if len(words) < k:
        return {_hash64(" ".join(words))} if words else set()
    total = len(words) - k + 1
    step = math.ceil(total / max_shingles) if (max_shingles and total > max_shingles) else 1
    out: set[int] = set()
    for i in range(total):
        h = _hash64(" ".join(words[i:i + k]))
        if step == 1 or h % step == 0:
            out.add(h)
    return out


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class MinHashLSH:
    """MinHash com LSH em bandas: devolve candidatos; o chamador confirma com Jaccard.

    Reprojetar uma permutação por shingle seria caro; o hash universal `(a*h+b) % p` gera
    `num_perm` hashes independentes a partir de um só (hashing universal).
    """

    def __init__(self, num_perm: int = 64, bands: int = 16, seed: int = 13):
        if num_perm % bands:
            raise ValueError("num_perm precisa ser múltiplo de bands")
        self.num_perm = num_perm
        self.rows = num_perm // bands
        rng = random.Random(seed)
        self.a = [rng.randrange(1, _P) for _ in range(num_perm)]
        self.b = [rng.randrange(_P) for _ in range(num_perm)]
        # bucket -> índices (lista: colisões não podem se perder, só custam candidatos)
        self.buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}

    def signature(self, shs: Iterable[int]) -> tuple[int, ...]:
        hs = list(shs)
        if not hs:
            return tuple([0] * self.num_perm)
        return tuple(min((a * h + b) % _P for h in hs) for a, b in zip(self.a, self.b))

    def query(self, sig: tuple[int, ...], max_candidates: int = 20) -> set[int]:
        """Índices já indexados que compartilham alguma banda com `sig`."""
        found: set[int] = set()
        for band in range(self.num_perm // self.rows):
            bk = (band, tuple(sig[band * self.rows:(band + 1) * self.rows]))
            for key in self.buckets.get(bk, ())[:max_candidates]:
                found.add(key)
        return found

    def add(self, sig: tuple[int, ...], key: int) -> None:
        """Indexa a assinatura em todas as bandas (chame só para exemplos ACEITOS)."""
        for band in range(self.num_perm // self.rows):
            bk = (band, tuple(sig[band * self.rows:(band + 1) * self.rows]))
            self.buckets.setdefault(bk, []).append(key)


@dataclass
class DedupStats:
    exato: int = 0
    pergunta: int = 0
    trecho: int = 0


class NearDedup:
    """Remove quase-duplicatas de pergunta e de trecho, mantendo o primeiro que chegou."""

    def __init__(self, q_threshold: float = 0.80, chunk_threshold: float = 0.85,
                 seed: int = 13, q_perms: int = 64, q_bands: int = 16,
                 c_perms: int = 32, c_bands: int = 8):
        self.q_threshold = q_threshold
        self.chunk_threshold = chunk_threshold
        self.q_lsh = MinHashLSH(q_perms, q_bands, seed)
        self.c_lsh = MinHashLSH(c_perms, c_bands, seed + 1)
        self.q_sets: list[set[int]] = []
        self.c_sets: list[set[int]] = []
        self.stats = DedupStats()

    def _dup_against(self, shs: set[int], candidates: set[int], sets: list[set[int]],
                     threshold: float) -> bool:
        return any(jaccard(shs, sets[i]) >= threshold for i in candidates)

    def is_duplicate(self, pergunta: str, chunk: str) -> str:
        """Devolve "pergunta", "trecho" ou "" (aceita — e o exemplo já foi indexado)."""
        q_sh = shingles(pergunta, k=3, max_shingles=120)
        q_sig = self.q_lsh.signature(q_sh)
        c_sh = shingles(chunk, k=3, max_shingles=64)
        c_sig = self.c_lsh.signature(c_sh)

        # confirma com Jaccard exato só entre os candidatos do LSH
        q_cand = self.q_lsh.query(q_sig)
        if self._dup_against(q_sh, q_cand, self.q_sets, self.q_threshold):
            self.stats.pergunta += 1
            return "pergunta"
        c_cand = self.c_lsh.query(c_sig)
        if self._dup_against(c_sh, c_cand, self.c_sets, self.chunk_threshold):
            self.stats.trecho += 1
            return "trecho"

        # aceito: indexa (na mesma ordem dos índices guardados)
        self.q_lsh.add(q_sig, len(self.q_sets))
        self.q_sets.append(q_sh)
        self.c_lsh.add(c_sig, len(self.c_sets))
        self.c_sets.append(c_sh)
        return ""


# ---------------------------------------------------------------- juiz de recusa

# A recusa é o único tipo que não tem verificação automática: o professor pode perguntar
# algo que o trecho RESPONDE e ainda assim escrever "isso não consta" (aconteceu no piloto).
# Uma segunda chamada barata (só nas ~5% de recusas) resolve.
DEFAULT_JUDGE_URL = "http://localhost:8001/v1/chat/completions"

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "responde": {"type": "string", "enum": ["sim", "parcial", "nao"]},
    },
    "required": ["responde"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """Você decide se um TRECHO contém a informação pedida por uma PERGUNTA.
Responda SOMENTE com o JSON pedido.
- 'sim': a resposta está no TRECHO (dá para respondê-la lendo só ele).
- 'parcial': o TRECHO tem parte da informação, mas não responde por inteiro.
- 'nao': o TRECHO não responde à pergunta."""


def recusa_respondivel(pergunta: str, trecho: str,
                       endpoint: str = DEFAULT_JUDGE_URL,
                       timeout: float = 180.0) -> bool | None:
    """True = o trecho RESPONDE à pergunta (logo a recusa está errada). None = erro."""
    body = {
        "model": "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL",
        "temperature": 0.0, "max_tokens": 64,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"TRECHO:\n{trecho}\n\nPERGUNTA: {pergunta}"},
        ],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "checagem", "schema": JUDGE_SCHEMA,
                                            "strict": True}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            dados = json.load(resp)
        return json.loads(dados["choices"][0]["message"]["content"])["responde"] == "sim"
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return None


def filtrar_recusas(recusas: list[dict], args) -> tuple[list[dict], int, int]:
    """Descarta recusas cujo trecho responde à pergunta. Devolve (mantidas, descartadas, erros)."""
    if not recusas:
        return [], 0, 0
    with ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
        vereditos = list(tqdm(pool.map(
            lambda r: recusa_respondivel(r["pergunta"], r["chunk_text"], args.judge_url,
                                         args.judge_timeout), recusas),
            total=len(recusas), desc="checando recusas", unit="rec"))
    mantidas = [r for r, v in zip(recusas, vereditos) if v is not True]
    descartadas = sum(1 for v in vereditos if v is True)
    erros = sum(1 for v in vereditos if v is None)
    return mantidas, descartadas, erros


# ---------------------------------------------------------------- split e I/O


def split_of(doc_id: str, seed: int, val_frac: float, heldout_frac: float) -> str:
    """Split determinístico por documento (hash uniforme → frações aproximadas)."""
    h = int.from_bytes(hashlib.blake2b(f"{seed}:{doc_id}".encode(), digest_size=8).digest(), "big")
    x = h / 2**64
    if x < heldout_frac:
        return "heldout"
    if x < heldout_frac + val_frac:
        return "val"
    return "train"


def read_jsonl(path: Path, limit: int = 0) -> Iterator[tuple[int, dict | None]]:
    """Lê tolerando a linha final truncada (o gerador pode estar escrevendo no arquivo)."""
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                yield i, None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path, default=ROOT / "data" / "sft" / "sft-v1.jsonl")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "sft" / "final")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--heldout-frac", type=float, default=0.05)
    p.add_argument("--q-threshold", type=float, default=0.80,
                   help="Jaccard mínimo p/ considerar duas perguntas quase-duplicadas")
    p.add_argument("--chunk-threshold", type=float, default=0.85,
                   help="Jaccard mínimo p/ considerar dois trechos quase-duplicados")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--limit", type=int, default=0, help="lê só as N primeiras linhas (teste)")
    p.add_argument("--max-rejects", type=int, default=500,
                   help="quantos exemplos rejeitados gravar em rejects.jsonl")
    p.add_argument("--judge-recusas", action="store_true",
                   help="confere cada recusa com o professor (descarta as que o trecho responde)")
    p.add_argument("--judge-url", default="http://localhost:8001/v1/chat/completions")
    p.add_argument("--judge-workers", type=int, default=8)
    p.add_argument("--judge-timeout", type=float, default=180.0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.inp.exists():
        sys.exit(f"{args.inp} não existe — rode scripts/gen_sft.py antes")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reasons: Counter = Counter()
    dedup = NearDedup(args.q_threshold, args.chunk_threshold, args.seed)
    seen_chunks: set[tuple[str, int]] = set()
    aceitos: list[dict] = []
    recusas: list[dict] = []
    rejects: list[dict] = []

    total = 0
    for _i, rec in tqdm(read_jsonl(args.inp, args.limit), desc="validando", unit="ex"):
        total += 1
        if rec is None:
            reasons["json_invalido"] += 1
            continue
        key = (rec.get("doc_id", ""), int(rec.get("chunk_idx", -1)))
        if key in seen_chunks:
            reasons["duplicado_exato"] += 1
            continue
        motivo = example_reject_reason(rec)
        if motivo:
            reasons[motivo.split(":")[0]] += 1
            if len(rejects) < args.max_rejects:
                rejects.append({"motivo": motivo, "doc_id": rec.get("doc_id"),
                                "pergunta": rec.get("pergunta"),
                                "resposta": rec.get("resposta"),
                                "evidencia": rec.get("evidencia")})
            continue
        dup = dedup.is_duplicate(rec["pergunta"], rec["chunk_text"])
        if dup:
            reasons[f"quase_duplicado_{dup}"] += 1
            continue
        seen_chunks.add(key)  # só conta como "visto" o que foi aceito
        if rec["tipo"] == "recusa" and args.judge_recusas:
            recusas.append(rec)  # julga depois, em paralelo, junto com as demais
            continue
        aceitos.append(rec)

    if recusas:
        print(f"checando {len(recusas)} recusas com o professor "
              f"(descarta as que o trecho responde)...")
        mantidas, descartadas, erros = filtrar_recusas(recusas, args)
        reasons["recusa_respondivel"] += descartadas
        reasons["judge_erro"] += erros
        aceitos.extend(mantidas)

    tipos: Counter = Counter(r["tipo"] for r in aceitos)
    modos: Counter = Counter("ancorado" if r["ancorado"] else "livre" for r in aceitos)
    splits: dict[str, list[dict]] = defaultdict(list)
    for rec in aceitos:
        rec["split"] = split_of(rec["doc_id"], args.seed, args.val_frac, args.heldout_frac)
        splits[rec["split"]].append(rec)

    # ---- ordem aleatória determinística (evita blocos por domínio) e gravação
    rng = random.Random(args.seed)
    docs_por_split: dict[str, set[str]] = {}
    for nome in ("train", "val", "heldout"):
        recs = splits.get(nome, [])
        rng.shuffle(recs)
        with (args.out_dir / f"{nome}.jsonl").open("w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        docs_por_split[nome] = {r["doc_id"] for r in recs}

    # ---- garantia de que não há vazamento entre splits (é a razão de separar por doc)
    leaks = {
        "train_val": len(docs_por_split["train"] & docs_por_split["val"]),
        "train_heldout": len(docs_por_split["train"] & docs_por_split["heldout"]),
        "val_heldout": len(docs_por_split["val"] & docs_por_split["heldout"]),
    }
    if any(leaks.values()):
        sys.exit(f"VAZAMENTO entre splits: {leaks} — nada foi publicado, investigue")

    with (args.out_dir / "rejects.jsonl").open("w", encoding="utf-8") as fh:
        for rec in rejects:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    aceitos = sum(len(v) for v in splits.values())
    stats = {
        "entrada": str(args.inp),
        "linhas_lidas": total,
        "aceitos": aceitos,
        "rejeitados": total - aceitos,
        "motivos": dict(reasons.most_common()),
        "quase_duplicatas": {"pergunta": dedup.stats.pergunta, "trecho": dedup.stats.trecho},
        "juiz_de_recusas": {
            "ligado": args.judge_recusas,
            "checadas": len(recusas) if args.judge_recusas else 0,
            "descartadas_respondiveis": reasons.get("recusa_respondivel", 0),
            "erros": reasons.get("judge_erro", 0),
        },
        "splits": {
            nome: {
                "exemplos": len(splits.get(nome, [])),
                "docs": len(docs_por_split[nome]),
                "exemplos_por_doc": round(len(splits.get(nome, [])) / max(len(docs_por_split[nome]), 1), 2),
            } for nome in ("train", "val", "heldout")
        },
        "vazamento_docs": leaks,
        "tipos": dict(tipos),
        "modos": dict(modos),
        "parametros": {
            "seed": args.seed, "val_frac": args.val_frac, "heldout_frac": args.heldout_frac,
            "q_threshold": args.q_threshold, "chunk_threshold": args.chunk_threshold,
        },
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
