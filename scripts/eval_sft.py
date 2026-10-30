#!/usr/bin/env python
"""SFT-4 — avaliação do modelo de instrução no conjunto `heldout`.

Três perguntas, na ordem em que importam:

1. **Ele responde no formato?** Gera no prompt de cada exemplo do `heldout` e mede se a
   resposta termina em `<|end|>` (aprendeu o fim de turno) e o quão perto fica da resposta de
   referência (F1 de palavras, estilo SQuAD).
2. **Ele é fiel?** Para os exemplos **ancorados** (o trecho vai no prompt) e também para os
   sem contexto (o trecho é a fonte de verdade do par), mede quanto do conteúdo da resposta
   gerada aparece no trecho de origem. Com `--judge`, o professor local dá o veredito
   (apoiada / parcial / não apoiada) — é o número que vale para "não inventou".
3. **Ele esqueceu o que sabia?** Compara a perplexidade no `val` do **pré-treino**
   (`data/tokens/val-*.bin`) antes (modelo base) e depois (SFT). Se a ppl subir muito, o SFT
   atropelou o pré-treino ("esquecimento catastrófico").

Uso:
  uv run python scripts/eval_sft.py                                   # SFT vs base, sem juiz
  uv run python scripts/eval_sft.py --n 300 --judge                   # + veredito do professor
  uv run python scripts/eval_sft.py --model runs/pt-120m/final         # avaliar o base
  uv run python scripts/eval_sft.py --no-ppl --no-gerar                # só reimprime métricas?
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pingo.data import load_tokens  # noqa: E402
from pingo.generate import generate  # noqa: E402
from pingo.model import PingoLM  # noqa: E402
from pingo.sft import (  # noqa: E402
    END_TOKEN, PT_STOP, RECUSA_ANCORADO, RECUSA_LIVRE, ensure_chat_tokens, normalize_ws,
    render,
)

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


# ---------------------------------------------------------------- métricas


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in PT_STOP]


def f1_words(gerado: str, referencia: str) -> float:
    """F1 de palavras (multiconjunto) entre geração e resposta de referência."""
    pred, ref = Counter(words(gerado)), Counter(words(referencia))
    if not pred or not ref:
        return 0.0
    comuns = sum((pred & ref).values())
    if not comuns:
        return 0.0
    precisao = comuns / sum(pred.values())
    recall = comuns / sum(ref.values())
    return 2 * precisao * recall / (precisao + recall)


def _stem(palavra: str) -> str:
    """Radical grosseiro: 5 primeiras letras ("investe"/"investimentos" → "inves")."""
    return palavra[:5]


def support_score(gerado: str, trecho: str) -> float:
    """Fração das palavras de conteúdo da geração sustentadas pelo trecho de origem.

    Compara radicais (5 letras) para não punir flexão ("investe" x "investimentos"). É uma
    medida **lexical**: 1,0 = tudo que ele afirmou está no trecho; baixo = inventou vocabulário.
    Paráfrase legítima também derruba o número (é o limite do método) — por isso `--judge`.
    """
    cw = [_stem(w) for w in content_words(gerado)]
    if not cw:
        return 0.0
    fonte = {_stem(w) for w in words(trecho)}
    return sum(1 for w in cw if w in fonte) / len(cw)


def recusa_ok(gerado: str, ancorado: bool) -> bool:
    alvo = (RECUSA_ANCORADO if ancorado else RECUSA_LIVRE).lower()
    return normalize_ws(gerado).lower().startswith(alvo[:24])


def limpar_geracao(texto: str) -> str:
    """Tira o que o tokenizer decodifica como token especial/ruído de borda."""
    texto = texto.replace(END_TOKEN, " ").replace("<|user|>", " ").replace("<|assistant|>", " ")
    return normalize_ws(texto)


def score_example(rec: dict, gerado: str) -> dict:
    """Métricas de UM exemplo (função pura — é o que os testes cobrem)."""
    ancorado = bool(rec["ancorado"])
    limpo = limpar_geracao(gerado)
    out = {
        "tipo": rec["tipo"], "ancorado": ancorado,
        "gerado": limpo,
        "terminou": END_TOKEN in gerado,
        "f1": round(f1_words(limpo, rec.get("resposta", "")), 4),
    }
    if rec["tipo"] == "recusa":
        out["recusa_ok"] = recusa_ok(limpo, ancorado)
    else:
        out["apoio"] = round(support_score(limpo, rec.get("chunk_text", "")), 4)
    return out


def agregar(scores: list[dict]) -> dict:
    def media(valores):
        valores = [v for v in valores if v is not None]
        return round(sum(valores) / len(valores), 4) if valores else None

    respondiveis = [s for s in scores if s["tipo"] != "recusa"]
    recusas = [s for s in scores if s["tipo"] == "recusa"]
    por_tipo: dict[str, dict] = {}
    for tipo in sorted({s["tipo"] for s in scores}):
        grupo = [s for s in scores if s["tipo"] == tipo]
        por_tipo[tipo] = {
            "n": len(grupo),
            "terminou": media([1.0 if s["terminou"] else 0.0 for s in grupo]),
            "f1": media([s["f1"] for s in grupo]),
            "apoio": media([s.get("apoio") for s in grupo]),
            "recusa_ok": media([1.0 if s.get("recusa_ok") else 0.0 for s in grupo
                                if "recusa_ok" in s]),
        }
    return {
        "n": len(scores),
        "taxa_terminou": media([1.0 if s["terminou"] else 0.0 for s in scores]),
        "f1": media([s["f1"] for s in scores]),
        "apoio_respondiveis": media([s.get("apoio") for s in respondiveis]),
        "apoio_ancorado": media([s.get("apoio") for s in respondiveis if s["ancorado"]]),
        "apoio_sem_contexto": media([s.get("apoio") for s in respondiveis
                                     if not s["ancorado"]]),
        "recusa_ok": media([1.0 if s.get("recusa_ok") else 0.0 for s in recusas
                            if "recusa_ok" in s]),
        "por_tipo": por_tipo,
    }


# ---------------------------------------------------------------- geração


def dominant_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def gerar_resposta(model, tokenizer, rec: dict, args, device: str,
                   end_id: int | None) -> str:
    prompt, _ = render(rec)
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if len(ids) > args.max_len:
        ids = ids[: args.max_len]
    out = generate(model, ids, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                   stop_token=end_id, seed=args.seed,
                   repetition_penalty=args.repetition_penalty)
    novos = out[0, len(ids):].tolist()
    return tokenizer.decode(novos, skip_special_tokens=False)


# ---------------------------------------------------------------- juiz (professor)


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "apoio": {"type": "string", "enum": ["sim", "parcial", "nao"]},
        "motivo": {"type": "string"},
    },
    "required": ["apoio", "motivo"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """Você avalia respostas de um modelo pequeno comparando-as com um TRECHO
fonte. Responda SOMENTE com o JSON pedido.

- 'sim': a resposta está sustentada pelo TRECHO (fatos conferem).
- 'parcial': parte da resposta confere, parte não aparece no TRECHO.
- 'nao': a resposta contradiz o TRECHO ou inclui fatos que não estão nele."""


def julgar(pergunta: str, trecho: str, resposta: str, url: str, timeout: float = 120.0) -> dict:
    body = {
        "model": "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL",
        "temperature": 0.0, "max_tokens": 200,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"TRECHO:\n{trecho}\n\nPERGUNTA: {pergunta}\n"
                                        f"RESPOSTA DO MODELO: {resposta}"},
        ],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "julgamento", "schema": JUDGE_SCHEMA,
                                            "strict": True}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            dados = json.load(resp)
        return json.loads(dados["choices"][0]["message"]["content"])
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        return {"apoio": "erro", "motivo": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------- perplexidade (esquecimento)


@torch.no_grad()
def ppl_pre_treino(model, tokens_dir: Path, args, device: str) -> dict:
    """Loss no `val` do PRÉ-treino (mesma régua do Passo 7) — mede esquecimento catastrófico."""
    from pingo.data import get_batch

    data = load_tokens(tokens_dir)
    rng = __import__("numpy").random.default_rng(20_260_101)
    model.eval()
    total, n = 0.0, 0
    for _ in range(args.ppl_iters):
        x, y = get_batch(data.val, args.ppl_batch, args.ppl_ctx, rng)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda"
               else contextlib.nullcontext())
        with ctx:
            loss = model(x, labels=y).loss
        total += loss.float().item()
        n += 1
    loss = total / max(n, 1)
    return {"val_loss": round(loss, 4), "val_pplx": round(math.exp(min(loss, 20)), 3),
            "tokens": n * args.ppl_batch * args.ppl_ctx}


# ---------------------------------------------------------------- CLI / main


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=ROOT / "runs" / "sft-120m" / "final")
    p.add_argument("--base", type=Path, default=ROOT / "runs" / "pt-120m" / "final",
                   help="modelo base p/ comparar a ppl do pré-treino")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer" / "tokenizer.json")
    p.add_argument("--heldout", type=Path, default=ROOT / "data" / "sft" / "final" / "heldout.jsonl")
    p.add_argument("--out", type=Path, default=None, help="default: <model>/eval_sft.json")
    p.add_argument("--n", type=int, default=200, help="quantos exemplos gerar (0 = todos)")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-len", type=int, default=1024, help="trunca o prompt (trecho + pergunta)")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default=None)
    p.add_argument("--judge", action="store_true", help="pede veredito ao professor local")
    p.add_argument("--judge-url", default="http://localhost:8001/v1/chat/completions")
    p.add_argument("--judge-workers", type=int, default=8)
    p.add_argument("--ppl", dest="ppl", action="store_true", default=True)
    p.add_argument("--no-ppl", dest="ppl", action="store_false")
    p.add_argument("--no-gerar", dest="gerar", action="store_false", default=True)
    p.add_argument("--tokens-dir", type=Path, default=ROOT / "data" / "tokens")
    p.add_argument("--ppl-iters", type=int, default=10)
    p.add_argument("--ppl-batch", type=int, default=8)
    p.add_argument("--ppl-ctx", type=int, default=1024)
    p.add_argument("--show", type=int, default=5, help="quantas gerações imprimir")
    return p.parse_args(argv)


def carregar(model_dir: Path, tokenizer_path: Path, device: str):
    tok_file = Path(model_dir) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_file if tok_file.exists() else tokenizer_path))
    ensure_chat_tokens(tokenizer)          # idempotente (base não tem, SFT já tem)
    model = PingoLM.from_pretrained(model_dir).to(device).eval()
    return model, tokenizer


def avaliar(model, tokenizer, recs: list[dict], args, device: str) -> tuple[list[dict], list[dict]]:
    end_id = tokenizer.token_to_id(END_TOKEN)
    scores, amostras = [], []
    for rec in tqdm(recs, desc="gerando", unit="ex"):
        gerado = gerar_resposta(model, tokenizer, rec, args, device, end_id)
        s = score_example(rec, gerado)
        s["pergunta"] = rec["pergunta"]
        scores.append(s)
        if len(amostras) < args.show:
            amostras.append({"pergunta": rec["pergunta"], "referencia": rec["resposta"],
                             "gerado": s["gerado"], "tipo": rec["tipo"],
                             "ancorado": s["ancorado"], "f1": s["f1"]})
    return scores, amostras


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = args.device or dominant_device()
    out = args.out or (Path(args.model) / "eval_sft.json")

    recs = [json.loads(linha) for linha in Path(args.heldout).read_text(
        encoding="utf-8").splitlines() if linha.strip()]
    random.Random(args.seed).shuffle(recs)
    if args.n:
        recs = recs[: args.n]

    report: dict = {"model": str(args.model), "heldout": str(args.heldout),
                    "n": len(recs), "device": device,
                    "decodificacao": {"temperature": args.temperature, "max_new_tokens":
                                      args.max_new_tokens,
                                      "repetition_penalty": args.repetition_penalty}}
    model, tokenizer = carregar(Path(args.model), args.tokenizer, device)
    report["vocab"] = tokenizer.get_vocab_size()

    if args.gerar:
        scores, amostras = avaliar(model, tokenizer, recs, args, device)
        report["metricas"] = agregar(scores)
        report["amostras"] = amostras
        print(json.dumps(report["metricas"], indent=2, ensure_ascii=False))
        for a in amostras:
            print(f"\n[{'ancorado' if a['ancorado'] else 'livre'}/{a['tipo']}] {a['pergunta']}")
            print(f"  ref:    {a['referencia'][:150]}")
            print(f"  gerado: {a['gerado'][:150]}")

        if args.judge:
            print(f"\nveredito do professor em {len(scores)} respostas...")
            with ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
                vereditos = list(tqdm(pool.map(
                    lambda par: julgar(par[0]["pergunta"], par[0]["chunk_text"],
                                       par[1]["gerado"], par[0]["url"], timeout=180.0),
                    zip(recs, scores)), total=len(scores), desc="juiz"))
            contagem = Counter(v["apoio"] for v in vereditos)
            report["juiz"] = {k: round(v / len(vereditos), 4) for k, v in contagem.items()}
            report["juiz_n"] = len(vereditos)
            print(json.dumps(report["juiz"], indent=2, ensure_ascii=False))

    if args.ppl:
        report["ppl_pre_treino"] = {"sft": ppl_pre_treino(model, args.tokens_dir, args, device)}
        if args.base and Path(args.base).exists():
            base, _ = carregar(Path(args.base), args.tokenizer, device)
            report["ppl_pre_treino"]["base"] = ppl_pre_treino(base, args.tokens_dir, args, device)
            antes = report["ppl_pre_treino"]["base"]["val_pplx"]
            depois = report["ppl_pre_treino"]["sft"]["val_pplx"]
            report["esquecimento"] = {"ppl_antes": antes, "ppl_depois": depois,
                                      "fator": round(depois / antes, 3)}
            print(f"ppl pré-treino: base {antes} → sft {depois} "
                  f"(fator {depois / antes:.2f}; >1,3 = investigar)")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nrelatório em {out}")
    return report


if __name__ == "__main__":
    main()
