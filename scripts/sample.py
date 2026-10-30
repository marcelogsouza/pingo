#!/usr/bin/env python
"""Passo 7 — avaliação do Pingo pré-treinado.

Três coisas, todas medidas no checkpoint `final/` do Passo 6:

1. **Perplexidade no val** (o split `val` inteiro nunca visto no treino): loss média e
   ppl, em bits/token também. Serve de régua absoluta — é o número que resume "o quanto
   o modelo entendeu de pt".
2. **Amostras qualitativas** de geração (greedy e com amostragem), incluindo
   `--repetition-penalty` p/ matar o loop de repetição do greedy.
3. **Benchmark do KV cache**: geração com cache (prefill + 1 token/step, custo O(ctx))
   contra re-forward completo da sequência a cada token (O(ctx²)). O critério do plano é
   **10×+** em contexto longo — em ctx curto a diferença é pequena.

Uso:
  uv run python scripts/sample.py                                  # tudo (GPU)
  uv run python scripts/sample.py --no-bench --max-new-tokens 60   # só ppl + amostras
  uv run python scripts/sample.py --prompt "Era uma vez" --temperature 0.9 --top-p 0.95
  uv run python scripts/sample.py --model runs/smoke-120m/final --no-ppl   # comparar runs
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pingo.data import load_tokens  # noqa: E402
from pingo.cache import PingoCache  # noqa: E402
from pingo.generate import generate  # noqa: E402
from pingo.model import PingoLM  # noqa: E402

def _autocast(device: str):
    """bf16 na GPU (como no treino); em CPU, sem autocast (fp32 já é o caminho nativo)."""
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


DEFAULT_PROMPTS = [
    "O Brasil é um país",
    "A empresa anunciou hoje que",
    "No ano de 2024, o governo federal",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=ROOT / "runs" / "pt-120m" / "final",
                   help="dir com config.json + model.safetensors (HF)")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer" / "tokenizer.json")
    p.add_argument("--tokens-dir", type=Path, default=ROOT / "data" / "tokens")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--seed", type=int, default=13)

    g = p.add_argument_group("geração")
    g.add_argument("--prompt", action="append", dest="prompts",
                   help="pode repetir; default: frases fixas de exemplo")
    g.add_argument("--max-new-tokens", type=int, default=100)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=None)
    g.add_argument("--repetition-penalty", type=float, default=1.15)
    g.add_argument("--no-greedy", action="store_true", help="pula a amostra greedy")

    a = p.add_argument_group("perplexidade")
    a.add_argument("--ppl", default=True, action=argparse.BooleanOptionalAction,
                   help="mede perplexidade no split val (--no-ppl desliga)")
    a.add_argument("--ppl-iters", type=int, default=25, help="batches de val")
    a.add_argument("--ppl-batch-size", type=int, default=8)
    a.add_argument("--ctx", type=int, default=2048)

    b = p.add_argument_group("benchmark do KV cache")
    b.add_argument("--bench", default=True, action=argparse.BooleanOptionalAction)
    b.add_argument("--bench-contexts", default="512,1024",
                   help="contextos de prompt a comparar (lista separada por vírgula)")
    b.add_argument("--bench-batches", default="1,16",
                   help="batch sizes a comparar (1 = latência/demo; 16 = serving)")
    b.add_argument("--bench-new-tokens", type=int, default=32)

    p.add_argument("--out", type=Path, default=None, help="salva o relatório em JSON")
    return p.parse_args(argv)


# ---------------------------------------------------------------- perplexidade

@torch.no_grad()
def perplexity(model, data, args, device) -> dict:
    """Loss média no val (rng fixo) + ppl + bits/token."""
    rng = np.random.default_rng(20_260_101)  # mesma régua do train.py → comparável
    total, ntokens = 0.0, 0
    model.eval()
    for _ in range(args.ppl_iters):
        x, y = data.get_batch("val", args.ppl_batch_size, args.ctx, rng)
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        with _autocast(device):
            loss = model(x, labels=y).loss
        total += loss.float().item() * y.numel()
        ntokens += y.numel()
    loss = total / ntokens
    return {
        "val_loss": round(loss, 4),
        "val_pplx": round(math.exp(min(loss, 20)), 2),
        "bits_per_token": round(loss / math.log(2), 4),
        "tokens_avaliados": ntokens,
        "batches": args.ppl_iters,
    }


# ---------------------------------------------------------------- geração

@torch.no_grad()
def generate_naive(model, prompt_ids: torch.Tensor, n_new: int) -> torch.Tensor:
    """Decode sem cache: re-forward da sequência inteira a cada token (O(ctx²))."""
    ids = prompt_ids
    for _ in range(n_new):
        if ids.shape[1] >= model.config.max_position_embeddings:
            break
        logits = model(ids).logits[:, -1]
        ids = torch.cat((ids, logits.argmax(-1, keepdim=True)), dim=-1)  # greedy
    return ids


@torch.no_grad()
def generate_cached(model, prompt_ids: torch.Tensor, n_new: int) -> torch.Tensor:
    """Decode com KV cache (prefill + 1 token/step), batched — o cenário de serving."""
    cache = PingoCache(model.config, batch_size=prompt_ids.shape[0])
    ids = prompt_ids
    logits = model(ids, cache=cache).logits[:, -1]
    out = [ids]
    for _ in range(n_new):
        if cache.cur_len >= model.config.max_position_embeddings:
            break
        nxt = logits.argmax(-1, keepdim=True)  # greedy: mesmo token p/ todas as linhas
        out.append(nxt)
        logits = model(nxt, cache=cache).logits[:, -1]
    return torch.cat(out, dim=1)


def _time(fn, model, prompt, n_new, device) -> float:
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), _autocast(device):
        ids = fn(model, prompt, n_new)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, ids


def bench_kv_cache(model, vocab_prompt: list[int], ctx_len: int, n_new: int,
                   batch_size: int, device) -> dict:
    """Compara cached vs naive (greedy, mesmo prompt em todas as linhas do batch).

    OB: em batch=1 na GPU o decode é limitado por latência de kernel, não por FLOPs —
    recalcular tudo sai quase de graça e o cache "perde". O ganho aparece com batch
    (serving) ou ctx bem longo; por isso medimos os dois.
    """
    # precisa sobrar espaço p/ gerar: o cache para no max_position_embeddings
    max_ctx = model.config.max_position_embeddings - n_new
    clipped = ctx_len > max_ctx
    ctx_len = min(ctx_len, max_ctx)
    prompt = torch.tensor([vocab_prompt[:ctx_len]], dtype=torch.long, device=device)
    prompt = prompt.expand(batch_size, -1).contiguous()
    out = {"ctx": int(prompt.shape[1]), "batch": batch_size, "ctx_ajustado": clipped}
    for name, fn in (("naive", generate_naive), ("cache", generate_cached)):
        dt, ids = _time(fn, model, prompt, n_new, device)
        out[name] = {
            "segundos": round(dt, 3),
            # tok/s conta apenas os tokens novos (decode), igual p/ os dois caminhos
            "tok_s": round(batch_size * n_new / dt, 1) if dt > 0 else None,
            "primeiros_ids": ids[0, prompt.shape[1]:][:16].tolist(),
        }
    out["speedup"] = round(out["naive"]["segundos"] / out["cache"]["segundos"], 2)
    out["mesmos_tokens"] = out["naive"]["primeiros_ids"] == out["cache"]["primeiros_ids"]
    for k in ("naive", "cache"):
        out[k].pop("primeiros_ids")
    return out


# ---------------------------------------------------------------- main

def main(argv=None) -> dict:
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"modelo {args.model} | device {device}")
    model = PingoLM.from_pretrained(args.model).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    report: dict = {"model": str(args.model), "params": n_params, "device": device}
    print(f"{n_params:,} params")

    if args.ppl:
        data = load_tokens(args.tokens_dir)
        report["perplexity"] = perplexity(model, data, args, device)
        pp = report["perplexity"]
        print(f"val: loss {pp['val_loss']} | ppl {pp['val_pplx']} | "
              f"{pp['bits_per_token']} bits/token ({pp['tokens_avaliados']:,} tokens)")

    tok = Tokenizer.from_file(str(args.tokenizer))
    prompts = args.prompts or DEFAULT_PROMPTS
    report["samples"] = []
    print("\n--- amostras ---")
    for prompt in prompts:
        ids = tok.encode(prompt, add_special_tokens=False).ids
        for label, kw in [("greedy", dict(temperature=0.0)),
                          ("sample", dict(temperature=args.temperature, top_k=args.top_k,
                                          top_p=args.top_p,
                                          repetition_penalty=args.repetition_penalty))]:
            if label == "greedy" and args.no_greedy:
                continue
            out = generate(model, ids, max_new_tokens=args.max_new_tokens,
                           seed=args.seed, **kw)
            text = tok.decode(out[0].tolist(), skip_special_tokens=True)
            report["samples"].append({"prompt": prompt, "modo": label, "texto": text})
            print(f"\n[{label}] {prompt!r}\n{text}")

    if args.bench:
        ctxs = [int(c) for c in args.bench_contexts.split(",") if c.strip()]
        seq = tok.encode(" ".join(prompts), add_special_tokens=False).ids
        if len(seq) < max(ctxs):
            # completa com texto real do val p/ chegar no maior contexto pedido
            data = load_tokens(args.tokens_dir)
            rng = np.random.default_rng(args.seed)
            extra, _ = data.get_batch("val", 1, max(ctxs) - len(seq) + 8, rng)
            seq = (seq + extra[0].tolist())[:max(ctxs)]
        batches = [int(b) for b in args.bench_batches.split(",") if b.strip()]
        report["kv_cache"] = []
        print(f"\n--- benchmark do KV cache (greedy, {args.bench_new_tokens} tokens novos) ---")
        print(f"{'ctx':>5} {'B':>3} | {'naive':>9} | {'cache':>9} | {'speedup':>7} | same")
        for ctx_len in ctxs:
            for bs in batches:
                r = bench_kv_cache(model, seq, ctx_len, args.bench_new_tokens, bs, device)
                report["kv_cache"].append(r)
                print(f"{r['ctx']:5d} {bs:3d} | {r['naive']['segundos']:8.2f}s | "
                      f"{r['cache']['segundos']:8.2f}s | {r['speedup']:6.2f}× | {r['mesmos_tokens']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"\nrelatório salvo em {args.out}")
    return report


if __name__ == "__main__":
    main()
