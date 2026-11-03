#!/usr/bin/env python
"""Ppl de um checkpoint em qualquer `--tokens-dir` (a régua para comparar corpora/runs).

Usa o MESMO harness do treino (`train.evaluate`: batch 16 × ctx 2048 × 20 iters, rng fixo),
então os números são comparáveis entre modelos — e batem com o `val_loss` que o run loga.

Uso:
  uv run python scripts/eval_ppl.py --model runs/pt-120m-wiki/final \
      --tokens data/tokens data/tokens/wiki-16k
  uv run python scripts/eval_ppl.py --model runs/wiki-120m/final runs/pt-120m/final \
      --tokens data/tokens/wiki-16k --device cpu     # sem GPU (lento, mas roda)
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pingo.data import load_tokens  # noqa: E402
from pingo.model import PingoLM  # noqa: E402

_spec = importlib.util.spec_from_file_location("pingo_train", ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
sys.modules["pingo_train"] = train
_spec.loader.exec_module(train)


class _Args:
    """Só o que o `train.evaluate` lê — os defaults do treino."""

    def __init__(self, batch_size: int, ctx: int, eval_iters: int):
        self.batch_size, self.ctx, self.eval_iters = batch_size, ctx, eval_iters


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, nargs="+", required=True,
                    help="pasta com config.json + model.safetensors (ex.: runs/x/final)")
    ap.add_argument("--tokens", type=Path, nargs="+", required=True, help="diretório(s) de tokens")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda se houver)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--eval-iters", type=int, default=20)
    args = ap.parse_args(argv)

    train.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    harness = _Args(args.batch_size, args.ctx, args.eval_iters)
    vals = {str(d): load_tokens(d) for d in args.tokens}
    print(f"device {train.device} | batch {args.batch_size} × ctx {args.ctx} × "
          f"{args.eval_iters} iters (o harness do treino)", flush=True)

    for pasta in args.model:
        if not (pasta / "config.json").exists():
            sys.exit(f"{pasta} não tem config.json — aponte para um `<run>/final`")
        modelo = PingoLM.from_pretrained(pasta).to(train.device)
        linha = [f"{str(pasta):34}"]
        for rotulo, dados in vals.items():
            vl = train.evaluate(modelo, dados, harness)
            linha.append(f"{rotulo}: loss {vl:.4f} ppl {math.exp(vl):6.2f}")
        print("  ".join(linha), flush=True)
        del modelo
        if train.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
