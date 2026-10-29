#!/usr/bin/env python
"""Passo 6 — loop de pré-treino do Pingo (estilo nanoGPT, retomável por design).

Receita do plano: AdamW (wd 0.1, sem decay em RMSNorm/embeddings), warmup linear
+ cosine, grad clip 1.0, autocast **bf16**, batch global ~0.5M tokens via
grad accumulation (~5.200 steps p/ 2,75B tokens). Checkpoints a cada ~15 min
(modelo + otimizador + passo + RNGs) → treino fatiável em noites; `--resume`
retoma EXATAMENTE de onde parou (mesma sequência de batches e de lr).

Uso:
  uv run python scripts/train.py                                   # run completo (GPU)
  uv run python scripts/train.py --tiny --max-steps 30             # dry-run CPU
  uv run python scripts/train.py --resume                          # retoma runs/pt-120m/ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pingo.config import PingoConfig
from pingo.data import TokenData, load_tokens
from pingo.model import PingoLM

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens-dir", type=Path, default=ROOT / "data" / "tokens")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "pt-120m")
    p.add_argument("--config", type=Path, default=None, help="PingoConfig json (default: Pingo-120M)")
    p.add_argument("--tiny", action="store_true", help="config minúscula p/ dry-run em CPU")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=16, help="micro-batch (B)")
    p.add_argument("--accum-steps", type=int, default=16, help="grad accumulation (B global = 16×16 = 256 seqs ≈ 0,52M tok)")
    p.add_argument("--max-steps", type=int, default=None, help="default: 1 época sobre os tokens do meta.json")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=400)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-steps", type=int, default=250, help="a cada quantos steps avaliar o val")
    p.add_argument("--eval-iters", type=int, default=20, help="batches de val por avaliação")
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--ckpt-min", type=float, default=15.0, help="checkpoint a cada N minutos de parede")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto — cuda se houver)")
    p.add_argument("--compile", action="store_true", help="torch.compile p/ ganho na GPU")
    p.add_argument("--resume", action="store_true", help="retoma de <out>/ckpt.pt")
    return p.parse_args(argv)


def make_config(args: argparse.Namespace) -> PingoConfig:
    if args.tiny:
        return PingoConfig(vocab_size=16_384, hidden_size=64, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=128,
                           max_position_embeddings=args.ctx)
    if args.config is not None:
        return PingoConfig.from_pretrained(args.config)
    return PingoConfig()


def build_optimizer(model, args) -> torch.optim.AdamW:
    # sem weight decay: RMSNorm (1-D) e embeddings (protege a tabela de tokens)
    # (unwrap `_orig_mod` p/ --compile: senão os nomes vêm como `_orig_mod.embed.weight`)
    unwrapped = getattr(model, "_orig_mod", model)
    no_decay = {n for n, p in unwrapped.named_parameters() if p.ndim < 2 or n == "embed.weight"}
    groups = [
        {"params": [p for n, p in unwrapped.named_parameters() if n not in no_decay],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in unwrapped.named_parameters() if n in no_decay],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2))


def lr_at(step: int, args, max_steps: int) -> float:
    """Warmup linear (step=warmup atinge --lr) + cosine até --min-lr em max_steps."""
    if step < args.warmup_steps:
        return args.lr * step / max(1, args.warmup_steps)
    if step >= max_steps:
        return args.min_lr
    p = (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))


def rng_states(np_rng: np.random.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np_rng.bit_generator.state,
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict, np_rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    np_rng.bit_generator.state = state["numpy"]
    torch.set_rng_state(state["torch"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])


def save_ckpt(path: Path, model, opt, step: int, args, np_rng) -> None:
    # se --compile estiver ligado, salva as chaves sem o prefixo `_orig_mod.`
    # (ckpt idêntico compilado ou não → --resume independe da flag)
    unwrapped = getattr(model, "_orig_mod", model)
    payload = {
        "step": step,
        "model": unwrapped.state_dict(),
        "opt": opt.state_dict(),
        "rng": rng_states(np_rng),
        "args": vars(args) | {"out": str(args.out)},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    if path.exists():  # rotação simples: ckpt.pt vira ckpt-prev.pt (2 gerações)
        path.replace(path.with_name("ckpt-prev.pt"))
    tmp.replace(path)


def evaluate(model, data: TokenData, args) -> float:
    """Val loss média sobre `--eval-iters` batches com rng FIXO (números comparáveis)."""
    eval_rng = np.random.default_rng(20_260_101)
    model.eval()
    with torch.no_grad():
        total = 0.0
        for _ in range(args.eval_iters):
            x, y = data.get_batch("val", args.batch_size, args.ctx, eval_rng)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                loss = model(x, labels=y).loss
            total += loss.float().item()
    model.train()
    return total / args.eval_iters


def run(args: argparse.Namespace) -> dict:
    global device
    if args.device:
        device = args.device
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    data = load_tokens(args.tokens_dir)
    cfg = make_config(args)
    assert cfg.max_position_embeddings >= args.ctx, "ctx maior que o contexto do modelo"
    model = PingoLM(cfg).to(device)
    if args.compile:
        model = torch.compile(model)
    opt = build_optimizer(model, args)

    global_tokens = args.batch_size * args.ctx * args.accum_steps
    max_steps = args.max_steps or max(1, round(data.meta["train_tokens"] / global_tokens))

    step = 0
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "ckpt.pt"
    if args.resume:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume sem {ckpt_path}")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        getattr(model, "_orig_mod", model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        restore_rng(ck["rng"], np_rng)
        step = ck["step"]
        print(f"resume em {ckpt_path}: retomando do step {step}")

    # referência p/ re-tocar lr por step (param_groups[0]/[1] herdam lr do setup)
    logf = (args.out / "log.jsonl").open("a", encoding="utf-8")

    def log(**fields):
        rec = {"step": step, "t": round(time.time() - t0, 1)} | fields
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    t0 = time.time()
    last_ckpt, last_log, loss_sum = t0, t0, 0.0
    tokens_seen = step * global_tokens
    print(f"modelo {sum(p.numel() for p in model.parameters()):,} params | "
          f"device {device} | batch global {global_tokens:,} tok | max_steps {max_steps}")

    try:
        while step < max_steps:
            lr = lr_at(step, args, max_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            for _ in range(args.accum_steps):
                x, y = data.get_batch("train", args.batch_size, args.ctx, np_rng)
                x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    loss = model(x, labels=y).loss / args.accum_steps
                loss.backward()
                loss_sum += loss.float().item()  # já dividido por accum → soma ≈ média por step
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tokens_seen += global_tokens

            if step % args.log_steps == 0:
                now = time.time()
                tps = global_tokens * args.log_steps / (now - last_log)
                avg = loss_sum / args.log_steps
                log(train_loss=round(avg, 4), lr=round(lr, 6), toks_s=int(tps),
                    tokens_seen=tokens_seen)
                print(f"step {step:5d}/{max_steps} | loss {avg:.4f} | lr {lr:.2e} | "
                      f"{tps/1e3:.0f}k tok/s | {tokens_seen/1e9:.2f}B tok")
                loss_sum, last_log = 0.0, now

            if step % args.eval_steps == 0:
                vl = evaluate(model, data, args)
                log(val_loss=round(vl, 4), val_pplx=round(math.exp(min(vl, 20)), 1))
                print(f"step {step:5d} | val loss {vl:.4f} | ppl {math.exp(min(vl, 20)):.1f}")

            if time.time() - last_ckpt >= args.ckpt_min * 60:
                save_ckpt(ckpt_path, model, opt, step, args, np_rng)
                last_ckpt = time.time()
                print(f"step {step:5d} | checkpoint salvo em {ckpt_path}")
    except KeyboardInterrupt:
        save_ckpt(ckpt_path, model, opt, step, args, np_rng)
        print(f"\ninterrompido — ckpt salvo no step {step}; retome com --resume")
        raise
    finally:
        logf.close()

    # fim: último ckpt + modelo final em formato HF (scaffolding do Passo 7)
    save_ckpt(ckpt_path, model, opt, step, args, np_rng)
    final = args.out / "final"
    unwrapped = getattr(model, "_orig_mod", model)  # torch.compile
    unwrapped.save_pretrained(final)
    vl = evaluate(model, data, args)
    print(f"treino completo ({step} steps, {tokens_seen/1e9:.2f}B tok) | "
          f"val loss {vl:.4f} | ppl {math.exp(min(vl, 20)):.1f} | salvo em {final}")
    return {"step": step, "val_loss": vl, "model": model}


if __name__ == "__main__":
    run(parse_args())
