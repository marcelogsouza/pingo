#!/usr/bin/env python
"""SFT-3 — treino de instrução do Pingo sobre o dataset do SFT-1/SFT-2.

Entra `data/sft/final/{train,val}.jsonl` (gerado pelo gerador + validador) e o checkpoint
pré-treinado (`runs/pt-120m/final`); sai `runs/sft-120m/{ckpt.pt,best.pt,final/,log.jsonl}`.

O que é específico do SFT (e não do pré-treino):

- **Tokens de chat** `<|user|> <|assistant|> <|end|>` (decisão do usuário): o vocab vai de
  16.384 para 16.387 e as linhas novas do embedding nascem na **média** das antigas (não em
  ruído), para o modelo não "esquecer" o que já sabia.
- **Loss só na resposta**: os tokens do prompt (e o padding) viram `-100`; o modelo não é
  cobrado por prever o enunciado, só a resposta — mais o `<|end|>` que ele tem de aprender
  a emitir.
- **Mask de padding**: batch com `attention_mask` (pad à direita, posições 0..n-1 intactas
  para o RoPE) — `pingo/model.py` aceita `attention_mask` desde o SFT-3.
- **Misto ancorado/sem contexto**: a serialização vem de `pingo/sft.render()` — metade dos
  exemplos leva `Contexto: ...` (mini-RAG) e metade é só `Pergunta: ...`.
- **Épocas curtas** (default 3) com cosine: 30k pares é pouquíssimo comparado ao pré-treino,
  então lr menor (1e-4 → 1e-5) e warmup curto.

Uso:
  uv run python scripts/train_sft.py --tiny --max-steps 20 --out /tmp/sft-tiny   # dry-run CPU
  uv run python scripts/train_sft.py                                            # run real (GPU)
  uv run python scripts/train_sft.py --resume                                   # retoma
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pingo.config import PingoConfig
from pingo.model import PingoLM
from pingo.sft import chat_token_ids, ensure_chat_tokens, render

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- dados


@dataclass
class SFTData:
    """Exemplos já tokenizados: `ids` (prompt+resposta) e onde a resposta começa."""

    ids: list[np.ndarray] = field(default_factory=list)
    n_prompt: list[int] = field(default_factory=list)
    descartados: int = 0

    def __len__(self) -> int:
        return len(self.ids)


def encode_example(tokenizer, rec: dict, max_len: int) -> tuple[np.ndarray, int] | None:
    """(ids, nº de tokens do prompt) ou None se o exemplo não cabe em `max_len`."""
    prompt, resposta = render(rec)
    p_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    r_ids = tokenizer.encode(resposta, add_special_tokens=False).ids
    if not r_ids or len(p_ids) + len(r_ids) > max_len:
        return None
    return np.asarray(p_ids + r_ids, dtype=np.int32), len(p_ids)


def load_sft_data(path: Path, tokenizer, max_len: int, limit: int = 0) -> SFTData:
    data = SFTData()
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            enc = encode_example(tokenizer, rec, max_len)
            if enc is None:
                data.descartados += 1
                continue
            data.ids.append(enc[0])
            data.n_prompt.append(enc[1])
    return data


def collate(data: SFTData, idx: np.ndarray, pad_id: int, pad_to_multiple: int = 8):
    """Monta o batch: pad à direita, mask de atenção e labels só na resposta."""
    seqs = [data.ids[i] for i in idx]
    n_prompt = [data.n_prompt[i] for i in idx]
    longest = max(len(s) for s in seqs)
    length = math.ceil(longest / pad_to_multiple) * pad_to_multiple  # shape estável p/ compile
    b = len(seqs)
    x = np.full((b, length), pad_id, dtype=np.int64)
    mask = np.zeros((b, length), dtype=np.int64)
    labels = np.full((b, length), -100, dtype=np.int64)
    for i, (s, np_) in enumerate(zip(seqs, n_prompt)):
        x[i, :len(s)] = s
        mask[i, :len(s)] = 1
        # O modelo prevê o PRÓXIMO token: o label da posição `t` é o token `t+1`. A resposta
        # começa em `np_`, então ela é prevista a partir da posição `np_ - 1` (o token
        # <|assistant|> que fecha o prompt) até o penúltimo token (o <|end|> é o último label).
        labels[i, np_ - 1:len(s) - 1] = s[np_:]
    return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(labels)


# ---------------------------------------------------------------- modelo


def resize_embeddings(model: PingoLM, new_size: int) -> int:
    """Aumenta o embedding p/ os tokens de chat, **iniciando na média** dos existentes."""
    old = model.embed
    if new_size <= old.num_embeddings:
        return 0
    w = old.weight.data
    acrescimo = new_size - w.shape[0]
    novo = torch.empty(new_size, w.shape[1], dtype=w.dtype, device=w.device)
    novo[: w.shape[0]] = w
    novo[w.shape[0]:] = w.mean(dim=0, keepdim=True).expand(acrescimo, -1)
    emb = torch.nn.Embedding(new_size, w.shape[1], dtype=w.dtype, device=w.device)
    with torch.no_grad():
        emb.weight.copy_(novo)
    model.embed = emb
    model.lm_head.weight = emb.weight          # re-tie (embeddings compartilhados)
    model.lm_head.out_features = new_size
    model.config.vocab_size = new_size
    return acrescimo


def build_optimizer(model, args) -> torch.optim.AdamW:
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
    unwrapped = getattr(model, "_orig_mod", model)
    payload = {
        "step": step,
        "model": unwrapped.state_dict(),
        "opt": opt.state_dict(),
        "rng": rng_states(np_rng),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    if path.exists():
        path.replace(path.with_name("ckpt-prev.pt"))
    tmp.replace(path)


def autocast_ctx():
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda" else contextlib.nullcontext())


def batch_loss(model, x, mask, labels) -> tuple[torch.Tensor, int]:
    """Loss média por token de resposta (o `cross_entropy` já ignora -100)."""
    with autocast_ctx():
        out = model(x, attention_mask=mask, labels=labels)
    n_tok = int((labels != -100).sum())
    return out.loss, n_tok


@torch.no_grad()
def evaluate(model, data: SFTData, args) -> dict:
    model.eval()
    total_loss, total_tok, acertos = 0.0, 0, 0
    idx = np.arange(len(data))
    for start in range(0, len(idx), args.batch_size):
        x, mask, labels = collate(data, idx[start:start + args.batch_size], args.pad_id)
        x, mask, labels = x.to(device), mask.to(device), labels.to(device)
        with autocast_ctx():
            out = model(x, attention_mask=mask, labels=labels)
        n = int((labels != -100).sum())
        if n:
            total_loss += out.loss.float().item() * n
            total_tok += n
            pred = out.logits.argmax(-1)          # labels já deslocados no collate
            acertos += int(((pred == labels) & (labels != -100)).sum())
    model.train()
    loss = total_loss / max(total_tok, 1)
    return {"loss": loss, "ppl": math.exp(min(loss, 20)), "acc": acertos / max(total_tok, 1),
            "tokens": total_tok}


# ---------------------------------------------------------------- CLI / run


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "sft" / "final")
    p.add_argument("--base", type=Path, default=ROOT / "runs" / "pt-120m" / "final")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer" / "tokenizer.json")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "sft-120m")
    p.add_argument("--tiny", action="store_true", help="config minúscula p/ dry-run em CPU")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=None, help="sobrepõe --epochs")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--ckpt-min", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit-train", type=int, default=0, help="só as N primeiras linhas (teste)")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    global device
    if args.device:
        device = args.device
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    criados = ensure_chat_tokens(tokenizer)
    ids_chat = chat_token_ids(tokenizer)
    pad_id = tokenizer.token_to_id("<|pad|>")
    args.pad_id = pad_id

    train = load_sft_data(args.data_dir / "train.jsonl", tokenizer, args.max_len,
                          args.limit_train)
    val = load_sft_data(args.data_dir / "val.jsonl", tokenizer, args.max_len)
    if not len(train) or not len(val):
        raise SystemExit(f"dataset vazio em {args.data_dir} — rode gen_sft.py e validate_sft.py")
    print(f"dataset: {len(train)} treino / {len(val)} val | descartados por tamanho: "
          f"{train.descartados} / {val.descartados} | tokens de chat novos: {criados}")

    cfg = (PingoConfig(vocab_size=tokenizer.get_vocab_size(), hidden_size=64,
                       num_hidden_layers=2, num_attention_heads=4, intermediate_size=128,
                       max_position_embeddings=args.max_len)
           if args.tiny else PingoConfig.from_pretrained(args.base))
    model = (PingoLM(cfg) if args.tiny
             else PingoLM.from_pretrained(args.base, config=cfg))
    acrescentados = resize_embeddings(model, tokenizer.get_vocab_size())
    if acrescentados:
        print(f"embeddings: +{acrescentados} tokens (init na média dos {cfg.vocab_size - acrescentados})")
    model = model.to(device)
    assert model.config.max_position_embeddings >= args.max_len
    if args.compile:
        model = torch.compile(model)
    opt = build_optimizer(model, args)

    steps_por_epoca = math.ceil(len(train) / (args.batch_size * args.accum_steps))
    max_steps = args.max_steps or max(1, round(args.epochs * steps_por_epoca))
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out / "tokenizer.json"))   # o SFT model viaja com o seu tokenizer
    (args.out / "chat_tokens.json").write_text(json.dumps(ids_chat, indent=2))

    def ordem_da_epoca(epoca: int) -> np.ndarray:
        """Ordem determinística dos exemplos por época (resume não precisa guardá-la)."""
        return np.random.default_rng([args.seed, epoca]).permutation(len(train))

    step = 0
    ckpt_path = args.out / "ckpt.pt"
    if args.resume:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume sem {ckpt_path}")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        getattr(model, "_orig_mod", model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        restore_rng(ck["rng"], np_rng)
        step = ck["step"]
        print(f"resume: retomando do step {step}/{max_steps}")

    logf = (args.out / "log.jsonl").open("a", encoding="utf-8")
    t0 = time.time()

    def log(**fields):
        logf.write(json.dumps({"step": step, "t": round(time.time() - t0, 1)} | fields) + "\n")
        logf.flush()

    print(f"modelo {sum(p.numel() for p in model.parameters()):,} params | device {device} | "
          f"{steps_por_epoca} steps/época | max_steps {max_steps} | "
          f"batch {args.batch_size}×{args.accum_steps} = "
          f"{args.batch_size * args.accum_steps} exemplos/step")

    last_log, last_ckpt, loss_sum, tok_sum = time.time(), time.time(), 0.0, 0
    best_val = math.inf
    try:
        while step < max_steps:
            lr = lr_at(step, args, max_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            for j in range(args.accum_steps):
                pos = step * args.accum_steps + j
                epoca = pos // steps_por_epoca
                ordem = ordem_da_epoca(epoca)
                inicio = (pos % steps_por_epoca) * args.batch_size
                idx = ordem[inicio:inicio + args.batch_size]
                if len(idx) < 2:  # fim da época: completa com o começo dela
                    idx = np.concatenate([idx, ordem[: args.batch_size - len(idx)]])
                x, mask, labels = collate(train, idx, pad_id)
                x, mask, labels = x.to(device), mask.to(device), labels.to(device)
                loss, n_tok = batch_loss(model, x, mask, labels)
                (loss / args.accum_steps).backward()
                loss_sum += loss.float().item() / args.accum_steps
                tok_sum += n_tok
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_steps == 0:
                agora = time.time()
                log(train_loss=round(loss_sum / args.log_steps, 4), lr=round(lr, 7),
                    ex_s=round(args.batch_size * args.accum_steps * args.log_steps
                               / (agora - last_log), 1),
                    resp_tok_s=int(tok_sum / (agora - last_log)))
                print(f"step {step:5d}/{max_steps} | loss {loss_sum / args.log_steps:.4f} | "
                      f"lr {lr:.1e} | {args.batch_size * args.accum_steps * args.log_steps / (agora - last_log):.1f} ex/s")
                loss_sum, tok_sum, last_log = 0.0, 0, agora

            if step % args.eval_steps == 0 or step == max_steps:
                m = evaluate(model, val, args)
                log(val_loss=round(m["loss"], 4), val_ppl=round(m["ppl"], 3),
                    val_acc=round(m["acc"], 4))
                print(f"step {step:5d} | val loss {m['loss']:.4f} | ppl {m['ppl']:.2f} | "
                      f"acc {m['acc']:.3f}")
                if m["loss"] < best_val:
                    best_val = m["loss"]
                    save_ckpt(args.out / "best.pt", model, opt, step, args, np_rng)
                    print(f"step {step:5d} | novo best (val {m['loss']:.4f})")

            if time.time() - last_ckpt >= args.ckpt_min * 60:
                save_ckpt(ckpt_path, model, opt, step, args, np_rng)
                last_ckpt = time.time()
                print(f"step {step:5d} | checkpoint em {ckpt_path}")
    except KeyboardInterrupt:
        save_ckpt(ckpt_path, model, opt, step, args, np_rng)
        print(f"\ninterrompido no step {step} — retome com --resume")
        raise
    finally:
        logf.close()

    save_ckpt(ckpt_path, model, opt, step, args, np_rng)
    final = args.out / "final"
    unwrapped = getattr(model, "_orig_mod", model)
    unwrapped.save_pretrained(final)
    tokenizer.save(str(final / "tokenizer.json"))
    m = evaluate(model, val, args)
    print(f"fim: {step} steps | val loss {m['loss']:.4f} | ppl {m['ppl']:.2f} | "
          f"acc {m['acc']:.3f} | salvo em {final}")
    return {"step": step, "val": m, "best_val": best_val}


if __name__ == "__main__":
    run(parse_args())
