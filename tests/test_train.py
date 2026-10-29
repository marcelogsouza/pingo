"""Testes do loop de pré-treino (Passo 6) — config tiny, tokens sintéticos, CPU.

Critério do plano: **`--resume` reproduz exatamente a mesma curva** ao retomar.
Teste central: run contínuo de 6 steps == run de 3 steps + resume até 6
(parâmetros bit a bit idênticos e mesma val loss final).
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("pingo_train", ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

from pingo.config import PingoConfig  # noqa: E402


@pytest.fixture()
def tokens_dir(tmp_path):
    for i, n in enumerate([5000, 3000, 1200]):
        np.arange(n, dtype=np.uint16).tofile(tmp_path / f"train-{i:04d}.bin")
    np.arange(2000, dtype=np.uint16).tofile(tmp_path / "val-0000.bin")
    (tmp_path / "meta.json").write_text(json.dumps(
        {"train_shards": 3, "train_tokens": 9200, "val_shards": 1,
         "val_tokens": 2000, "vocab_size": 16384}))
    return tmp_path


def base_args(tokens_dir, out, **over):
    argv = ["--tokens-dir", str(tokens_dir), "--out", str(out), "--tiny",
            "--device", "cpu",
            "--ctx", "32", "--batch-size", "4", "--accum-steps", "2",
            "--warmup-steps", "2", "--eval-steps", "6", "--eval-iters", "2",
            "--log-steps", "3", "--ckpt-min", "1000000", "--seed", "7"]
    for k, v in over.items():  # flags bool (resume=True) vs opções com valor (max_steps=6)
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            argv.append(flag)
        else:
            argv += [flag, str(v)]
    return train.parse_args(argv)


# ---------------------------------------------------------------- lr schedule

def test_lr_warmup_cosine():
    args = train.parse_args([])
    args.warmup_steps, args.lr, args.min_lr = 4, 1.0, 0.1
    curve = [train.lr_at(s, args, 14) for s in range(15)]
    assert curve[0] == 0.0
    assert curve[2] == pytest.approx(0.5)              # warmup linear
    assert curve[4] == pytest.approx(1.0)              # pico no fim do warmup
    assert curve[9] == pytest.approx(0.55, abs=1e-6)   # meio do cosine
    assert curve[13] < curve[9] < curve[4]             # decaimento monotônico
    assert train.lr_at(99, args, 14) == args.min_lr


def test_optimizer_grupos_de_decay():
    model = train.PingoLM(PingoConfig(hidden_size=32, num_hidden_layers=1,
                                      num_attention_heads=4, intermediate_size=64))
    args = train.parse_args([])
    opt = train.build_optimizer(model, args)
    g_decay, g_nodecay = opt.param_groups
    assert g_decay["weight_decay"] == args.weight_decay == 0.1
    assert g_nodecay["weight_decay"] == 0.0
    ids_nodecay = {id(p) for p in g_nodecay["params"]}
    for p in (model.embed.weight, model.norm.weight,
              model.blocks[0].norm_attn.weight, model.blocks[0].norm_mlp.weight):
        assert id(p) in ids_nodecay, "norm/embedding não deveriam ter decay"
    ids_decay = {id(p) for p in g_decay["params"]}
    for p in (model.blocks[0].attn.q_proj.weight, model.blocks[0].mlp.down_proj.weight):
        assert id(p) in ids_decay
    # cobertura total sem sobreposição
    assert len(g_decay["params"]) + len(g_nodecay["params"]) == \
        sum(1 for _ in model.parameters())


def test_optimizer_com_pesos_de_compile():
    """`--compile` envolve o modelo: os nomes viram `_orig_mod.*` e o embed/norm
    cairia no grupo com weight decay se o unwrap for esquecido (regressão)."""
    model = train.PingoLM(PingoConfig(hidden_size=32, num_hidden_layers=1,
                                      num_attention_heads=4, intermediate_size=64))

    class Wrapper(torch.nn.Module):  # imita o que torch.compile faz com _orig_mod
        def __init__(self, m):
            super().__init__()
            self._orig_mod = m

        def forward(self, *a, **kw):
            return self._orig_mod(*a, **kw)

    wrapped = Wrapper(model)
    assert any(n.startswith("_orig_mod.") for n, _ in wrapped.named_parameters())
    args = train.parse_args([])
    g_decay, g_nodecay = train.build_optimizer(wrapped, args).param_groups
    ids_nodecay = {id(p) for p in g_nodecay["params"]}
    for p in (model.embed.weight, model.norm.weight,
              model.blocks[0].norm_attn.weight, model.blocks[0].norm_mlp.weight):
        assert id(p) in ids_nodecay, "norm/embedding sem decay mesmo com compile"
    assert len(g_decay["params"]) + len(g_nodecay["params"]) == \
        sum(1 for _ in model.parameters())


# ---------------------------------------------------------------- resume

def test_resume_reproduz_run_contínuo(tokens_dir, tmp_path):
    outA, outB = tmp_path / "A", tmp_path / "B"

    resA = train.run(base_args(tokens_dir, outA, max_steps=6))     # 1 run contínuo

    train.run(base_args(tokens_dir, outB, max_steps=3))            # "noite 1"
    resB = train.run(base_args(tokens_dir, outB, max_steps=6, resume=True))  # "noite 2"

    assert resB["step"] == 6
    assert resB["val_loss"] == pytest.approx(resA["val_loss"], abs=1e-9)
    sa, sb = resA["model"].state_dict(), resB["model"].state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"peso {k} divergiu após resume"


def test_ckpt_payload_e_logs(tokens_dir, tmp_path):
    out = tmp_path / "run"
    train.run(base_args(tokens_dir, out, max_steps=3))
    ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 3
    for key in ("model", "opt", "rng", "args"):
        assert key in ck
    assert set(ck["rng"]) == {"python", "numpy", "torch", "torch_cuda"}

    recs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    losses = [r["train_loss"] for r in recs if "train_loss" in r]
    assert len(losses) == 1  # log-steps=3 → step 3
    assert losses[0] == pytest.approx(math.log(16384), abs=1.0)  # init ~ ln V
    val = [r for r in recs if "val_loss" in r]
    assert not val  # eval-steps=6 > 3 steps: sem avaliação ainda

    # max_steps default = época sobre meta.json (9200 / (4·32·2) = 36)
    args = base_args(tokens_dir, tmp_path / "x")
    data = train.load_tokens(tokens_dir)
    glob = args.batch_size * args.ctx * args.accum_steps
    assert (args.max_steps or max(1, round(data.meta["train_tokens"] / glob))) == 36
