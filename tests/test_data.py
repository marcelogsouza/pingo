"""Testes do loader memmap do Passo 4 (`pingo/data.py`).

Usa shards sintéticos (valores em `arange` → alvo é exatamente `x+1`) e um teste
de integração com os tokens reais, pulado se `data/tokens/` não existir.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pingo.data import DTYPE, load_tokens

TOK_ROOT = Path(__file__).resolve().parents[1] / "data" / "tokens"


@pytest.fixture()
def tokens_dir(tmp_path):
    for i, n in enumerate([5000, 3000, 1200]):
        np.arange(n, dtype=DTYPE).tofile(tmp_path / f"train-{i:04d}.bin")
    np.arange(2000, dtype=DTYPE).tofile(tmp_path / "val-0000.bin")
    (tmp_path / "meta.json").write_text(json.dumps(
        {"train_shards": 3, "train_tokens": 9200, "val_shards": 1, "val_tokens": 2000,
         "vocab_size": 16384}))
    return tmp_path


def test_load_tokens(tokens_dir):
    data = load_tokens(tokens_dir)
    assert [len(s) for s in data.train] == [5000, 3000, 1200]
    assert [len(s) for s in data.val] == [2000]
    assert data.meta["train_tokens"] == 9200


def test_batch_shapes_e_shift(tokens_dir):
    data = load_tokens(tokens_dir)
    x, y = data.get_batch("train", batch_size=16, ctx=64, rng=np.random.default_rng(7))
    assert x.shape == y.shape == (16, 64)
    assert y.dtype == x.dtype == np.int64
    # shards sintéticos em arange: janela y é x deslocada de 1 no mesmo shard
    assert (y - x == 1).all()
    assert np.array_equal(y[:, :-1], x[:, 1:])


def test_batch_determinismo(tokens_dir):
    data = load_tokens(tokens_dir)
    a1, b1 = data.get_batch("train", 8, 32, np.random.default_rng(42))
    a2, b2 = data.get_batch("train", 8, 32, np.random.default_rng(42))
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    a3, _ = data.get_batch("train", 8, 32, np.random.default_rng(43))
    assert not np.array_equal(a1, a3)


def test_batch_janela_respeita_shards_curto(tokens_dir):
    # shard de 1200 tokens: janela de 64 precisa caber (off <= len - ctx - 1)
    data = load_tokens(tokens_dir)
    x, y = data.get_batch("train", batch_size=64, ctx=64, rng=np.random.default_rng(3))
    assert x.max() < 5000  # só valores dos shards sintéticos
    assert (y - x == 1).all()


@pytest.mark.skipif(not (TOK_ROOT / "meta.json").exists(),
                    reason="tokens reais ainda não binarizados nesta máquina")
def test_integracao_tokens_reais():
    data = load_tokens(TOK_ROOT)
    meta = data.meta
    assert meta["train_shards"] == len(data.train)
    assert sum(len(s) for s in data.train) == meta["train_tokens"]
    x, y = data.get_batch("val", batch_size=4, ctx=128, rng=np.random.default_rng(1))
    assert x.shape == (4, 128)
    assert ((x >= 0) & (x < meta["vocab_size"])).all()
    assert np.array_equal(y[:, :-1], x[:, 1:])
