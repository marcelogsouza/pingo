"""Testes do Passo 7 (scripts/sample.py + penalidade de repetição) — CPU, config tiny.

Cobrem os três entregáveis: perplexidade no val (sanity: init ≈ ln V), amostras
(determinismo greedy) e o benchmark do KV cache (cache e naive geram os MESMOS tokens —
se divergissem, o benchmark estaria medindo coisa errada).
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pingo_sample", ROOT / "scripts" / "sample.py")
sample = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sample)

from pingo.config import PingoConfig  # noqa: E402
from pingo.generate import generate  # noqa: E402
from pingo.model import PingoLM  # noqa: E402

VOCAB = 64


def tiny_model(seed: int = 0) -> PingoLM:
    torch.manual_seed(seed)
    cfg = PingoConfig(vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=4, intermediate_size=64,
                      max_position_embeddings=64)
    return PingoLM(cfg).eval()


@pytest.fixture()
def tokens_dir(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(2):
        rng.integers(3, VOCAB, size=3000, dtype=np.uint16).tofile(tmp_path / f"train-{i:04d}.bin")
        rng.integers(3, VOCAB, size=1200, dtype=np.uint16).tofile(tmp_path / f"val-{i:04d}.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"train_tokens": 6000, "val_tokens": 2400}))
    return tmp_path


def test_perplexidade_no_init_aprox_ln_v(tokens_dir):
    args = sample.parse_args(["--ppl-iters", "2", "--ppl-batch-size", "2", "--ctx", "32"])
    r = sample.perplexity(tiny_model(), sample.load_tokens(tokens_dir), args, "cpu")
    assert r["tokens_avaliados"] == 2 * 2 * 32
    # modelo aleatório → loss ≈ ln(V) (é o teste de sanidade da régua)
    assert r["val_loss"] == pytest.approx(np.log(VOCAB), abs=0.5)
    assert r["val_pplx"] == pytest.approx(VOCAB, rel=0.7)
    assert r["bits_per_token"] == pytest.approx(r["val_loss"] / np.log(2), rel=1e-3)


def test_cache_e_naive_geram_mesmos_tokens():
    model, prompt = tiny_model(), torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    a = sample.generate_naive(model, prompt, 8)
    b = sample.generate_cached(model, prompt, 8)
    assert torch.equal(a, b)          # é o que valida o benchmark
    assert a.shape == (1, 16)


def test_bench_kv_cache_reporta_speedup_e_corretude():
    r = sample.bench_kv_cache(tiny_model(), list(range(1, 40)), ctx_len=16,
                             n_new=4, batch_size=2, device="cpu")
    assert r["ctx"] == 16 and r["batch"] == 2
    assert r["mesmos_tokens"] is True
    assert r["naive"]["segundos"] > 0 and r["cache"]["segundos"] > 0
    assert r["speedup"] > 0


def test_bench_limita_ctx_para_sobrar_espaco():
    # ctx pedido = max_position_embeddings e ainda queremos 4 tokens novos
    r = sample.bench_kv_cache(tiny_model(), list(range(1, 70)), ctx_len=64,
                             n_new=4, batch_size=1, device="cpu")
    assert r["ctx"] == 60 and r["ctx_ajustado"] is True
    assert r["mesmos_tokens"] is True


# ------------------------------------------------------- penalidade de repetição

class _RepeatToken(torch.nn.Module):
    """Modelo falso que sempre prefere o token 7 (para forçar o loop do greedy)."""

    def __init__(self, config, token: int = 7):
        super().__init__()
        self.config = config
        self.token = token
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids, cache=None, labels=None):
        if cache is not None:  # imita PingoLM: escreve no cache e faz commit
            zeros = torch.zeros(ids.shape[0], cache.n_heads, ids.shape[1], cache.head_dim)
            cache.update(0, zeros, zeros)
            cache.commit()
        # todos os tokens empatados em 4.0, o dominante em 5.0: com penalidade 2.0 o
        # dominante cai para 2.5 e perde dos demais → o loop se quebra
        logits = torch.full((ids.shape[0], ids.shape[1], self.config.vocab_size), 4.0)
        logits[..., self.token] = 5.0
        return SimpleNamespace(logits=logits)


def test_penalidade_de_repeticao_quebra_loop_greedy():
    cfg = PingoConfig(vocab_size=VOCAB, hidden_size=32, num_hidden_layers=1,
                      num_attention_heads=2, intermediate_size=64, max_position_embeddings=32)
    stub = _RepeatToken(cfg).eval()
    livre = generate(stub, [1, 2, 3], max_new_tokens=6, temperature=0)
    assert (livre[0, 3:] == 7).all(), "sem penalidade o greedy repete o token dominante"

    com_penalidade = generate(stub, [1, 2, 3], max_new_tokens=6, temperature=0,
                              repetition_penalty=2.0)
    gerados = com_penalidade[0, 3:]
    assert (gerados == 7).sum() < 6, "com penalidade o token dominante não domina mais"
    assert len(set(gerados.tolist())) > 1, "a penalidade fez o greedy alternar de token"


def test_main_gera_relatorio(tmp_path, tokens_dir):
    tok_path = ROOT / "tokenizer" / "tokenizer.json"
    if not tok_path.exists():
        pytest.skip("tokenizer não está nesta máquina")
    # vocab do tokenizer real (16k): ids reais não caberiam no modelo tiny de 64
    torch.manual_seed(0)
    cfg = PingoConfig(vocab_size=16_384, hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=4, intermediate_size=64,
                      max_position_embeddings=64)
    model_dir = tmp_path / "model"
    PingoLM(cfg).eval().save_pretrained(model_dir)
    out = tmp_path / "eval.json"
    sample.main([
        "--model", str(model_dir), "--tokenizer", str(tok_path),
        "--tokens-dir", str(tokens_dir), "--device", "cpu", "--out", str(out),
        "--ppl-iters", "1", "--ppl-batch-size", "1", "--ctx", "16",
        "--max-new-tokens", "4", "--no-bench", "--prompt", "Olá",
    ])
    r = json.loads(out.read_text())
    assert r["params"] > 0 and r["device"] == "cpu"
    assert r["perplexity"]["tokens_avaliados"] == 16
    assert len(r["samples"]) == 2          # greedy + sample do único prompt
    assert r["samples"][0]["texto"].startswith("Olá")
