"""Testes do modelo (Passo 5) — config tiny, roda em CPU.

Cobrem: contagem de params do Pingo-120M, shapes/loss, tied embeddings,
**equivalência do KV cache** (`max|logits_full − logits_cached| < 1e-5` fp32),
buffers pré-alocados + overflow, overfit de batch pequeno, generate() e
roundtrip save_pretrained/from_pretrained.
"""

import pytest
import torch
import torch.nn.functional as F

from pingo.cache import PingoCache
from pingo.config import PingoConfig
from pingo.generate import generate, sample_next
from pingo.model import PingoLM, apply_rope


def tiny_config(**overrides) -> PingoConfig:
    kw = dict(vocab_size=101, hidden_size=32, num_hidden_layers=2,
              num_attention_heads=4, intermediate_size=64,
              max_position_embeddings=32)
    kw.update(overrides)
    return PingoConfig(**kw)


@pytest.fixture()
def tiny_model():
    torch.manual_seed(0)
    return PingoLM(tiny_config())


# ---------------------------------------------------------------- config/params

def test_contagem_params_pingo120m():
    """Alvo do plano: ≈127,6M (14,7M emb tied + 12×9,4M de blocos + norm final)."""
    model = PingoLM(PingoConfig())  # default: 12L/896d/14h, vocab 16.384
    n = sum(p.numel() for p in model.parameters())
    expected = 16_384 * 896 + 12 * (4 * 896**2 + 3 * 896 * 2_304 + 2 * 896) + 896
    assert n == expected == 127_555_456


def test_head_dim_default_e_validacao():
    assert PingoConfig().head_dim == 64  # 896 / 14
    with pytest.raises(ValueError):
        PingoConfig(hidden_size=900, num_attention_heads=14)


def test_tied_embeddings(tiny_model):
    assert tiny_model.lm_head.weight is tiny_model.embed.weight
    # parameters() deduplica compartilhados: lm_head.weight some da contagem
    names = {n for n, _ in tiny_model.named_parameters()}
    assert "lm_head.weight" not in names
    emb = tiny_model.embed.weight.numel()  # 101×32
    per_layer = 4 * 32 * 32 + 3 * 32 * 64 + 2 * 32
    expected = emb + 2 * per_layer + 32  # + norm final
    assert sum(p.numel() for p in tiny_model.parameters()) == expected


# ---------------------------------------------------------------- forward/loss

def test_forward_shapes_e_loss(tiny_model):
    x = torch.randint(0, 101, (3, 16))
    out = tiny_model(x, labels=x)
    assert out.logits.shape == (3, 16, 101)
    manual = F.cross_entropy(out.logits.view(-1, 101), x.view(-1))
    assert torch.allclose(out.loss, manual)


def test_init_residuais_escalados(tiny_model):
    """o_proj/down_proj init com std 0.02/√(2L); lineares normais sem viés."""
    import math
    std = 0.02 / math.sqrt(2 * tiny_model.config.num_hidden_layers)
    o = tiny_model.blocks[0].attn.o_proj.weight
    assert 0.5 * std < o.std() < 1.5 * std
    assert tiny_model.blocks[0].attn.q_proj.bias is None
    assert torch.all(tiny_model.norm.weight == 1.0)


def test_rope_invariancia_relativa(tiny_model):
    """Propriedade do RoPE: score q·k depende só da DISTANÇA entre posições."""
    torch.manual_seed(0)
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 4, 1, 8)
    rope = tiny_model.rope
    cos_a, sin_a = rope.tables(torch.tensor([[5]]))
    cos_b, sin_b = rope.tables(torch.tensor([[9]]))  # distância 4
    cos_c, sin_c = rope.tables(torch.tensor([[1]]))
    cos_d, sin_d = rope.tables(torch.tensor([[5]]))  # distância 4
    s1 = (apply_rope(q, cos_a, sin_a) * apply_rope(k, cos_b, sin_b)).sum()
    s2 = (apply_rope(q, cos_c, sin_c) * apply_rope(k, cos_d, sin_d)).sum()
    assert torch.allclose(s1, s2, atol=1e-5)
    # e posições absolutas diferentes mudam o q rotacionado
    assert not torch.allclose(apply_rope(q, cos_a, sin_a), apply_rope(q, cos_c, sin_c))


# ---------------------------------------------------------------- KV cache

def _allclose_maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def test_cache_equivalencia_full_vs_prefill(tiny_model):
    """Prefill com cache (mesma janela) == forward sem cache, posição a posição."""
    torch.manual_seed(1)
    x = torch.randint(0, 101, (2, 13))
    with torch.no_grad():
        full = tiny_model(x).logits
        cache = PingoCache(tiny_model.config, batch_size=2, max_seq=32)
        cached = tiny_model(x, cache=cache).logits
    assert cache.cur_len == 13
    assert _allclose_maxdiff(full, cached) < 1e-5


def test_cache_equivalencia_prefill_e_steps(tiny_model):
    """Prefill parcial + decode token a token == forward cheio (critério < 1e-5 fp32)."""
    torch.manual_seed(2)
    x = torch.randint(0, 101, (2, 13))
    with torch.no_grad():
        full = tiny_model(x).logits
        cache = PingoCache(tiny_model.config, batch_size=2)
        n_prefill = 8
        logits = [tiny_model(x[:, :n_prefill], cache=cache).logits[:, -1]]
        for t in range(n_prefill, 13):
            logits.append(tiny_model(x[:, t:t + 1], cache=cache).logits[:, -1])
        cached = torch.stack(logits, dim=1)  # [2, 6, 101] = posições 7..12
    assert cached.shape == (2, 13 - n_prefill + 1, 101)
    ref = full[:, n_prefill - 1:]
    assert _allclose_maxdiff(ref, cached) < 1e-5
    assert cache.cur_len == 13


def test_cache_buffers_pre_alocados(tiny_model):
    x = torch.randint(0, 101, (2, 5))
    with torch.no_grad():
        cache = PingoCache(tiny_model.config, batch_size=2, max_seq=16)
        tiny_model(x, cache=cache)
    shape = (2, tiny_model.config.num_attention_heads, 16, tiny_model.config.head_dim)
    assert all(c.shape == shape for c in cache._k)
    assert all(c.shape == shape for c in cache._v)


def test_cache_overflow_e_steps_multiplos(tiny_model):
    x = torch.randint(0, 101, (1, 30))
    with torch.no_grad():
        cache = PingoCache(tiny_model.config, batch_size=1)  # max_seq=32
        tiny_model(x, cache=cache)
        with pytest.raises(ValueError, match="cache cheio"):
            tiny_model(torch.randint(0, 101, (1, 3)), cache=cache)
        # prefill com t>1 em cache NÃO vazio é proibido (mascararia futuro)
        cache2 = PingoCache(tiny_model.config, batch_size=1)
        tiny_model(x[:, :4], cache=cache2)
        with pytest.raises(ValueError, match="prefill"):
            tiny_model(x[:, 4:9], cache=cache2)


# ---------------------------------------------------------------- overfit

def test_overfit_batch_pequeno(tiny_model):
    """Batch fixo 8×24 → loss ≈ 0 (o modelo memoriza; pega bug de atenção/loss)."""
    torch.manual_seed(3)
    x = torch.randint(0, 101, (8, 24))
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=3e-3)
    for _ in range(300):
        loss = tiny_model(x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


# ---------------------------------------------------------------- generate

def test_sample_next_greedy_topk():
    logits = torch.tensor([[2.0, 5.0, 1.0, 3.0]])
    assert int(sample_next(logits, temperature=0)) == 1
    torch.manual_seed(0)
    draws = {int(sample_next(logits, temperature=1.0, top_k=2)) for _ in range(50)}
    assert draws <= {1, 3}  # top-2 são os ids 1 e 3


def test_generate_greedy_deterministico_e_sem_cache_match(tiny_model):
    """Greedy com cache == re-forward completo sem cache (mesmos ids)."""
    prompt = [5, 11, 23, 7]
    a = generate(tiny_model, prompt, max_new_tokens=8, temperature=0)
    b = generate(tiny_model, prompt, max_new_tokens=8, temperature=0)
    assert torch.equal(a, b)
    assert a.shape == (1, 12)

    # referência: re-forward da sequência inteira a cada step (sem cache)
    ids = torch.tensor([prompt])
    for _ in range(8):
        with torch.no_grad():
            nxt = tiny_model(ids).logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat((ids, nxt), dim=-1)
    assert torch.equal(a, ids)


def test_generate_stop_e_limite_contexto(tiny_model):
    base = generate(tiny_model, [1, 2, 3], max_new_tokens=20, temperature=0)
    gen = base[0, 3:].tolist()
    stop_id = gen[-1]
    expected_len = 3 + gen.index(stop_id) + 1
    out = generate(tiny_model, [1, 2, 3], max_new_tokens=20, temperature=0, stop_token=stop_id)
    assert out.shape == (1, expected_len)
    assert torch.equal(out[0], base[0, :expected_len])
    # ctx=32: para de preencher quando o cache lota (32 tokens no total)
    out = generate(tiny_model, [4] * 30, max_new_tokens=50, temperature=0)
    assert out.shape[1] == 32


def test_generate_stochastic_seeds(tiny_model):
    p = [1, 2, 3]
    a = generate(tiny_model, p, max_new_tokens=16, temperature=1.0, top_k=20, seed=42)
    b = generate(tiny_model, p, max_new_tokens=16, temperature=1.0, top_k=20, seed=42)
    assert torch.equal(a, b)


# ---------------------------------------------------------------- HF scaffolding

def test_save_load_roundtrip(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path)
    reloaded = PingoLM.from_pretrained(tmp_path, torch_dtype=torch.float32)
    x = torch.randint(0, 101, (2, 16))
    with torch.no_grad():
        assert _allclose_maxdiff(tiny_model(x).logits, reloaded(x).logits) < 1e-6
    assert reloaded.lm_head.weight is reloaded.embed.weight
