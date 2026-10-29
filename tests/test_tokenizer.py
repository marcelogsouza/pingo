"""Testes do tokenizer BPE byte-level treinado (Passo 3).

O artefato em `tokenizer/` é gitignored → os testes pulam se ele não existir
nessa máquina (ex.: CI). Para gerá-lo: `uv run python scripts/train_tokenizer.py`.
"""

from pathlib import Path

import pytest

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"
pytestmark = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(),
                                reason="tokenizer ainda não treinado nesta máquina")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)


def test_especiais_e_vocabs(tok):
    assert tok.vocab_size == 16_384
    assert tok.convert_ids_to_tokens([0, 1, 2]) == ["<|bos|>", "<|eos|>", "<|pad|>"]
    assert (tok.bos_token, tok.eos_token, tok.pad_token) == ("<|bos|>", "<|eos|>", "<|pad|>")


def test_roundtrip_e_byte_fallback(tok):
    s = "Coração, São Paulo — 123 https://exemplo.pt café ãõç"
    assert tok.decode(tok.encode(s, add_special_tokens=False)) == s
    # fora do domínio pt: cai em bytes, sem UNK
    assert not any(t == tok.unk_token for t in tok.convert_ids_to_tokens(
        tok.encode("日本語のテキスト", add_special_tokens=False)) if t)


def test_compressao_pt(tok):
    s = ("O Brasil possui uma das maiores biodiversidades do mundo, com biomas "
         "como a Amazônia, o Cerrado, a Mata Atlântica e o Pantanal.")
    assert len(s) / len(tok.encode(s, add_special_tokens=False)) > 3.5
