"""Testes do SFT-3 (`scripts/train_sft.py`) — CPU, sem rede, config tiny.

O que não pode falhar em silêncio:

- **alinhamento de labels**: o modelo prevê o próximo token, então o label da posição `t` é o
  token `t+1`; errar isso dá uma loss plausível (~12) e destrói o treino sem avisar;
- **loss só na resposta**: prompt e padding ficam `-100`;
- **tokens de chat**: embedding cresce com as linhas novas na MÉDIA das antigas, e o tie
  `lm_head ↔ embed` continua valendo (senão o modelo esquece o vocabulário que já sabia);
- **resume bit-exato**: 4 steps == 2 steps + `--resume` até 4.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("pingo_train_sft",
                                               ROOT / "scripts" / "train_sft.py")
t = importlib.util.module_from_spec(_spec)
sys.modules["pingo_train_sft"] = t
_spec.loader.exec_module(t)

from pingo.config import PingoConfig  # noqa: E402
from pingo.model import PingoLM  # noqa: E402


# ------------------------------------------------------------------ utilidades


class FakeTokenizer:
    """Tokenizer de mentira (1 id por caractere) — o teste não depende do tokenizer real."""

    CHAT = ["<|user|>", "<|assistant|>", "<|end|>"]

    def __init__(self):
        self.vocab = {"<|bos|>": 0, "<|eos|>": 1, "<|pad|>": 2}
        self.next_id = 63  # 3..62 reservados p/ os caracteres

    @classmethod
    def from_file(cls, path):
        return cls()

    def token_to_id(self, token):
        return self.vocab.get(token)

    def get_vocab_size(self):
        return self.next_id

    def add_special_tokens(self, tokens):
        for tok in tokens:
            if tok not in self.vocab:
                self.vocab[tok] = self.next_id
                self.next_id += 1

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            for tok, tid in self.vocab.items():
                if tok.startswith("<") and text.startswith(tok, i):
                    ids.append(tid)
                    i += len(tok)
                    break
            else:
                ids.append(3 + (ord(text[i]) % 60))
                i += 1
        return SimpleNamespace(ids=ids)

    def save(self, path):
        Path(path).write_text("{}")


def tiny_model(vocab: int = 64, ctx: int = 64) -> PingoLM:
    torch.manual_seed(0)
    cfg = PingoConfig(vocab_size=vocab, hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=4, intermediate_size=64,
                      max_position_embeddings=ctx)
    return PingoLM(cfg)


def sft_data(n: int = 12, n_prompt: int = 5, n_resp: int = 4) -> t.SFTData:
    data = t.SFTData()
    rng = np.random.default_rng(0)
    for _ in range(n):
        ids = rng.integers(10, 60, size=n_prompt + n_resp).astype(np.int32)
        data.ids.append(ids)
        data.n_prompt.append(n_prompt)
    return data


# ------------------------------------------------------------------ collate


def test_collate_desloca_labels_e_mascara_prompt_e_pad():
    data = t.SFTData(ids=[np.array([10, 11, 12, 13, 14], dtype=np.int32),
                          np.array([20, 21, 22], dtype=np.int32)],
                     n_prompt=[3, 2])
    x, mask, labels = t.collate(data, np.array([0, 1]), pad_id=2)
    assert x.shape == mask.shape == labels.shape
    assert x.shape[1] >= 5

    # exemplo 0: prompt = [10,11,12], resposta = [13,14]
    assert x[0, :5].tolist() == [10, 11, 12, 13, 14]
    assert labels[0, :5].tolist() == [-100, -100, 13, 14, -100]  # label[t] = x[t+1]
    assert labels[0, 5:].tolist() == [-100] * (x.shape[1] - 5)   # padding fora da loss

    # exemplo 1: prompt = [20,21], resposta = [22]
    assert labels[1, :3].tolist() == [-100, 22, -100]
    assert mask[1, :3].tolist() == [1, 1, 1] and mask[1, 3:].tolist() == [0] * (x.shape[1] - 3)
    assert x[1, 3:].tolist() == [2] * (x.shape[1] - 3)          # pad à direita


def test_loss_ignora_prompt_e_padding():
    data = sft_data()
    model = tiny_model(vocab=64).eval()
    x, mask, labels = t.collate(data, np.arange(4), pad_id=2)
    out = model(x, attention_mask=mask, labels=labels)
    alvo = labels[labels != -100]
    esperado = torch.nn.functional.cross_entropy(out.logits[labels != -100], alvo)
    assert torch.allclose(out.loss, esperado, atol=1e-5)


def test_padding_nao_contamina_as_posicoes_reais():
    """Mascarar o padding tem de dar o mesmo resultado do forward sem padding."""
    data = t.SFTData(ids=[np.array([10, 11, 12, 13, 14, 15], dtype=np.int32),
                          np.array([20, 21, 22, 23], dtype=np.int32)],
                     n_prompt=[3, 2])
    model = tiny_model(vocab=64).eval()
    x, mask, labels = t.collate(data, np.array([0, 1]), pad_id=2)
    with torch.no_grad():
        com_pad = model(x, attention_mask=mask).logits
        sozinho = model(torch.tensor([[20, 21, 22, 23]]))
    assert torch.allclose(com_pad[1, :4], sozinho.logits[0, :4], atol=1e-4)


# ------------------------------------------------------------------ embedding / tokens


def test_resize_embeddings_inicia_na_media_e_mantem_o_tie():
    model = tiny_model(vocab=64)
    antigo = model.embed.weight.data.clone()
    media = antigo.mean(dim=0)
    assert t.resize_embeddings(model, 67) == 3
    w = model.embed.weight.data
    assert w.shape[0] == 67 and model.config.vocab_size == 67
    assert torch.allclose(w[:64], antigo)                     # nada do que já sabia se perde
    for i in (64, 65, 66):
        assert torch.allclose(w[i], media, atol=1e-6)         # novos = média
    assert model.lm_head.weight is model.embed.weight         # tie preservado
    assert model.lm_head.weight.shape[0] == 67


def test_ensure_chat_tokens_e_ids_estaveis():
    tok = FakeTokenizer()
    assert t.ensure_chat_tokens(tok) == 3
    assert t.ensure_chat_tokens(tok) == 0
    assert t.chat_token_ids(tok) == {"<|user|>": 63, "<|assistant|>": 64, "<|end|>": 65}


def test_encode_example_descarta_o_que_nao_cabe():
    tok = FakeTokenizer()
    t.ensure_chat_tokens(tok)
    rec = {"ancorado": False, "chunk_text": "", "pergunta": "Quem descobriu o Brasil?",
           "resposta": "Pedro Álvares Cabral."}
    enc = t.encode_example(tok, rec, max_len=1024)
    assert enc is not None
    ids, n_prompt = enc
    # a resposta começa logo depois do <|assistant|>
    assert ids[n_prompt - 1] == tok.token_to_id("<|assistant|>")
    assert ids[-1] == tok.token_to_id("<|end|>")
    assert t.encode_example(tok, rec, max_len=10) is None    # não cabe → descartado


# ------------------------------------------------------------------ run / resume


def _dataset(dir_: Path, n: int = 12) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for nome in ("train", "val"):
        with (dir_ / f"{nome}.jsonl").open("w", encoding="utf-8") as fh:
            for i in range(n):
                rec = {"doc_id": f"d{i % 4}", "url": "https://x", "split": "train",
                       "chunk_idx": 0, "chunk_text": "A empresa cresceu bastante no ano.",
                       "tipo": "extracao", "ancorado": i % 2 == 0,
                       "pergunta": f"Qual foi o resultado da empresa no ano {i}?",
                       "resposta": "A empresa cresceu bastante.", "evidencia": ""}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _run_args(tmp_path: Path, out: Path, **over) -> list[str]:
    base = ["--tiny", "--data-dir", str(tmp_path / "data"), "--tokenizer", "ignorado",
            "--out", str(out), "--max-steps", "4", "--batch-size", "3", "--accum-steps", "1",
            "--eval-steps", "2", "--log-steps", "1", "--device", "cpu", "--max-len", "256",
            "--lr", "1e-3", "--warmup-steps", "1", "--seed", "7"]
    for k, v in over.items():
        base += [f"--{k.replace('_', '-')}", str(v)]
    return base


def test_run_gera_ckpt_best_final_e_log(tmp_path, monkeypatch):
    import tokenizers

    monkeypatch.setattr(tokenizers, "Tokenizer", FakeTokenizer)
    _dataset(tmp_path / "data")
    out = tmp_path / "run"
    res = t.run(t.parse_args(_run_args(tmp_path, out)))
    assert res["step"] == 4
    for nome in ("ckpt.pt", "best.pt", "log.jsonl", "chat_tokens.json"):
        assert (out / nome).exists(), nome
    assert (out / "final" / "model.safetensors").exists()
    assert (out / "final" / "tokenizer.json").exists()
    logs = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    assert any("train_loss" in r for r in logs)
    assert any("val_loss" in r for r in logs)


def test_resume_reproduz_bit_exato(tmp_path, monkeypatch):
    import tokenizers

    monkeypatch.setattr(tokenizers, "Tokenizer", FakeTokenizer)
    _dataset(tmp_path / "data")

    # run contínuo de 4 steps
    out_a = tmp_path / "continuo"
    t.run(t.parse_args(_run_args(tmp_path, out_a)))
    ck_a = torch.load(out_a / "ckpt.pt", map_location="cpu", weights_only=False)

    # 2 steps + --resume até 4
    out_b = tmp_path / "fatiado"
    t.run(t.parse_args(_run_args(tmp_path, out_b, max_steps=2)))
    t.run(t.parse_args(_run_args(tmp_path, out_b, max_steps=4) + ["--resume"]))
    ck_b = torch.load(out_b / "ckpt.pt", map_location="cpu", weights_only=False)

    assert ck_a["step"] == ck_b["step"] == 4
    for k, v in ck_a["model"].items():
        assert torch.equal(v, ck_b["model"][k]), f"peso {k} divergiu no resume"


def test_dataset_vazio_avisa(tmp_path, monkeypatch):
    import tokenizers

    monkeypatch.setattr(tokenizers, "Tokenizer", FakeTokenizer)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.jsonl").write_text("")
    (tmp_path / "data" / "val.jsonl").write_text("")
    with pytest.raises(SystemExit):
        t.run(t.parse_args(_run_args(tmp_path, tmp_path / "run")))
