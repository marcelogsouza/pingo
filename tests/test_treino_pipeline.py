"""Treino-5 — tokenizers múltiplos de ponta a ponta (CPU, sem rede, tudo em tmp/).

A corrente que a UI do /treino vai montar:

    JSONL (QA) → train_tokenizer --jsonl --specials(…) → tokenizer/<id>/
               → prepare_tokens --input jsonl --tokenizer tokenizer/<id>/ → meta.json amarrado
               → train.py DERIVA vocab_size do meta.json (nunca digita à mão)
               → train_sft.py com o MESMO tokenizer — chat já embutido, resize 0.

E prova que nada toca no `tokenizer/` default nem nos runs antigos.
"""

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_tokens  # noqa: E402  (script importado como módulo p/ main(argv))
import train_tokenizer  # noqa: E402

_spec = importlib.util.spec_from_file_location("pingo_train", ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

_spec_sft = importlib.util.spec_from_file_location("pingo_train_sft_5",
                                                   ROOT / "scripts" / "train_sft.py")
tsft = importlib.util.module_from_spec(_spec_sft)
sys.modules["pingo_train_sft_5"] = tsft
_spec_sft.loader.exec_module(tsft)

FRASES = [
    "A empresa anunciou nesta terça um plano de investimentos em energia limpa no interior do estado.",
    "O governo aprovou o novo marco fiscal depois de semanas de negociação no Congresso Nacional.",
    "Pesquisadores da universidade desenvolveram uma vacina experimental contra a doença tropical.",
    "O time venceu o clássico por três a dois e garantiu vaga na final do campeonato estadual.",
    "A economia cresceu no trimestre puxada pelo consumo das famílias e pela agropecuária exportadora.",
    "A prefeitura abriu licitação para reformar o sistema de transporte público da região metropolitana.",
    "Chuvas fortes deixaram ruas alagadas e provocaram deslizamentos nas encostas do morro.",
    "O banco central manteve a taxa de juros e sinalizou cortes gradualmente ao longo do próximo ano.",
]
ESPECIAIS = "<|bos|>,<|eos|>,<|pad|>,<|user|>,<|assistant|>,<|end|>"


def _corpus(tmp_path: Path, n_docs: int = 80) -> Path:
    rng = random.Random(3)
    p = tmp_path / "corpus.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for i in range(n_docs):
            texto = " ".join(rng.choice(FRASES) for _ in range(4)) + f" Documento {i}."
            fh.write(json.dumps({"content": texto}, ensure_ascii=False) + "\n")
    return p


def test_corrente_completa_com_tokenizer_novo(tmp_path, capsys):
    corpus = _corpus(tmp_path)
    tok_dir = tmp_path / "tokenizer" / "exp-400"
    tdir = tmp_path / "tokens" / "corpus-exp400"

    # 1) tokenizer novo, com os tokens de chat JÁ embutidos na origem
    train_tokenizer.main(["--id", "exp-400", "--out", str(tok_dir), "--vocab", "400",
                          "--min-frequency", "1", "--sample-chars", "200_000",
                          "--val-chars", "40_000", "--specials", ESPECIAIS,
                          "--jsonl", str(corpus)])
    stats = json.loads((tok_dir / "vocab_stats.json").read_text())
    assert stats["vocab_size"] <= 400 and stats["vocab_size"] > 100
    assert stats["special_tokens"] == ESPECIAIS.split(",")
    assert stats["chat_tokens_embutidos"] == ["<|user|>", "<|assistant|>", "<|end|>"]
    assert (tok_dir / "tokenizer.json").exists()

    # 2) binarização JSONL amarra o meta.json ao tokenizer
    prepare_tokens.main(["--input", "jsonl", "--jsonl", str(corpus),
                         "--tokenizer", str(tok_dir), "--out", str(tdir),
                         "--val-pct", "25", "--shard-tokens", "500_000",
                         "--flush-chars", "50_000", "--n-process", "1"])
    meta = json.loads((tdir / "meta.json").read_text())
    assert meta["vocab_size"] == stats["vocab_size"] and meta["input"] == "jsonl"
    assert meta["tokenizer_id"] == "exp-400"
    assert meta["train_tokens"] > 0 and meta["val_tokens"] > 0
    assert list(tdir.glob("train-*.bin")) and list(tdir.glob("val-*.bin"))

    # split estável: reprocessar com a mesma seed/val-pct dá a mesma conta
    tdir2 = tmp_path / "tokens" / "repetido"
    prepare_tokens.main(["--input", "jsonl", "--jsonl", str(corpus),
                         "--tokenizer", str(tok_dir), "--out", str(tdir2),
                         "--val-pct", "25", "--shard-tokens", "500_000",
                         "--flush-chars", "50_000", "--n-process", "1"])
    assert json.loads((tdir2 / "meta.json").read_text())["train_tokens"] == meta["train_tokens"]

    # 3) pré-treino tiny: o vocab vem do meta.json, não de quem digitou
    res = train.run(train.parse_args([
        "--tokens-dir", str(tdir), "--out", str(tmp_path / "run-pt"), "--tiny",
        "--device", "cpu", "--ctx", "64", "--batch-size", "4", "--accum-steps", "2",
        "--warmup-steps", "1", "--eval-steps", "2", "--eval-iters", "2",
        "--log-steps", "1", "--ckpt-min", "1000000", "--max-steps", "2", "--seed", "7"]))
    assert res["model"].config.vocab_size == stats["vocab_size"]

    # 4) SFT tiny sobre o MESMO tokenizer (diretório!) — chat embutido ⇒ resize 0
    dados = tmp_path / "sft"
    dados.mkdir()
    for nome in ("train", "val"):
        with (dados / f"{nome}.jsonl").open("w", encoding="utf-8") as fh:
            for i in range(8):
                rec = {"ancorado": i % 2 == 0, "chunk_text": FRASES[i % len(FRASES)],
                       "pergunta": "O que foi anunciado nesta terça?",
                       "resposta": "Um plano de investimentos em energia limpa."}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    res_sft = tsft.run(tsft.parse_args([
        "--tiny", "--data-dir", str(dados), "--tokenizer", str(tok_dir),
        "--out", str(tmp_path / "run-sft"), "--max-steps", "2", "--batch-size", "3",
        "--accum-steps", "1", "--eval-steps", "2", "--log-steps", "1", "--device", "cpu",
        "--max-len", "256", "--lr", "1e-3", "--warmup-steps", "1", "--seed", "7"]))
    assert res_sft["step"] == 2
    assert "tokens de chat novos: 0" in capsys.readouterr().out
    assert (tmp_path / "run-sft" / "final" / "tokenizer.json").exists()


@pytest.fixture()
def tokens_dir_v777(tmp_path):
    for i, n in enumerate([3000, 2000]):
        (np.arange(n, dtype=np.uint16) % 700).tofile(tmp_path / f"train-{i:04d}.bin")
    (np.arange(1500, dtype=np.uint16) % 700).tofile(tmp_path / "val-0000.bin")
    (tmp_path / "meta.json").write_text(json.dumps(
        {"train_shards": 2, "train_tokens": 5000, "val_shards": 1,
         "val_tokens": 1500, "vocab_size": 777}))
    return tmp_path


def test_vocab_do_run_nunca_diverge_do_meta(tokens_dir_v777, tmp_path):
    """Meta diz 777? O modelo tem 777 — config default (16.384) não vence o meta."""
    res = train.run(train.parse_args([
        "--tokens-dir", str(tokens_dir_v777), "--out", str(tmp_path / "run"), "--tiny",
        "--device", "cpu", "--ctx", "32", "--batch-size", "4", "--accum-steps", "2",
        "--warmup-steps", "1", "--eval-steps", "2", "--eval-iters", "2",
        "--log-steps", "1", "--ckpt-min", "1000000", "--max-steps", "2", "--seed", "7"]))
    assert res["model"].config.vocab_size == 777
