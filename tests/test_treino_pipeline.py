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


def test_corpus_ja_fatiado_val_jsonl_e_guarda(tmp_path, capsys):
    """Ingest que fatia na origem (`<stem>.jsonl` + `<stem>-val.jsonl`) não pode ser re-fatiado
    por hash: o train-only NÃO contém os docs do val, o val sairia vazio e o eval mentiria.
    Foi o bug do wiki (2026-11-03): auto-detecção do `<stem>-val.jsonl` + guarda de val vazio."""
    import hashlib  # para montar um train-only de verdade (todos com hash%100 ≥ 2)
    from process_data import norm_key

    rng = random.Random(11)
    docs = []
    i = 0
    while len(docs) < 100:                     # só os que o hash chamaria de "train"
        t = " ".join(rng.choice(FRASES) for _ in range(3)) + f" Documento {i}."
        if int.from_bytes(norm_key(t), "big") % 100 >= 2:
            docs.append(t)
        i += 1
    corpus = tmp_path / "pt.jsonl"
    corpus.write_text("".join(json.dumps({"content": d}, ensure_ascii=False) + "\n" for d in docs),
                      encoding="utf-8")
    val_docs = [" ".join(rng.choice(FRASES) for _ in range(3)) + f" Val {j}." for j in range(20)]
    (tmp_path / "pt-val.jsonl").write_text(
        "".join(json.dumps({"content": d}, ensure_ascii=False) + "\n" for d in val_docs),
        encoding="utf-8")

    tok_dir = tmp_path / "tokenizer" / "exp-fatiado"
    train_tokenizer.main(["--id", "exp-fatiado", "--out", str(tok_dir), "--vocab", "200",
                          "--min-frequency", "1", "--sample-chars", "100_000",
                          "--val-chars", "20_000", "--specials", ESPECIAIS, "--jsonl", str(corpus)])

    tdir = tmp_path / "tokens" / "fatiado"
    prepare_tokens.main(["--input", "jsonl", "--jsonl", str(corpus),
                         "--tokenizer", str(tok_dir), "--out", str(tdir),
                         "--val-pct", "2", "--shard-tokens", "500_000",
                         "--flush-chars", "50_000", "--n-process", "1"])
    meta = json.loads((tdir / "meta.json").read_text())
    assert meta["val_jsonl"].endswith("pt-val.jsonl")     # auto-detetado, não re-fatiado
    assert meta["val_stats"]["docs"] == 20 and meta["val_tokens"] > 0
    assert meta["train_stats"]["docs"] == 100
    assert "val: " in capsys.readouterr().out

    # sem o arquivo do val, o train-only não produz val nenhum → para em vez de treinar às cegas
    (tmp_path / "pt-val.jsonl").rename(tmp_path / "pt-val.jsonl.off")
    with pytest.raises(SystemExit):
        prepare_tokens.main(["--input", "jsonl", "--jsonl", str(corpus),
                             "--tokenizer", str(tok_dir), "--out", str(tmp_path / "tokens" / "ruim"),
                             "--val-pct", "2", "--shard-tokens", "500_000",
                             "--flush-chars", "50_000", "--n-process", "1"])


def test_iter_docs_jsonl_ja_fatiado_devolve_o_val_inteiro(tmp_path):
    train, val = tmp_path / "a.jsonl", tmp_path / "a-val.jsonl"
    train.write_text('{"content": "treino um"}\n{"content": "treino dois"}\n', encoding="utf-8")
    val.write_text('{"content": "val do qual"}\n{"content": "val do hash"}\n', encoding="utf-8")
    assert list(prepare_tokens.iter_docs_jsonl(train, "val", 2, val)) == ["val do qual", "val do hash"]
    assert list(prepare_tokens.iter_docs_jsonl(train, "train", 2, val)) == ["treino um", "treino dois"]


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
