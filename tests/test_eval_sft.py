"""Testes do SFT-4 (`scripts/eval_sft.py`) — CPU, sem rede.

As métricas são o produto do SFT-4: se `f1_words`, `support_score` ou `score_example`
estiverem errados, o relatório "aprova" um modelo ruim. Cobrimos o que dá para afirmar
sem ambiguidade: igualdade exata, ausência de sobreposição, recusa por modo, e a agregação
(que não pode deixar `None` contaminar a média).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("pingo_eval_sft", ROOT / "scripts" / "eval_sft.py")
e = importlib.util.module_from_spec(_spec)
sys.modules["pingo_eval_sft"] = e
_spec.loader.exec_module(e)

_spec_s = importlib.util.spec_from_file_location("pingo_sample2", ROOT / "scripts" / "sample.py")
sample = importlib.util.module_from_spec(_spec_s)
sys.modules["pingo_sample2"] = sample
_spec_s.loader.exec_module(sample)

from pingo.sft import END_TOKEN, RECUSA_ANCORADO, RECUSA_LIVRE  # noqa: E402

CHUNK = ("A Companhia Energética de Minas Gerais anunciou um plano de investimentos de 12 "
         "bilhões de reais até 2028, com foco em redes de distribuição no interior do estado.")


def rec(**over) -> dict:
    base = {"doc_id": "d", "url": "https://x", "split": "train", "chunk_idx": 0,
            "chunk_text": CHUNK, "tipo": "extracao", "ancorado": True,
            "pergunta": "Quanto a companhia planeja investir até 2028?",
            "resposta": "A companhia planeja investir 12 bilhões de reais até 2028.",
            "evidencia": "um plano de investimentos de 12 bilhões de reais até 2028"}
    base.update(over)
    return base


# ------------------------------------------------------------------ métricas puras


def test_f1_words_extremos():
    assert e.f1_words("o gato subiu no telhado", "o gato subiu no telhado") == 1.0
    assert e.f1_words("totalmente outra frase", "o gato subiu no telhado") == 0.0
    assert e.f1_words("", "o gato subiu") == 0.0
    # parcial: metade das palavras
    f1 = e.f1_words("gato subiu", "o gato subiu no telhado")
    assert 0.5 < f1 < 1.0


def test_f1_words_ignora_caixa_e_pontuacao():
    assert e.f1_words("GATO, subiu!", "gato subiu") == 1.0


def test_support_score():
    resposta = "A companhia investe 12 bilhões em redes de distribuição no interior"
    assert e.support_score(resposta, CHUNK) == 1.0
    assert e.support_score("A empresa planeja comprar foguetes para Marte", CHUNK) == 0.0
    assert e.support_score("", CHUNK) == 0.0
    # metade inventada
    meio = e.support_score("A companhia investe 12 bilhões em foguetes de Marte", CHUNK)
    assert 0.0 < meio < 1.0


def test_limpar_geracao_tira_tokens_de_chat():
    assert e.limpar_geracao(f"  resposta   aqui {END_TOKEN}") == "resposta aqui"
    assert e.limpar_geracao("<|assistant|><|assistant|>") == ""


def test_recusa_ok_por_modo():
    assert e.recusa_ok(RECUSA_ANCORADO, ancorado=True)
    assert not e.recusa_ok(RECUSA_ANCORADO, ancorado=False)
    assert e.recusa_ok(RECUSA_LIVRE, ancorado=False)
    assert not e.recusa_ok("A resposta é 42.", ancorado=True)


# ------------------------------------------------------------------ score_example / agregar


def test_score_example_respondivel():
    s = e.score_example(rec(), "A companhia planeja investir 12 bilhões de reais até 2028.")
    assert s["f1"] == 1.0
    # "planeja" é paráfrase de "plano": a métrica lexical deixa passar 1 palavra de conteúdo
    assert s["apoio"] >= 0.8
    assert s["terminou"] is False and "recusa_ok" not in s


def test_score_example_marca_fim_de_turno():
    gerado = f"12 bilhões de reais até 2028.{END_TOKEN}"
    s = e.score_example(rec(), gerado)
    assert s["terminou"] is True
    assert END_TOKEN not in s["gerado"]           # o gerado vai limpo p/ o relatório


def test_score_example_recusa():
    s = e.score_example(rec(tipo="recusa"), RECUSA_ANCORADO)
    assert s["recusa_ok"] is True and "apoio" not in s
    s2 = e.score_example(rec(tipo="recusa"), "A companhia investe 12 bilhões.")
    assert s2["recusa_ok"] is False


def test_agregar_nao_contamina_media_com_none():
    scores = [
        e.score_example(rec(), "12 bilhões de reais até 2028."),
        e.score_example(rec(ancorado=False), "resposta completamente inventada sobre Marte"),
        e.score_example(rec(tipo="recusa"), RECUSA_ANCORADO),
        e.score_example(rec(tipo="recusa", ancorado=False), "sei lá"),
    ]
    m = e.agregar(scores)
    assert m["n"] == 4
    assert m["recusa_ok"] == 0.5                  # 1 de 2 recusas aceitou
    assert m["apoio_ancorado"] == 1.0             # só o ancorado entra nessa média
    assert m["apoio_sem_contexto"] == 0.0
    assert m["por_tipo"]["recusa"]["n"] == 2
    assert m["taxa_terminou"] == 0.0


def test_agregar_lista_vazia_nao_quebra():
    m = e.agregar([])
    assert m["n"] == 0 and m["f1"] is None and m["recusa_ok"] is None


# ------------------------------------------------------------------ tokenizer do modelo


def test_sample_resolve_tokenizer_do_modelo(tmp_path):
    modelo = tmp_path / "modelo"
    modelo.mkdir()
    assert sample.resolve_tokenizer(modelo, Path("tokenizer/tokenizer.json")) == \
        Path("tokenizer/tokenizer.json")          # sem tokenizer.json → usa o default
    (modelo / "tokenizer.json").write_text("{}")
    assert sample.resolve_tokenizer(modelo, Path("tokenizer/tokenizer.json")) == \
        modelo / "tokenizer.json"                 # o do modelo SFT manda


# ------------------------------------------------------------------ ponta a ponta


@pytest.mark.skipif(not (ROOT / "tokenizer" / "tokenizer.json").exists(),
                    reason="tokenizer/ não existe nesta máquina (gitignored)")
def test_main_ponta_a_ponta_com_modelo_tiny(tmp_path):
    import torch
    from tokenizers import Tokenizer

    from pingo.config import PingoConfig
    from pingo.model import PingoLM
    from pingo.sft import ensure_chat_tokens

    torch.manual_seed(0)
    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
    ensure_chat_tokens(tok)
    modelo = tmp_path / "tiny"
    cfg = PingoConfig(vocab_size=tok.get_vocab_size(), hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=4, intermediate_size=64,
                      max_position_embeddings=256)
    PingoLM(cfg).save_pretrained(modelo)
    tok.save(str(modelo / "tokenizer.json"))

    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in
                                 [rec(), rec(ancorado=False), rec(tipo="recusa", ancorado=False)]))
    out = tmp_path / "eval.json"
    rep = e.main(["--model", str(modelo), "--heldout", str(heldout), "--out", str(out),
                  "--n", "3", "--max-new-tokens", "4", "--no-ppl", "--show", "1",
                  "--device", "cpu"])

    assert rep["n"] == 3 and rep["vocab"] == tok.get_vocab_size()
    assert "metricas" in rep and rep["metricas"]["n"] == 3
    assert len(rep["amostras"]) == 1
    assert json.loads(out.read_text())["n"] == 3
