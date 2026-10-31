"""Testes da WebUI (`scripts/webui.py`) — CPU, sem rede e sem o tokenizer real.

A WebUI é fina de propósito, mas tem duas partes que precisam de teste: o **prompt**
(se ele não for igual ao do treino, o modelo responde pior e ninguém entende por quê) e o
**protocolo SSE** (meta → delta → done), que é o contrato com o JavaScript. Um tokenizer de
mentira (WordLevel) e um modelo tiny deixam tudo rodando sem baixar nada.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("pingo_webui", ROOT / "scripts" / "webui.py")
w = importlib.util.module_from_spec(_spec)
sys.modules["pingo_webui"] = w
_spec.loader.exec_module(w)

from pingo.config import PingoConfig  # noqa: E402
from pingo.model import PingoLM  # noqa: E402
from pingo.sft import ASSISTANT_TOKEN, USER_TOKEN  # noqa: E402

VOCAB = {"<unk>": 0, "<|padding|>": 1, "<|bos|>": 2, "<|eos|>": 3,
         "<|user|>": 4, "<|assistant|>": 5, "<|end|>": 6,
         "Pergunta:": 7, "Contexto:": 8, "oi": 9, "mundo": 10, "resposta": 11}


# ------------------------------------------------------------------ fixtures


def _checkpoint(raiz: Path, nome: str = "final") -> Path:
    """Salva um PingoLM tiny + tokenizer.json no formato que o `Server` espera ler."""
    d = raiz / nome
    d.mkdir(parents=True)
    cfg = PingoConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1,
                      num_attention_heads=2, intermediate_size=32,
                      max_position_embeddings=16)
    torch.manual_seed(0)
    PingoLM(cfg).save_pretrained(d)
    tok = Tokenizer(models.WordLevel(vocab=dict(VOCAB), unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    return d


@pytest.fixture()
def ambiente(tmp_path):
    """(app Flask, Server) com um único checkpoint SFT em tmp_path."""
    d = _checkpoint(tmp_path)
    servidor = w.Server([w.ModelSpec("sft", "SFT de teste", d, True)], device="cpu")
    app = w.create_app(servidor)
    app.config.update(TESTING=True)
    return app.test_client(), servidor


def _eventos(corpo: str) -> list[tuple[str, dict]]:
    """Separa o corpo SSE em [(evento, dados)] — o mesmo parser do JavaScript."""
    saida = []
    for bloco in corpo.strip().split("\n\n"):
        evento = dados = None
        for linha in bloco.splitlines():
            if linha.startswith("event: "):
                evento = linha[7:]
            elif linha.startswith("data: "):
                dados = json.loads(linha[6:])
        if evento:
            saida.append((evento, dados))
    return saida


# ------------------------------------------------------------------ servidor


def test_specs_ignoram_diretorio_inexistente(tmp_path):
    servidor = w.Server([w.ModelSpec("nada", "x", tmp_path / "nao-existe", True)], device="cpu")
    assert servidor.specs == {}


def test_carregar_le_tokenizer_do_checkpoint(ambiente):
    _cliente, servidor = ambiente
    carregado = servidor.carregar("sft")
    assert carregado.spec.id == "sft"
    assert carregado.end_id == VOCAB["<|end|>"]
    assert carregado.tokenizer.get_vocab_size() == len(VOCAB)
    assert next(carregado.model.parameters()).device.type == "cpu"
    # segunda chamada usa o cache — mesmo objeto, sem recarregar do disco
    assert servidor.carregar("sft") is carregado


def test_trocar_de_modelo_descarrega_o_anterior(tmp_path):
    d = _checkpoint(tmp_path)
    servidor = w.Server([w.ModelSpec("sft", "SFT", d, True),
                         w.ModelSpec("base", "Base", d, False)], device="cpu")
    a = servidor.carregar("sft")
    b = servidor.carregar("base")
    assert b.spec.id == "base" and b.end_id is None
    assert servidor._loaded is b and a is not b
    info = servidor.info()
    assert info["carregado"] == "base"
    assert [m["id"] for m in info["modelos"]] == ["sft", "base"]  # instrução primeiro


# ------------------------------------------------------------------ prompt/params


def test_montar_prompt_segue_o_formato_do_treino():
    assert w.montar_prompt("  Quanto custa?  ") == \
        f"{USER_TOKEN}Pergunta: Quanto custa?{ASSISTANT_TOKEN}"
    com_ctx = w.montar_prompt("Quanto custa?", " O plano custa 10 reais. ")
    assert com_ctx == (f"{USER_TOKEN}Contexto: O plano custa 10 reais.\n\n"
                       f"Pergunta: Quanto custa?{ASSISTANT_TOKEN}")


def test_montar_prompt_do_base_e_continuacao_crua():
    assert w.montar_prompt("Era uma vez", instrucao=False) == "Era uma vez"
    assert w.montar_prompt("Era uma vez", "Contexto", instrucao=False) == "Contexto\n\nEra uma vez"


def test_params_clampa_o_que_vem_do_navegador():
    p = w.params_de({"temperature": 9, "top_p": 0, "top_k": -3, "repetition_penalty": 3,
                     "max_new_tokens": 9999, "seed": "abc"})
    assert p == {"temperature": 2.0, "top_p": 0.05, "top_k": 0, "repetition_penalty": 2.0,
                 "max_new_tokens": 512, "seed": -1}


def test_delta_decoder_entrega_so_o_que_e_novo():
    tok = Tokenizer(models.WordLevel(vocab=dict(VOCAB), unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    dec = w.DeltaDecoder(tok)
    assert dec.push(VOCAB["oi"]) == "oi"
    assert dec.push(VOCAB["mundo"]) == " mundo"   # só o incremento
    assert dec.enviado == "oi mundo"


# ------------------------------------------------------------------ rotas


def test_pagina_e_status(ambiente):
    cliente, _ = ambiente
    r = cliente.get("/")
    assert r.status_code == 200 and "Pingo" in r.get_data(as_text=True)
    assert "WebUI" in r.get_data(as_text=True)

    s = cliente.get("/api/status").get_json()
    assert s["device"] == "cpu" and s["carregado"] is None
    assert [(m["id"], m["instrucao"], m["carregado"]) for m in s["modelos"]] == \
        [("sft", True, False)]
    assert s["modelos"][0]["caminho"].endswith("final")   # de onde o peso veio


def test_chat_emite_meta_delta_e_done(ambiente):
    cliente, servidor = ambiente
    r = cliente.post("/api/chat", json={"mensagem": "oi", "modelo": "sft",
                                        "max_new_tokens": 4, "temperature": 0})
    assert r.status_code == 200
    assert r.mimetype == "text/event-stream"
    eventos = _eventos(r.get_data(as_text=True))
    nomes = [e for e, _ in eventos]
    assert nomes[0] == "meta" and nomes[-1] == "done"
    meta = eventos[0][1]
    assert meta["modelo"] == "sft" and meta["instrucao"] is True
    tok = servidor.carregar("sft").tokenizer
    esperado = len(tok.encode(w.montar_prompt("oi"), add_special_tokens=False).ids)
    assert meta["prompt_tokens"] == esperado    # o prompt que a UI mandou é o do treino
    done = eventos[-1][1]
    assert done["tokens"] <= 4 and done["parada"] in ("end_token", "max_tokens")
    # o texto dos deltas, junto, é uma string (o modelo tiny não diz nada com nexo)
    assert all(isinstance(d["t"], str) for e, d in eventos if e == "delta")
    # e o lock foi devolvido no fim
    assert servidor.gen_lock.acquire(blocking=False)
    servidor.gen_lock.release()


def test_chat_rejeita_entradas_invalidas(ambiente):
    cliente, servidor = ambiente
    assert cliente.post("/api/chat", json={"mensagem": "  "}).status_code == 400
    assert cliente.post("/api/chat", json={"mensagem": "oi",
                                           "modelo": "gpt-4"}).status_code == 400
    assert cliente.post("/api/chat", json={"mensagem": "x" * 5000}).status_code == 400
    # nada disso pode ter deixado o lock preso
    assert not servidor.gen_lock.locked()


def test_chat_recusa_duas_geracoes_ao_mesmo_tempo(ambiente):
    cliente, servidor = ambiente
    assert servidor.gen_lock.acquire(blocking=False)
    try:
        r = cliente.post("/api/chat", json={"mensagem": "oi"})
        assert r.status_code == 429
        assert "gerando" in r.get_json()["erro"]
    finally:
        servidor.gen_lock.release()


def test_chat_sem_checkpoint_devolve_503(tmp_path):
    servidor = w.Server([], device="cpu")
    app = w.create_app(servidor)
    app.config.update(TESTING=True)
    r = app.test_client().post("/api/chat", json={"mensagem": "oi"})
    assert r.status_code == 503


def test_prompt_maior_que_o_contexto_devolve_400(tmp_path):
    d = _checkpoint(tmp_path)          # ctx 16 tokens
    servidor = w.Server([w.ModelSpec("sft", "SFT", d, True)], device="cpu")
    app = w.create_app(servidor)
    app.config.update(TESTING=True)
    r = app.test_client().post("/api/chat", json={"mensagem": "oi mundo resposta " * 40})
    assert r.status_code == 400
    assert "contexto" in r.get_json()["erro"]
    assert not servidor.gen_lock.locked()
