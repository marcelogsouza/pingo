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
from pingo.didatic import CATEGORIAS  # noqa: E402
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


def test_padrao_escuta_na_lan_e_porta_8010():
    """O chat é para ser aberto de outra máquina: bind em 0.0.0.0 (e 127.0.0.1 fecha)."""
    args = w.parse_args([])
    assert args.host == "0.0.0.0" and args.port == 8010
    assert w.parse_args(["--host", "127.0.0.1"]).host == "127.0.0.1"
    assert w.ip_da_lan() is None or w.ip_da_lan().count(".") == 3


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


# ------------------------------------------------------------------ laboratório (modo didático)


def test_lab_enolve_o_texto_com_os_tokens_de_chat(ambiente):
    cliente, servidor = ambiente
    r = cliente.post("/api/lab/tokenize", json={"texto": "oi mundo", "modelo": "sft"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["prompt"] == f"{USER_TOKEN}oi mundo{ASSISTANT_TOKEN}"   # sem o "Pergunta:"
    assert d["chat"] is True and d["modelo"] == "sft"
    ids = servidor.tokenizador_de("sft").encode(d["prompt"], add_special_tokens=False).ids
    assert [t["id"] for t in d["tokens"]] == ids      # a tela mostra exatamente o encode()
    assert d["resumo"]["n_tokens"] == len(ids) and d["resumo"]["vocab"] == len(VOCAB)
    assert {t["kind"] for t in d["tokens"]} <= set(CATEGORIAS)
    offsets = [t["offset"] for t in d["tokens"]]
    assert offsets == sorted(offsets) and all(0 <= a <= b <= len(d["prompt"]) for a, b in offsets)
    assert all(t["id"] == d["tokens"][t["repetido"]]["id"] for t in d["tokens"]
               if t["repetido"] is not None)


def test_montar_prompt_lab_e_so_o_involucao_o_chat_mantem_o_rotulo():
    assert w.montar_prompt_lab("oi mundo") == f"{USER_TOKEN}oi mundo{ASSISTANT_TOKEN}"
    assert w.montar_prompt_lab("Era uma vez", instrucao=False) == "Era uma vez"
    # o lab mostra o invólucro; o chat continua mandando o formato do treino
    assert w.montar_prompt("oi mundo") == f"{USER_TOKEN}Pergunta: oi mundo{ASSISTANT_TOKEN}"


def test_lab_do_base_nao_recebe_tokens_de_chat(tmp_path):
    d = _checkpoint(tmp_path)
    servidor = w.Server([w.ModelSpec("sft", "SFT", d, True),
                         w.ModelSpec("base", "Base", d, False)], device="cpu")
    app = w.create_app(servidor)
    app.config.update(TESTING=True)
    cliente = app.test_client()
    base = cliente.post("/api/lab/tokenize", json={"texto": "oi mundo", "modelo": "base"}).get_json()
    assert base["prompt"] == "oi mundo" and base["chat"] is False
    assert base["resumo"]["especiais"] == 0
    # e o `cru` de antes não muda mais nada: o invólucro é sempre o do chat
    sft = cliente.post("/api/lab/tokenize",
                       json={"texto": "oi mundo", "modelo": "sft", "cru": True}).get_json()
    assert sft["prompt"] == f"{USER_TOKEN}oi mundo{ASSISTANT_TOKEN}" and sft["chat"] is True
    assert servidor.tabela_de("sft") is not servidor.tabela_de("base")


def test_lab_rejeita_entradas_ruins(ambiente):
    cliente, servidor = ambiente
    assert cliente.post("/api/lab/tokenize", json={"texto": "   "}).status_code == 400
    assert cliente.post("/api/lab/tokenize", json={"texto": "x" * 5001}).status_code == 400
    assert cliente.post("/api/lab/tokenize", json={"texto": "oi",
                                                   "modelo": "gpt-4"}).status_code == 400
    assert not servidor.gen_lock.locked()      # o lab não usa o lock de geração


def test_lab_embedding_ve_do_safetensors_sem_carregar_o_modelo(ambiente):
    cliente, servidor = ambiente
    r = cliente.get("/api/lab/embedding?modelo=sft&id=9")
    assert r.status_code == 200
    f = r.get_json()
    assert f["id"] == 9 and f["token"] == "oi" and f["texto"] == "oi"
    assert f["dim"] == 16 and len(f["valores"]) == 16
    assert len(f["vizinhos"]) == 8 and 9 not in [v["id"] for v in f["vizinhos"]]
    assert servidor._loaded is None                       # nada foi para a GPU/CPU de modelo
    assert not servidor.gen_lock.locked()
    assert servidor.tabela_de("sft") is servidor.tabela_de("sft")   # cacheada por modelo


def test_lab_embedding_valida_o_id(ambiente):
    cliente, _ = ambiente
    assert cliente.get("/api/lab/embedding?modelo=sft&id=999").status_code == 400
    assert "fora do vocab" in cliente.get(
        "/api/lab/embedding?modelo=sft&id=999").get_json()["erro"]
    assert cliente.get("/api/lab/embedding?modelo=sft").status_code == 400
    assert cliente.get("/api/lab/embedding?modelo=ninguem&id=1").status_code == 400


def test_lab_matriz_e_uma_linha_por_token_sem_carregar_o_modelo(ambiente):
    cliente, servidor = ambiente
    r = cliente.post("/api/lab/matriz", json={"modelo": "sft", "ids": [4, 9, 10, 9]})
    assert r.status_code == 200
    m = r.get_json()
    assert m["dim"] == 16 and len(m["linhas"]) == 4
    assert [l["id"] for l in m["linhas"]] == [4, 9, 10, 9]      # a ordem da frase, com repetição
    assert all(len(l["faixa"]) == 16 for l in m["linhas"])      # bins=min(128, dim)=dim aqui
    assert m["linhas"][0]["token"] == "<|user|>"
    assert servidor._loaded is None                             # continua sem modelo na memória
    assert not servidor.gen_lock.locked()


def test_lab_matriz_rejeita_ids_ruins_e_capta_a_frase_gigante(ambiente):
    cliente, _ = ambiente
    assert cliente.post("/api/lab/matriz", json={"modelo": "sft"}).status_code == 400
    assert cliente.post("/api/lab/matriz", json={"modelo": "sft", "ids": []}).status_code == 400
    assert cliente.post("/api/lab/matriz",
                        json={"modelo": "sft", "ids": ["olá"]}).status_code == 400
    r = cliente.post("/api/lab/matriz", json={"modelo": "sft", "ids": [999]})
    assert r.status_code == 400 and "fora do vocab" in r.get_json()["erro"]
    d = cliente.post("/api/lab/matriz",
                     json={"modelo": "sft", "ids": [9] * 700}).get_json()
    assert len(d["linhas"]) == 96 and d["cortado_em"] == 512   # teto de ids, depois teto de linhas


def test_lab_stats_mostra_a_tabela_inteira(ambiente):
    cliente, servidor = ambiente
    s = cliente.get("/api/lab/stats?modelo=sft").get_json()
    assert (s["vocab"], s["dim"]) == (len(VOCAB), 16)
    assert sum(b["cont"] for b in s["histograma"]) == len(VOCAB)
    assert s["maiores"][0]["norma"] >= s["menores"][0]["norma"]
    assert servidor._loaded is None and not servidor.gen_lock.locked()
    assert cliente.get("/api/lab/stats?modelo=ninguem").status_code == 400


def test_lab_lida_de_volta_traz_nota_e_cosseno(ambiente):
    cliente, servidor = ambiente
    d = cliente.get("/api/lab/lida?modelo=sft&id=9").get_json()
    assert d["id"] == 9 and d["token"] == "oi"
    assert len(d["por_pontos"]) == 8 and len(d["por_cos"]) == 8
    assert [x["pontos"] for x in d["por_pontos"]] == sorted((x["pontos"] for x in d["por_pontos"]),
                                                            reverse=True)
    assert 9 not in [x["id"] for x in d["por_cos"]]
    tabela = servidor.tabela_de("sft")
    esperado = tabela.pesos @ tabela.pesos[9]
    assert d["pontos_consigo"] == pytest.approx(float(esperado[9]), abs=1e-4)
    assert cliente.get("/api/lab/lida?modelo=sft&id=999").status_code == 400
    assert cliente.get("/api/lab/lida?modelo=sft").status_code == 400


# ---------------------------------------------------------------- passo ③ · RoPE+atenção


def test_lab_attn_pesos_reais_casuais_e_modelo_carregado(ambiente):
    cliente, servidor = ambiente
    r = cliente.post("/api/lab/attn", json={"modelo": "sft", "ids": [4, 9, 10, 5], "camada": 0})
    assert r.status_code == 200
    a = r.get_json()
    assert a["n_camadas"] == 1 and a["heads"] == 2 and a["n"] == 4
    pesos = a["pesos"]
    assert all(abs(sum(linha) - 1) < 1e-3 for h in pesos for linha in h)
    assert all(pesos[h][i][j] == 0.0 for h in range(2) for i in range(4) for j in range(i + 1, 4))
    assert servidor._loaded is not None            # ③ é a exceção deliberada: carrega o modelo


def test_lab_rope_vetores_reais_e_rotacao_fiel(ambiente):
    cliente, servidor = ambiente
    r = cliente.get("/api/lab/rope?modelo=sft&ids=4,9,10,5&camada=0&head=1&i=2")
    assert r.status_code == 200
    d = r.get_json()
    assert d["pos"] == 2 and len(d["inv_freq"]) == 4          # head_dim 8 → 4 ponteiros
    assert len(d["q_lin"]) == 8 and len(d["q"]) == 8
    assert d["inv_freq"] == sorted(d["inv_freq"], reverse=True)  # par 0 é o ponteiro veloz
    ql, q = torch.tensor(d["q_lin"]), torch.tensor(d["q"])
    assert torch.allclose(ql.norm(), q.norm(), atol=1e-4)     # rotação preserva a norma
    # a captura fica em cache (chave modelo+ids): trocar só a camada não refaz o forward
    antes = servidor._captura
    assert cliente.post("/api/lab/attn", json={"modelo": "sft", "ids": [4, 9, 10, 5],
                                               "camada": 0}).status_code == 200
    assert servidor._captura is antes


def test_lab_attn_valida_entradas_e_responde_429(ambiente):
    cliente, servidor = ambiente
    assert cliente.post("/api/lab/attn", json={"modelo": "sft", "ids": []}).status_code == 400
    assert cliente.post("/api/lab/attn", json={"modelo": "sft", "ids": ["x"]}).status_code == 400
    r = cliente.post("/api/lab/attn", json={"modelo": "sft", "ids": [4, 9], "camada": 7})
    assert r.status_code == 400 and "camada" in r.get_json()["erro"]
    assert cliente.get("/api/lab/rope?modelo=sft&ids=4,9&camada=0&head=9&i=0").status_code == 400
    servidor.gen_lock.acquire(blocking=False)      # o chat generating → 429, sem fila
    try:
        assert cliente.post("/api/lab/attn", json={"modelo": "sft", "ids": [4]}).status_code == 429
        assert cliente.get("/api/lab/rope?modelo=sft&ids=4").status_code == 429
    finally:
        servidor.gen_lock.release()
    d = cliente.post("/api/lab/attn",
                     json={"modelo": "sft", "ids": [4] * 700}).get_json()
    assert d["n"] == 64                            # teto de 64 tokens na captura


# ---------------------------------------------------------------- passo ④ · RMSNorm + SwiGLU


def test_lab_bloco_regua_gamma_e_swinglu(ambiente):
    cliente, servidor = ambiente
    r = cliente.post("/api/lab/bloco", json={"modelo": "sft", "ids": [4, 9, 10, 5],
                                             "camada": 0, "i": 1})
    assert r.status_code == 200
    d = r.get_json()
    assert d["modelo"] == "sft" and d["camada"] == 0 and d["n_camadas"] == 1
    assert (d["n"], d["d"], d["hidden"]) == (4, 16, 32)
    assert d["raiz_d"] == pytest.approx(16 ** 0.5, abs=0.01)
    assert [l["id"] for l in d["tokens"]] == [4, 9, 10, 5]
    assert d["tokens"][1]["token"] == "oi"
    for l in d["tokens"]:
        assert l["n1p"] == pytest.approx(d["raiz_d"], abs=0.1)   # a régua vale para todos
    assert len(d["gamma"]["mlp"]["faixa"]) == 16                  # bins=min(128, dim)=dim aqui
    assert len(d["mlp"]["s"]) == 32 and d["mlp"]["abertas"] >= 1  # faixa do MLP cap no hidden
    assert servidor._loaded is not None                           # carona do forward do ③
    assert not servidor.gen_lock.locked()


def test_lab_bloco_valida_entradas_e_responde_429(ambiente):
    cliente, servidor = ambiente
    assert cliente.post("/api/lab/bloco", json={"modelo": "sft", "ids": []}).status_code == 400
    assert cliente.post("/api/lab/bloco", json={"modelo": "sft", "ids": ["x"]}).status_code == 400
    r = cliente.post("/api/lab/bloco", json={"modelo": "sft", "ids": [4, 9], "camada": 7})
    assert r.status_code == 400 and "camada" in r.get_json()["erro"]
    r = cliente.post("/api/lab/bloco", json={"modelo": "sft", "ids": [4, 9], "i": 5})
    assert r.status_code == 400 and "fora da frase" in r.get_json()["erro"]
    servidor.gen_lock.acquire(blocking=False)
    try:
        assert cliente.post("/api/lab/bloco", json={"modelo": "sft", "ids": [4]}).status_code == 429
    finally:
        servidor.gen_lock.release()
    d = cliente.post("/api/lab/bloco", json={"modelo": "sft", "ids": [4] * 700}).get_json()
    assert d["n"] == 64                                           # mesmo teto de tokens do ③


def test_lab_bloco_e_attn_compartilham_a_captura(ambiente):
    """④ é carona: a captura por `modelo+ids` serve o ③ e o ④ — um forward só."""
    cliente, servidor = ambiente
    assert cliente.post("/api/lab/attn", json={"modelo": "sft", "ids": [4, 9, 10, 5],
                                               "camada": 0}).status_code == 200
    antes = servidor._captura
    assert cliente.post("/api/lab/bloco", json={"modelo": "sft", "ids": [4, 9, 10, 5],
                                                "camada": 0, "i": 2}).status_code == 200
    assert servidor._captura is antes


# ---------------------------------------------------------------- passo ⑤ · lm head


def test_lab_lmhead_norm_final_e_logits_reais(ambiente):
    cliente, servidor = ambiente
    ids = [4, 9, 10, 5]
    r = cliente.post("/api/lab/lmhead", json={"modelo": "sft", "ids": ids, "i": 1})
    assert r.status_code == 200
    d = r.get_json()
    assert d["modelo"] == "sft" and d["n_camadas"] == 1
    assert (d["n"], d["d"], d["vocab"], d["i"]) == (4, 16, len(VOCAB), 1)
    assert d["raiz_d"] == pytest.approx(16 ** 0.5, abs=0.01)
    for l in d["tokens"]:                          # a última régua vale para todos os tokens
        assert l["nfinp"] == pytest.approx(d["raiz_d"], abs=0.1)
    assert [x["logit"] for x in d["top"]] == sorted((x["logit"] for x in d["top"]), reverse=True)
    # os logits são os do próprio forward — não uma recomputação a partir da tabela
    esperado = servidor._captura[1]["logits"][1]
    assert d["top"][0]["logit"] == pytest.approx(float(esperado.max()), abs=5e-4)
    assert d["proximo"]["id"] == 10 and d["proximo"]["token"] == "mundo"
    assert d["proximo"]["rank"] == int((esperado > esperado[10]).sum()) + 1
    assert servidor._loaded is not None                      # carona do forward do ③
    assert not servidor.gen_lock.locked()
    # última posição da frase: sem sucessor — é o cardápio da geração (passo ⑥)
    assert cliente.post("/api/lab/lmhead",
                        json={"modelo": "sft", "ids": ids, "i": 3}).get_json()["proximo"] is None


def test_lab_lmhead_valida_entradas_e_responde_429(ambiente):
    cliente, servidor = ambiente
    assert cliente.post("/api/lab/lmhead", json={"modelo": "sft", "ids": []}).status_code == 400
    assert cliente.post("/api/lab/lmhead", json={"modelo": "sft", "ids": ["x"]}).status_code == 400
    r = cliente.post("/api/lab/lmhead", json={"modelo": "sft", "ids": [4, 9], "i": 5})
    assert r.status_code == 400 and "fora da frase" in r.get_json()["erro"]
    servidor.gen_lock.acquire(blocking=False)
    try:
        assert cliente.post("/api/lab/lmhead", json={"modelo": "sft", "ids": [4]}).status_code == 429
    finally:
        servidor.gen_lock.release()
    d = cliente.post("/api/lab/lmhead", json={"modelo": "sft", "ids": [4] * 700, "i": 0}).get_json()
    assert d["n"] == 64                                          # mesmo teto de tokens do ③


def test_lab_lmhead_e_attn_compartilham_o_mesmo_forward(ambiente):
    """⑤ também é carona: a captura do ③ já saiu com a norm final e os logits."""
    cliente, servidor = ambiente
    assert cliente.post("/api/lab/attn", json={"modelo": "sft", "ids": [4, 9, 10, 5],
                                               "camada": 0}).status_code == 200
    antes = servidor._captura
    assert cliente.post("/api/lab/lmhead", json={"modelo": "sft", "ids": [4, 9, 10, 5],
                                                 "i": 2}).status_code == 200
    assert servidor._captura is antes


# ---------------------------------------------------------------- passo ⑥ · amostragem (a roleta)


def test_lab_amostra_menu_vem_dos_logits_reais_caroneados(ambiente):
    cliente, servidor = ambiente
    ids = [4, 9, 10, 5]                       # <|user|> oi mundo <|assistant|>
    r = cliente.post("/api/lab/amostra", json={"modelo": "sft", "ids": ids})
    assert r.status_code == 200
    d = r.get_json()
    assert d["modelo"] == "sft" and d["n"] == 4 and d["vocab"] == len(VOCAB)
    assert d["acao"] == "menu" and "sorteado" not in d
    assert d["retidos"] == len(VOCAB)                       # T=1 sem filtro: todos giram
    graus = sum(c["graus"] for c in d["candidatos"])
    assert abs(graus - 360.0) < 0.6                         # a roda fecha em 360°
    probs = [c["prob"] for c in d["candidatos"]]
    assert probs == sorted(probs, reverse=True)
    assert {c["kind"] for c in d["candidatos"]} <= set(CATEGORIAS)
    linha = servidor._captura[1]["logits"][-1]              # a carona do ③–⑤, sem recomputar
    assert d["candidatos"][0]["id"] == int(linha.argmax())
    assert not servidor.gen_lock.locked()


def test_lab_amostra_e_lmhead_compartilham_a_captura(ambiente):
    """⑥ é carona do MESMO forward: a captura em cache não é refeita nem substituída."""
    cliente, servidor = ambiente
    assert cliente.post("/api/lab/lmhead", json={"modelo": "sft", "ids": [4, 9],
                                                 "i": 1}).status_code == 200
    antes = servidor._captura
    assert cliente.post("/api/lab/amostra", json={"modelo": "sft", "ids": [4, 9],
                                                  "acao": "girar"}).status_code == 200
    assert servidor._captura is antes


def test_lab_amostra_semente_repete_o_giro_e_o_filtro_corta(ambiente):
    cliente, _ = ambiente
    corpo = {"modelo": "sft", "ids": [4, 9], "seed": 7, "acao": "girar", "top_k": 3}
    a = cliente.post("/api/lab/amostra", json=corpo).get_json()
    b = cliente.post("/api/lab/amostra", json=corpo).get_json()
    assert a["retidos"] == 3 and len(a["candidatos"]) == 3
    assert a["sorteado"]["id"] == b["sorteado"]["id"]       # mesma semente, mesmo giro
    assert a["sorteado"]["id"] in {c["id"] for c in a["candidatos"]}
    assert a["end_id"] == VOCAB["<|end|>"]                  # a tela precisa saber onde a história para


def test_lab_amostra_forcar_obedece_ate_fora_do_filtro(ambiente):
    cliente, _ = ambiente
    alvo = VOCAB["resposta"]
    d = cliente.post("/api/lab/amostra", json={"modelo": "sft", "ids": [4, 9], "acao": "forcar",
                                               "token": alvo, "top_k": 1}).get_json()
    assert d["sorteado"]["id"] == alvo and d["sorteado"]["modo"] == "mão"
    dentro = d["candidatos"][0]["id"] == alvo
    assert d["sorteado"]["dentro"] is dentro
    if not dentro:
        assert d["sorteado"]["prob"] == 0.0 and d["sorteado"]["p_bruto"] > 0
    assert cliente.post("/api/lab/amostra", json={"modelo": "sft", "ids": [4, 9],
                                                  "acao": "forcar", "token": 999}).status_code == 400
    assert cliente.post("/api/lab/amostra", json={"modelo": "sft", "ids": [4, 9],
                                                  "acao": "nada"}).status_code == 400


def test_lab_amostra_valida_entradas_reserva_o_teto_e_para_duas(ambiente):
    cliente, servidor = ambiente
    assert cliente.post("/api/lab/amostra", json={"modelo": "sft", "ids": []}).status_code == 400
    assert cliente.post("/api/lab/amostra", json={"modelo": "sft", "ids": ["x"]}).status_code == 400
    d = cliente.post("/api/lab/amostra", json={"modelo": "sft", "ids": [9] * 400}).get_json()
    assert d["n"] == 160 and d["cortado_em"] == 160         # teto do laboratório, avisado
    assert servidor.gen_lock.acquire(blocking=False)
    try:
        assert cliente.post("/api/lab/amostra",
                            json={"modelo": "sft", "ids": [4, 9]}).status_code == 429
    finally:
        servidor.gen_lock.release()


def test_lab_desdobrar_continua_a_historia_ate_o_fim(ambiente):
    cliente, servidor = ambiente
    ids = [4, 9]                                            # <|user|> oi
    r = cliente.post("/api/lab/desdobrar", json={"modelo": "sft", "ids": ids,
                                                 "temperature": 0, "max_new_tokens": 6})
    assert r.status_code == 200
    d = r.get_json()
    assert d["modelo"] == "sft" and d["n_novos"] == len(d["tokens"])
    assert d["parada"] in ("end_token", "max_tokens")
    if d["parada"] == "end_token":
        assert d["tokens"][-1]["id"] == VOCAB["<|end|>"]    # o <|end|> é emitido e fecha
    assert all(t["token"] in VOCAB for t in d["tokens"])
    assert isinstance(d["texto"], str)
    assert not servidor.gen_lock.locked()


def test_lab_desdobrar_semente_reproduz_a_continuacao(ambiente):
    cliente, _ = ambiente
    corpo = {"modelo": "sft", "ids": [4, 9], "seed": 42, "temperature": 0.8, "top_p": 0.9,
             "max_new_tokens": 4}
    e = cliente.post("/api/lab/desdobrar", json=corpo).get_json()
    f = cliente.post("/api/lab/desdobrar", json=corpo).get_json()
    assert [t["id"] for t in e["tokens"]] == [t["id"] for t in f["tokens"]]
    assert e["segundos"] >= 0


def test_lab_desdobrar_valida_entradas(ambiente):
    cliente, _ = ambiente
    assert cliente.post("/api/lab/desdobrar", json={"modelo": "sft", "ids": []}).status_code == 400
    assert cliente.post("/api/lab/desdobrar",
                        json={"modelo": "ninguem", "ids": [4, 9]}).status_code == 400


# ------------------------------------------------------------------ passo ⑦: loop + KV cache


def test_lab_loop_devolve_o_replay_com_o_cache_em_cena(ambiente):
    cliente, servidor = ambiente
    r = cliente.post("/api/lab/loop", json={"modelo": "sft", "ids": [4, 9], "temperature": 0,
                                            "max_new_tokens": 4})
    assert r.status_code == 200
    d = r.get_json()
    assert d["modelo"] == "sft" and d["n_prefill"] == 2 and d["n_novos"] == 4
    ps = d["passos"]
    assert ps[0]["modo"] == "prefill" and ps[0]["no_forward"] == 2 and ps[0]["escrito"] == 2
    assert all(p["modo"] == "decode" and p["no_forward"] == 1 and p["escrito"] == 1 for p in ps[1:])
    ditos = [p["escolhido"]["id"] for p in ps if p.get("escolhido")]
    assert len(ditos) == 4
    assert all(depois["entrou"]["id"] == antes for antes, depois in zip(ditos, ps[1:]))
    assert d["economia"]["com_cache"] == 2 + 4                    # prefill + 1 por step
    assert d["economia"]["sem_cache"] > d["economia"]["com_cache"]
    assert d["fita"]["cur_len"] == 2 + 4 and len(d["fita"]["normas_k"]) == 1
    c = d["compara"]
    assert c["tokens"] == 6 and c["maxdiff"] < 1e-4 and c["argmax_igual"] is True
    assert not servidor.gen_lock.locked()


def test_lab_loop_semente_reproduz_a_historia(ambiente):
    cliente, _ = ambiente
    corpo = {"modelo": "sft", "ids": [4, 9], "seed": 42, "temperature": 0.8,
             "top_k": 3, "max_new_tokens": 4}
    a = cliente.post("/api/lab/loop", json=corpo).get_json()
    b = cliente.post("/api/lab/loop", json=corpo).get_json()
    ditos = lambda d: [p["escolhido"]["id"] for p in d["passos"] if p.get("escolhido")]
    assert a["semente"] == 42 and ditos(a) == ditos(b)


def test_lab_loop_valida_entradas_e_reserva_o_gen_lock(ambiente):
    cliente, servidor = ambiente
    assert cliente.post("/api/lab/loop", json={"modelo": "sft", "ids": []}).status_code == 400
    assert cliente.post("/api/lab/loop", json={"modelo": "sft", "ids": ["x"]}).status_code == 400
    assert cliente.post("/api/lab/loop",
                        json={"modelo": "ninguem", "ids": [4, 9]}).status_code == 400
    r = cliente.post("/api/lab/loop", json={"modelo": "sft", "ids": [4] * 20})   # max_pos 16
    assert r.status_code == 400 and "contexto" in r.get_json()["erro"]
    d = cliente.post("/api/lab/loop", json={"modelo": "sft", "ids": [4, 9],
                                            "max_new_tokens": 999}).get_json()
    assert d["parada"] in ("contexto", "end_token", "max_tokens")
    assert d["n_novos"] <= 16 - 2             # quem manda no teto é o contexto do modelo (16)
    assert servidor.gen_lock.acquire(blocking=False)
    try:
        assert cliente.post("/api/lab/loop", json={"modelo": "sft", "ids": [4, 9]}).status_code == 429
    finally:
        servidor.gen_lock.release()


# ------------------------------------------------------------------ busca de vocab


def test_lab_lookup_por_id_e_por_palavra(ambiente):
    cliente, _ = ambiente
    por_id = cliente.get("/api/lab/lookup?modelo=sft&q=9").get_json()["achados"]
    assert [a["id"] for a in por_id][:1] == [9] and por_id[0]["token"] == "oi"
    por_texto = cliente.get("/api/lab/lookup?modelo=sft&q=mu").get_json()["achados"]
    assert any(a["token"] == "mundo" for a in por_texto)
    assert cliente.get("/api/lab/lookup?modelo=sft&q=").status_code == 400


def test_lab_exige_tokenizer_do_mesmo_vocab_que_a_tabela(tmp_path):
    """16.384 × 16.387 é o tipo de erro que faria a ficha mostrar o vetor do token errado."""
    d = _checkpoint(tmp_path)                      # modelo com vocab = len(VOCAB)
    menor = {t: i for i, t in enumerate(list(VOCAB)[:len(VOCAB) - 3])}
    Tokenizer(models.WordLevel(vocab=menor, unk_token="<unk>")).save(str(d / "tokenizer.json"))
    servidor = w.Server([w.ModelSpec("sft", "SFT", d, True)], device="cpu")
    app = w.create_app(servidor)
    app.config.update(TESTING=True)
    r = app.test_client().get("/api/lab/embedding?modelo=sft&id=1")
    assert r.status_code == 400 and "tokenizer" in r.get_json()["erro"]


def test_lab_pagina_e_o_link_do_chat(ambiente):
    cliente, _ = ambiente
    corpo = cliente.get("/lab").get_data(as_text=True)
    assert cliente.get("/lab").status_code == 200
    assert "tokenização" in corpo and "/api/lab/tokenize" in corpo and "em breve" in corpo
    assert "/api/lab/matriz" in corpo and "/api/lab/stats" in corpo and "/api/lab/lida" in corpo
    assert "/api/lab/attn" in corpo and "/api/lab/rope" in corpo and "/api/lab/bloco" in corpo
    assert "/api/lab/lmhead" in corpo
    assert "/api/lab/amostra" in corpo and "/api/lab/desdobrar" in corpo
    assert "/api/lab/loop" in corpo                              # ⑦ no ar: a aba deixou de estar "off"
    assert "['loop: o token volta', true]" in corpo
    pagina_do_chat = cliente.get("/").get_data(as_text=True)
    assert 'href="/lab"' in pagina_do_chat

