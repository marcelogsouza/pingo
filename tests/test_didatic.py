"""Testes do modo didático (`pingo/didatic.py`) — CPU, sem rede e sem o tokenizer real.

Duas famílias:

- **byte-level puro**: o alfabeto que faz `ĠatÃ©` virar `b" at\\xc3\\xa9"` é a alma do passo 1
  e não depende de nenhum arquivo — se ele quebrar, a tela mostra texto errado e o curso
  inteiro ensina bobagem.
- **tokenização/ embeddings**: um `Tokenizer` WordLevel de mentira e um checkpoint tiny
  escrito em `tmp_path` (o `save_pretrained` grava o `embed.weight` que o lab lê na CPU).

O teste com o tokenizer real é pulado nas máquinas onde `tokenizer/` e `runs/` não existem
(são gitignored) — igual ao resto da suíte.
"""

from pathlib import Path

import pytest
import torch
from tokenizers import Regex, Tokenizer, models, pre_tokenizers

from pingo.config import PingoConfig
from pingo.didatic import (MAX_LAB_TOKENS, TabelaEmbeddings, buscar, bytes_para_superficie,
                           categoria, especial_ids, superficie_para_bytes, texto_legivel,
                           tokenizar)
from pingo.model import PingoLM

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_REAL = next((p for p in (ROOT / "runs" / "sft-120m-ep1" / "final" / "tokenizer.json",
                                   ROOT / "tokenizer" / "tokenizer.json") if p.exists()), None)

VOCAB = {"<unk>": 0, "<|padding|>": 1, "<|bos|>": 2, "<|eos|>": 3, "<|user|>": 4,
         "<|assistant|>": 5, "<|end|>": 6, "Pergunta": 7, "Contexto": 8, "oi": 9,
         "mundo": 10, "2028": 11, ",": 12, "não": 13, ".": 14}


def tokenizer_fake() -> Tokenizer:
    r"""WordLevel de mentira. O `Split(\s+)` (e não o `Whitespace`) é de propósito: o
    `Whitespace` do tokenizers também separa pontuação e estilhaçaria `<|user|>`."""
    tok = Tokenizer(models.WordLevel(vocab=dict(VOCAB), unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(pattern=Regex(r"\s+"), behavior="removed")
    return tok


# ------------------------------------------------------------------ byte-level


def test_alfabeto_byte_level_e_o_do_tokenizers():
    assert superficie_para_bytes("ĠatÃ©") == b" at\xc3\xa9"
    assert superficie_para_bytes("Ã§Ã£o") == "ção".encode()
    assert superficie_para_bytes("ĠJoÃ£o") == " João".encode()
    assert superficie_para_bytes("Ċ") == b"\n"
    assert bytes_para_superficie(" preço".encode()) == "ĠpreÃ§o"


def test_texto_legivel_devolve_o_que_um_humano_le():
    assert texto_legivel("ĠpreÃ§o") == " preço"
    assert texto_legivel("Ã©") == "é"
    assert texto_legivel("ĠatÃ") == " at\ufffd"       # byte no meio do caractere: honesto


def test_categoria_do_especial_ao_fim_da_palavra():
    assert categoria("<|user|>", "<|user|>", especial=True) == "especial"
    assert categoria("Ċ", "\n") == "linha"
    assert categoria("â", "≈") == "bytes"              # byte puro ≥ 0x80 = byte-fallback
    assert categoria("ĠJoÃ£o", " João") == "início"    # começa com o byte do espaço
    assert categoria("Ã£o", "ão") == "continuação"
    assert categoria(",", ",") == "pontuação"
    assert categoria("Ġ(", " (") == "pontuação"        # espaço + símbolo: ainda é pontuação
    assert categoria("Ġ202", " 202") == "número"
    assert categoria("2028", "2028") == "número"


# ------------------------------------------------------------------ tokenizar


def test_tokenizar_cobre_o_prompt_inteiro_com_ids_e_categorias():
    tok, prompt = tokenizer_fake(), "<|user|> Pergunta oi mundo 2028 , não ."
    saida = tokenizar(tok, prompt)
    # o WordLevel de mentira consome o espaço (num BPE byte-level o espaço vai dentro do
    # token — é o que o teste com o tokenizer real abaixo verifica)
    assert "".join(t["texto"] for t in saida["tokens"]) == prompt.replace(" ", "")
    assert [t["id"] for t in saida["tokens"]] == [4, 7, 9, 10, 11, 12, 13, 14]
    assert saida["tokens"][0]["kind"] == "especial"
    assert {t["kind"] for t in saida["tokens"]} == {"especial", "continuação", "número",
                                                    "pontuação"}
    r = saida["resumo"]
    assert (r["chars"], r["n_tokens"], r["vocab"]) == (len(prompt), 8, len(VOCAB))
    assert r["especiais"] == 1 and r["cortado_em"] is None
    assert r["acentuados"] == 1                        # é o "não" (o vocab fake é latin-1)


def test_tokenizar_marca_repeticoes_e_acentos():
    tok = tokenizer_fake()
    saida = tokenizar(tok, "oi mundo oi oi")
    repetidos = [t["repetido"] for t in saida["tokens"]]
    assert repetidos == [None, None, 0, 0]             # a 1ª vez é None; depois, o índice da 1ª
    assert saida["resumo"]["repetidos"] == 2 and saida["resumo"]["distintos"] == 2
    com_acento = tokenizar(tok, "não")["tokens"][0]
    assert com_acento["acentuado"] and com_acento["bytes"]     # byte ≥ 0x80 em algum lugar


def test_tokenizar_corta_em_512_e_avisa():
    tok = tokenizer_fake()
    dito = " ".join(["oi"] * (MAX_LAB_TOKENS + 40))
    saida = tokenizar(tok, dito)
    assert len(saida["tokens"]) == MAX_LAB_TOKENS
    assert saida["resumo"]["cortado_em"] == MAX_LAB_TOKENS + 40


def test_especial_ids_pega_os_tokens_de_chat_mesmo_sem_added_tokens():
    ids = especial_ids(tokenizer_fake())
    assert {0, 2, 3, 4, 5, 6} <= ids                   # no fixture são vocab comum, não added


# ------------------------------------------------------------------ embeddings


def checkpoint_tiny(raiz: Path) -> Path:
    d = raiz / "final"
    d.mkdir(parents=True)
    torch.manual_seed(7)
    cfg = PingoConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1,
                      num_attention_heads=2, intermediate_size=32, max_position_embeddings=16)
    PingoLM(cfg).save_pretrained(d)
    return d


def test_tabela_le_so_o_embed_weight_sem_construir_o_modelo(tmp_path):
    tabela = TabelaEmbeddings.do_checkpoint(checkpoint_tiny(tmp_path))
    assert tabela.pesos.shape == (len(VOCAB), 16)
    assert tabela.pesos.device.type == "cpu"
    assert torch.allclose(tabela.normas, tabela.pesos.norm(dim=1))


def test_tabela_recusa_checkpoint_sem_a_tabela(tmp_path):
    d = tmp_path / "vazio"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"nada")
    with pytest.raises(Exception):
        TabelaEmbeddings.do_checkpoint(d)


def test_ficha_tem_os_896_numeros_a_norma_e_os_vizinhos(tmp_path):
    tabela = TabelaEmbeddings.do_checkpoint(checkpoint_tiny(tmp_path)).vocabulario(tokenizer_fake())
    f = tabela.ficha(9, k=4)
    assert f["dim"] == 16 and len(f["valores"]) == 16
    assert f["token"] == "oi" and f["texto"] == "oi" and f["bytes"] == "6f 69"
    assert f["norma"] == pytest.approx(float(tabela.pesos[9].norm()), abs=1e-4)
    assert f["escala"] == pytest.approx(float(tabela.pesos.abs().max()), abs=1e-4)
    cossenos = [v["cos"] for v in f["vizinhos"]]
    assert len(f["vizinhos"]) == 4 and cossenos == sorted(cossenos, reverse=True)
    assert 9 not in [v["id"] for v in f["vizinhos"]]   # o próprio token não é vizinho de si
    assert f["dim_mais_ativa"]["valor"] == max(f["valores"], key=abs)


def test_ficha_fora_do_vocab_e_erro_claro(tmp_path):
    tabela = TabelaEmbeddings.do_checkpoint(checkpoint_tiny(tmp_path))
    with pytest.raises(IndexError):
        tabela.ficha(999)


# ------------------------------------------------------------------ passo ②


def test_faixa_resume_o_vetor_preservando_os_picos(tmp_path):
    tabela = TabelaEmbeddings.do_checkpoint(checkpoint_tiny(tmp_path))
    f = tabela.faixa(9, bins=4)                       # dim 16 → grupos de 4
    v = tabela.pesos[9]
    assert len(f) == 4
    for i, valor in enumerate(f):
        grupo = v[i * 4:(i + 1) * 4]
        esperado = float(grupo[grupo.abs().argmax()])
        assert valor == pytest.approx(esperado, abs=1e-4)


def test_matriz_da_frase_linha_por_token_e_corte_honesto(tmp_path):
    tabela = TabelaEmbeddings.do_checkpoint(checkpoint_tiny(tmp_path)).vocabulario(tokenizer_fake())
    m = tabela.matriz([4, 9, 10, 9], bins=4)
    assert m["dim"] == 16 and m["bins"] == 4 and m["cortado_em"] is None
    assert [l["id"] for l in m["linhas"]] == [4, 9, 10, 9]
    assert [l["token"] for l in m["linhas"]] == ["<|user|>", "oi", "mundo", "oi"]
    assert all(len(l["faixa"]) == 4 for l in m["linhas"])
    assert m["linhas"][1]["norma"] == pytest.approx(float(tabela.pesos[9].norm()), abs=5e-4)
    cheio = tabela.matriz([9] * (96 + 24), bins=4)
    assert len(cheio["linhas"]) == 96 and cheio["cortado_em"] == 120
    with pytest.raises(IndexError):
        tabela.matriz([9, 999])


def test_estatisticas_histograma_soma_o_vocab_e_os_extremos_tem_nome(tmp_path):
    tabela = TabelaEmbeddings.do_checkpoint(checkpoint_tiny(tmp_path)).vocabulario(tokenizer_fake())
    s = tabela.estatisticas(bins=5, extremos=3)
    assert (s["vocab"], s["dim"]) == (len(VOCAB), 16)
    assert sum(b["cont"] for b in s["histograma"]) == len(VOCAB)
    assert len(s["histograma"]) == 5
    n = tabela.normas
    assert s["norma"]["media"] == pytest.approx(float(n.mean()), abs=1e-3)
    assert s["norma"]["mediana"] == pytest.approx(float(torch.quantile(n, 0.5)), abs=1e-3)
    assert [x["norma"] for x in s["maiores"]] == sorted([x["norma"] for x in s["maiores"]],
                                                        reverse=True)
    assert s["maiores"][0]["norma"] == pytest.approx(float(n.max()), abs=1e-3)
    assert s["menores"][0]["norma"] == pytest.approx(float(n.min()), abs=1e-3)
    assert s["menores"][0]["token"] in tabela.id_para_token.values()


def test_lida_de_volta_a_nota_do_lm_head_e_produto_escalar_com_a_tabela(tmp_path):
    tabela = TabelaEmbeddings.do_checkpoint(checkpoint_tiny(tmp_path)).vocabulario(tokenizer_fake())
    d = tabela.lida_de_volta(9, k=5)
    esperado = tabela.pesos @ tabela.pesos[9]
    assert d["id"] == 9 and d["token"] == "oi"
    assert d["pontos_consigo"] == pytest.approx(float(esperado[9]), abs=1e-4)
    assert [x["pontos"] for x in d["por_pontos"]] == sorted((x["pontos"] for x in d["por_pontos"]),
                                                            reverse=True)
    assert [x["pontos"] for x in d["por_pontos"]] == [round(float(v), 4) for v in
                                                      esperado.topk(5).values.tolist()]
    assert 9 not in [x["id"] for x in d["por_cos"]]   # de novo: ninguém é vizinho de si
    assert [x["cos"] for x in d["por_cos"]] == sorted((x["cos"] for x in d["por_cos"]),
                                                      reverse=True)
    with pytest.raises(IndexError):
        tabela.lida_de_volta(999)


# ------------------------------------------------------------------ busca


def test_busca_por_id_palavra_e_superficie():
    tok = tokenizer_fake()
    assert [a["id"] for a in buscar(tok, "9")] == [9]
    assert buscar(tok, "oi")[0]["token"] == "oi"
    assert buscar(tok, "20") == [{"id": 11, "token": "2028", "texto": "2028",
                                  "bytes": "32 30 32 38"}]
    assert buscar(tok, "   ") == [] and buscar(tok, "zzzz") == []


@pytest.mark.skipif(TOKENIZER_REAL is None, reason="tokenizer real não está nesta máquina")
def test_com_o_tokenizer_real_a_fatia_e_o_texto_e_a_superficie_e_o_byte_level():
    tok = Tokenizer.from_file(str(TOKENIZER_REAL))
    prompt = "<|user|>Pergunta: Olá, São João — não é o são João.<|assistant|>"
    saida = tokenizar(tok, prompt)
    assert "".join(t["texto"] for t in saida["tokens"]) == prompt   # offsets char a char
    assert saida["resumo"]["vocab"] == tok.get_vocab_size()
    porsuperficie = {t["superficie"]: t for t in saida["tokens"]}
    assert "ĠJoÃ£o" in porsuperficie and porsuperficie["ĠJoÃ£o"]["texto"] == " João"
    assert porsuperficie["ĠJoÃ£o"]["kind"] == "início"
    assert porsuperficie["ĠJoÃ£o"]["acentuado"]
    joao = [t for t in saida["tokens"] if t["superficie"] == "ĠJoÃ£o"]
    assert len(joao) == 2 and joao[1]["repetido"] == joao[0]["i"]
    busca = buscar(tok, "ção")
    assert any(a["texto"] == "ção" for a in busca)
