"""Testes do SFT-1 (scripts/gen_sft.py) — CPU, sem rede: o professor é substituído por
um fake (`ask_teacher` monkeypatched).

O contrato que importa: o dataset só aceita pares cuja `evidencia` é cópia literal do
trecho (a única defesa barata contra invenção), recusas vêm no formato padrão, e um trecho
ruim é descartado sem travar o run.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pingo_gen_sft", ROOT / "scripts" / "gen_sft.py")
gen = importlib.util.module_from_spec(_spec)
sys.modules["pingo_gen_sft"] = gen  # dataclass precisa do módulo registrado
_spec.loader.exec_module(gen)


CHUNK = (
    "A Companhia Energética de Minas Gerais anunciou um plano de investimentos de 12 bilhões "
    "de reais até 2028, com foco em redes de distribuição no interior do estado. "
    "O programa prevê a contratação de 3 mil eletricistas em regime de tempo integral."
)
PERGUNTA = "Quanto a companhia planeja investir até 2028?"
EVIDENCIA = "um plano de investimentos de 12 bilhões de reais até 2028"


def qa_ok(tipo="extracao", **over):
    d = {"status": "ok", "tipo": tipo, "pergunta": PERGUNTA,
         "resposta": "A companhia planeja investir 12 bilhões de reais até 2028.",
         "evidencia": EVIDENCIA}
    d.update(over)
    return d


# ------------------------------------------------------------------ filtro de fonte


def test_clean_paragraphs_tira_boilerplate_e_linhas_curtas():
    content = "\n".join([
        "Facebook",
        "Compartilhe nas redes sociais e assine a nossa newsletter para mais notícias.",
        "curta",
        "A Companhia Energética de Minas Gerais anunciou um plano de investimentos grande.",
        "https://a.com https://b.com https://c.com",
        "Leia mais sobre isso no nosso site oficial, sempre com novidades para você.",
    ])
    paras = gen.clean_paragraphs(content)
    assert paras == [content.split("\n")[3]]


def test_doc_reject_reason():
    cfg = gen.Config()
    bom = "\n".join([f"Parágrafo número {i} com conteúdo textual suficiente para passar "
                     f"no filtro de parágrafos do gerador." for i in range(5)]) * 6
    assert gen.doc_reject_reason(bom, cfg) is None
    assert gen.doc_reject_reason("curto demais", cfg) == "doc_curto"
    poucos = ("Uma única linha bem longa de texto em português, com palavras de verdade "
              "repetidas várias vezes para passar do tamanho mínimo. ") * 40
    assert gen.doc_reject_reason(poucos, cfg) == "poucos_paragrafos"
    numerico = "\n".join(["12 34 56 78 90 1234 5678 9012 3456 7890 12345 67890 111 222"
                          for _ in range(60)])
    assert gen.doc_reject_reason(numerico, cfg) is not None
    gritando = "\n".join(["TEXTO TODO EM MAIÚSCULAS GRITANDO NO SITE INTEIRO DO GOVERNO"
                          for _ in range(60)])
    assert gen.doc_reject_reason(gritando, cfg) == "muitas_maiusculas"


def test_make_chunks_respeita_limites_e_nao_corta_frases():
    paras = [" ".join([f"Frase {i} do parágrafo com algumas palavras."] * 5) for i in range(12)]
    cfg = gen.Config(chunk_chars=600, chunk_min_chars=200)
    chunks = gen.make_chunks(paras, cfg)
    assert len(chunks) > 1
    joined = " ".join(chunks)
    for par in paras:  # nenhum parágrafo ficou pela metade
        assert par in joined
    for c in chunks:
        assert len(c) <= cfg.chunk_chars or len(c.split(".")) > 1
    assert all(len(c) >= cfg.chunk_min_chars for c in chunks)


def test_paragrafo_gigante_e_quebrado_em_frases():
    gigante = " ".join(["Esta é uma frase completa do texto original."] * 60)
    cfg = gen.Config(chunk_chars=500, chunk_min_chars=100)
    chunks = gen.make_chunks([gigante], cfg)
    assert len(chunks) > 1
    assert all(len(c) <= cfg.chunk_chars + 100 for c in chunks)


# ------------------------------------------------------------------ tipos


def test_pick_tipo_respeita_fracao_de_recusa():
    import random
    assert all(gen.pick_tipo(random.Random(i), 1.0) == "recusa" for i in range(50))
    tipos = {gen.pick_tipo(random.Random(i), 0.0) for i in range(200)}
    assert tipos == set(gen.TIPO_WEIGHTS)


# ------------------------------------------------------------------ validação


def test_validate_aceita_par_bom():
    motivo, pergunta, resposta, evidencia = gen.validate(qa_ok(), CHUNK, "extracao")
    assert motivo == ""
    assert pergunta == PERGUNTA
    assert evidencia.startswith("um plano de investimentos")


def test_validate_rejeita_evidencia_inventada():
    qa = qa_ok(evidencia="um plano de investimentos de 15 bilhões de reais até 2030")
    assert gen.validate(qa, CHUNK, "extracao")[0] == "evidencia_nao_literal"


def test_validate_aceita_evidencia_com_espacos_diferentes():
    qa = qa_ok(evidencia="um plano   de\ninvestimentos de 12 bilhões de reais até 2028")
    assert gen.validate(qa, CHUNK, "extracao")[0] == ""


def test_validate_recusa():
    ok = qa_ok(tipo="recusa", resposta="Isso não consta no texto.", evidencia="")
    assert gen.validate(ok, CHUNK, "recusa")[0] == ""
    com_ev = qa_ok(tipo="recusa", resposta="Isso não consta no texto.", evidencia=EVIDENCIA)
    assert gen.validate(com_ev, CHUNK, "recusa")[0] == "recusa_com_evidencia"
    respondida = qa_ok(tipo="recusa")
    assert gen.validate(respondida, CHUNK, "recusa")[0] == "recusa_ruim"


def test_validate_outros_motivos():
    assert gen.validate(None, CHUNK, "extracao")[0] == "json_invalido"
    assert gen.validate(qa_ok(status="invalido"), CHUNK, "extracao")[0] == "fonte_invalida"
    assert gen.validate(qa_ok(), CHUNK, "resumo")[0] == "tipo_diferente"
    assert gen.validate(qa_ok(pergunta="Quanto a companhia planeja investir até 2028"),
                        CHUNK, "extracao")[0] == "pergunta_ruim"
    assert gen.validate(qa_ok(pergunta="What did the company announce about its plan?"),
                        CHUNK, "extracao")[0] == "nao_pt"
    assert gen.validate(qa_ok(resposta="Was"), CHUNK, "extracao")[0] == "resposta_ruim"


# ------------------------------------------------------------------ chunks + professor


def _chunk(idx=0, url="https://exemplo.com.br/noticia"):
    return gen.Chunk("doc1", url, "train", "2024-10", idx, CHUNK)


def test_process_chunk_aceita_e_conta_tokens(monkeypatch):
    import threading
    monkeypatch.setattr(gen, "ask_teacher",
                        lambda cfg, c, tipo, anc, seed: (qa_ok(tipo), {"completion_tokens": 40}, ""))
    res = gen.process_chunk(_chunk(), gen.Config(recusa_frac=0.0), set(), threading.Lock())
    assert len(res.records) == 1 and res.failures == [] and res.completion_tokens == 40
    rec = res.records[0]
    assert rec["doc_id"] == "doc1" and rec["chunk_text"] == CHUNK
    assert rec["prompt_version"] == gen.PROMPT_VERSION


def test_process_chunk_reamostra_ate_acertar(monkeypatch):
    import threading
    calls = []

    def fake(cfg, c, tipo, anc, seed):
        calls.append(tipo)
        if len(calls) < 2:
            return qa_ok(tipo, evidencia="texto que não existe no trecho"), {}, ""
        return qa_ok(tipo), {}, ""

    monkeypatch.setattr(gen, "ask_teacher", fake)
    res = gen.process_chunk(_chunk(), gen.Config(tries=3, recusa_frac=0.0), set(),
                            threading.Lock())
    assert len(res.records) == 1
    assert res.failures == ["evidencia_nao_literal"]


def test_process_chunk_nao_trava_quando_tudo_falha(monkeypatch):
    import threading
    monkeypatch.setattr(gen, "ask_teacher",
                        lambda cfg, c, tipo, anc, seed: (None, {}, "http:URLError"))
    res = gen.process_chunk(_chunk(), gen.Config(tries=2), set(), threading.Lock())
    assert res.records == [] and res.failures == ["http:URLError"] * 2


def test_modo_ancorado_muda_prompt_e_recusa():
    anc = gen.build_body(gen.Config(), CHUNK, "recusa", True, 1)["messages"][1]["content"]
    liv = gen.build_body(gen.Config(), CHUNK, "recusa", False, 1)["messages"][1]["content"]
    assert "ANCORADO" in anc and "SEM CONTEXTO" in liv
    assert gen.RECUSA_ANCORADO.rstrip(".") in anc
    assert gen.RECUSA_LIVRE.rstrip(".") in liv


def test_validate_fonte_so_vale_sem_contexto():
    pergunta = "Qual foi a taxa de desemprego citada no texto em 2024?"
    qa = qa_ok(pergunta=pergunta)
    assert gen.validate(qa, CHUNK, "extracao", ancorado=True)[0] == ""
    assert gen.validate(qa, CHUNK, "extracao", ancorado=False)[0] == "menciona_fonte"


def test_validate_recusa_por_modo():
    livre = qa_ok(tipo="recusa", resposta=gen.RECUSA_LIVRE, evidencia="")
    assert gen.validate(livre, CHUNK, "recusa", ancorado=False)[0] == ""
    assert gen.validate(livre, CHUNK, "recusa", ancorado=True)[0] == "recusa_ruim"


def test_deny_list_de_fonte():
    cfg = gen.Config()
    bom = "\n".join(["O programa prevê a modernização de subestações e a instalação de "
                     "medidores inteligentes em várias cidades do interior do estado."] * 25)
    assert gen.doc_reject_reason(bom, cfg, "https://noticias.com.br/a") is None
    assert gen.doc_reject_reason(bom, cfg, "https://gruposdeputariatelegram.com/x") == "fonte_bloqueada"
    assert gen.doc_reject_reason(bom, cfg, "https://downloadfreecracks.com/a") == "fonte_bloqueada"
    picante = bom + "\nOferecemos uma variedade incrível de vídeos picantes e excitantes."
    assert gen.doc_reject_reason(picante, cfg, "https://site.com.br/a") == "conteudo_bloqueado"
    assert gen.doc_reject_reason(bom, gen.Config(deny_url_regex=""), "https://x.com/a") is None


def test_process_chunk_marca_modo(monkeypatch):
    import threading
    monkeypatch.setattr(gen, "ask_teacher",
                        lambda cfg, c, tipo, anc, seed: (qa_ok(tipo), {}, ""))
    cfg = gen.Config(ancorado_frac=1.0, recusa_frac=0.0)
    res = gen.process_chunk(_chunk(), cfg, set(), threading.Lock())
    assert res.records[0]["ancorado"] is True


def test_ask_teacher_le_resposta_do_servidor(monkeypatch):
    captured = {}

    def fake_post(url, body, timeout):
        captured.update(body)
        return {"choices": [{"message": {"content": json.dumps(qa_ok())}}],
                "usage": {"completion_tokens": 7, "prompt_tokens": 99}}

    monkeypatch.setattr(gen, "post_json", fake_post)
    qa, usage, err = gen.ask_teacher(gen.Config(), CHUNK, "extracao", True, 1)
    assert err == "" and qa["pergunta"] == PERGUNTA and usage["completion_tokens"] == 7
    # JSON garantido: schema completo com strict (json_object+schema é ignorado no llama.cpp)
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


# ------------------------------------------------------------------ ponta a ponta


def _parquet(tmp_path) -> Path:
    """Parquet sintético: 2 docs bons (um em `val`, que NÃO pode entrar) + 1 ruim."""
    paras = ["A companhia anunciou um plano de investimentos de 12 bilhões de reais até 2028, "
             "com foco em redes de distribuição no interior do estado."]
    paras += ["O programa prevê a modernização de subestações e a instalação de medidores "
              "inteligentes em várias cidades do interior, segundo o diretor de operações."] * 19
    bom = "\n".join(paras)
    table = pa.table({
        "id": ["d1", "d2", "d3"],
        "content": [bom, bom.replace("12 bilhões", "15 bilhões"), "curto"],
        "url": ["https://a.com.br/1", "https://b.com.br/2", "https://c.com.br/3"],
        "split": ["train", "val", "train"],
        "dump": ["2024-10"] * 3,
    })
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "parte.parquet"
    pq.write_table(table, path, row_group_size=3)
    return tmp_path


def _fake_teacher():
    """Professor falso: evidência tirada do próprio trecho e pergunta única por chamada."""
    n = [0]

    def fake(cfg, chunk, tipo, ancorado, seed):
        n[0] += 1
        return {"status": "ok", "tipo": tipo,
                "pergunta": f"Qual é a informação número {n[0]} desta notícia?",
                "resposta": "A companhia anunciou um programa de investimentos bilionários.",
                "evidencia": " ".join(chunk.split()[:12])}, {"completion_tokens": 30}, ""

    return fake


def test_main_ponta_a_ponta(tmp_path, monkeypatch):
    raw = _parquet(tmp_path / "raw")
    out = tmp_path / "sft" / "pilot.jsonl"
    monkeypatch.setattr(gen, "ask_teacher", _fake_teacher())
    gen.main(["--raw-dir", str(raw), "--out", str(out), "--max-examples", "3",
              "--chunk-chars", "900", "--chunk-min-chars", "300", "--recusa-frac", "0"])

    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(recs) == 3
    assert {r["doc_id"] for r in recs} == {"d1"}  # d2 é val, d3 é doc_curto
    for r in recs:
        assert r["evidencia"] in gen.normalize_ws(r["chunk_text"])
        assert r["split"] == "train"
    stats = json.loads(out.with_suffix(".stats.json").read_text())
    assert stats["accepted"] == 3 and stats["completion_tokens"] >= 90

    # --resume não reescreve o arquivo e não regenera o que já está lá
    calls = []

    def counting(cfg, c, tipo, anc, seed):
        calls.append(1)
        return _fake_teacher()(cfg, c, tipo, anc, seed)

    monkeypatch.setattr(gen, "ask_teacher", counting)
    before = out.read_text()
    gen.main(["--raw-dir", str(raw), "--out", str(out), "--max-examples", "3", "--resume",
              "--chunk-chars", "900", "--chunk-min-chars", "300"])
    assert out.read_text() == before
    assert calls == []  # nada foi regenerado


def test_main_recusa_existente_sem_resume(tmp_path):
    raw = _parquet(tmp_path / "raw")
    out = tmp_path / "x.jsonl"
    out.write_text("")
    try:
        gen.main(["--raw-dir", str(raw), "--out", str(out)])
    except SystemExit as exc:
        assert "já existe" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("deveria recusar sobrescrever o arquivo")
