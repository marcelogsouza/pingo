"""Testes da API /treino (treino-4) — CPU, sem rede, sem tocar nos runs reais.

A página é fina; o que não pode quebrar é o contrato:
- leitura (status/hw/datasets/qa/tokenizadores/tokens/runs) sempre JSON estruturado;
- disparo valida ANTES de gastar processo: nome de run, runs protegidos, out ocupado
  pedindo retomar, tokens/tokenizer/base inexistentes, fila de um-por-vez (429);
- o mutex chat↔job: com job rodando, Server.carregar recusa (a VRAM é uma só);
- um pré-treino tiny REAL de ponta a ponta pela API (subprocess CPU, CUDA off).

ROOT do webui é remapeado p/ tmp/ — nenhum teste escreve em runs/, data/ ou tokenizer/.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location("pingo_webui_treino", ROOT / "scripts" / "webui.py")
w = importlib.util.module_from_spec(_spec)
sys.modules["pingo_webui_treino"] = w
_spec.loader.exec_module(w)

from pingo.jobs import Job, JobRunner  # noqa: E402

DOC = "O governo anunciou nesta terça-feira um pacote de investimentos em infraestrutura. " * 10


@pytest.fixture()
def ambiente(tmp_path, monkeypatch):
    """webui com ROOT falso (tmp) e JobRunner com jobs.json em tmp — nada toca o repo."""
    raiz = tmp_path / "repo"
    for sub in ("runs", "data/tokens", "tokenizer", "data/datasets"):
        (raiz / sub).mkdir(parents=True)
    monkeypatch.setattr(w, "ROOT", raiz)

    runner = JobRunner(jobs_path=raiz / "runs" / "jobs.json")
    app = w.create_app(w.Server([]), runner)   # sem checkpoints: rotas de chat não são o alvo
    client = app.test_client()

    def tokens_dir(nome="toks", vocab=16_384, train_tokens=9200):
        d = raiz / "data/tokens" / nome
        d.mkdir(parents=True, exist_ok=True)
        for i, n in enumerate([5000, 3000, 1200]):
            (np.arange(n, dtype=np.uint16) % vocab).tofile(d / f"train-{i:04d}.bin")
        (np.arange(2000, dtype=np.uint16) % vocab).tofile(d / "val-0000.bin")
        (d / "meta.json").write_text(json.dumps({
            "train_shards": 3, "train_tokens": train_tokens, "val_shards": 1, "val_tokens": 2000,
            "vocab_size": vocab, "tokenizer": "tokenizer", "tokenizer_id": "tokenizer"}))
        return d

    def dataset_bom(nome="bom.jsonl"):
        p = raiz / "data/datasets" / nome
        p.write_text("".join(json.dumps({"content": DOC + f" Documento {i}."},
                                        ensure_ascii=False) + "\n" for i in range(10)),
                     encoding="utf-8")
        return p

    return {"raiz": raiz, "client": client, "runner": runner,
            "tokens_dir": tokens_dir, "dataset_bom": dataset_bom}


def esperar_fim(client, timeout=180.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get("/api/treino/status").get_json()
        if not s["rodando"]:
            return s["job"]
        time.sleep(0.5)
    pytest.fail("o job não terminou dentro do prazo")


# ------------------------------------------------------------------ página e leitura


def test_pagina_treino_existe_e_serve(ambiente):
    r = ambiente["client"].get("/treino")
    assert r.status_code == 200
    assert "treino" in r.get_data(as_text=True)
    assert r.mimetype == "text/html"


def test_status_fila_livre_e_hw_sao_json_sanos(ambiente):
    s = ambiente["client"].get("/api/treino/status").get_json()
    assert s["rodando"] is False and s["job"] is None
    h = ambiente["client"].get("/api/treino/hw").get_json()
    assert h["ram"]["usada_gb"] > 0 and "cpu_pct" in h and "disco" in h


def test_datasets_lista_jsonl_do_repo(ambiente):
    p = ambiente["dataset_bom"]()
    achados = ambiente["client"].get("/api/treino/datasets").get_json()["datasets"]
    assert any(d["caminho"] == "data/datasets/bom.jsonl" and d["bytes"] == p.stat().st_size
               for d in achados)


def test_qa_aprova_bom_reprova_ruim_e_recusa_fora_do_repo(ambiente):
    bom = ambiente["dataset_bom"]()
    r = ambiente["client"].get(f"/api/treino/qa?path={bom}").get_json()
    assert r["aprovado"] is True and r["linhas"] == 10

    ruim = ambiente["raiz"] / "data/datasets/ruim.jsonl"
    ruim.write_text('{"text": "' + DOC + '"}\n', encoding="utf-8")
    r2 = ambiente["client"].get(f"/api/treino/qa?path=data/datasets/ruim.jsonl").get_json()
    assert r2["aprovado"] is False and r2["detalhe"]["campo_candidato"] == "text"

    assert ambiente["client"].get("/api/treino/qa?path=/etc/passwd").status_code == 404
    (ambiente["raiz"] / "data" / "x.txt").write_text("oi")
    assert ambiente["client"].get("/api/treino/qa?path=data/x.txt").status_code == 400


def test_tokenizadores_listam_default_e_experimental(ambiente):
    base = ambiente["raiz"] / "tokenizer"
    (base / "tokenizer.json").write_text("{}")
    (base / "vocab_stats.json").write_text(json.dumps(
        {"vocab_size": 16384, "special_tokens": ["<|bos|>"], "our_chars_per_token": 3.3}))
    exp = base / "wiki-16k"
    exp.mkdir()
    (exp / "tokenizer.json").write_text("{}")
    (exp / "vocab_stats.json").write_text(json.dumps({"vocab_size": 16000}))
    itens = ambiente["client"].get("/api/treino/tokenizadores").get_json()["tokenizadores"]
    por_id = {i["id"]: i for i in itens}
    assert por_id["default"]["congelado"] is True and por_id["default"]["vocab"] == 16384
    assert por_id["wiki-16k"]["vocab"] == 16000


def test_tokens_listam_e_dir_unico(ambiente):
    ambiente["tokens_dir"]("wiki")
    lista = ambiente["client"].get("/api/treino/tokens").get_json()["tokens"]
    assert any(t["dir"] == "data/tokens/wiki" and t["train_tokens"] == 9200 for t in lista)
    um = ambiente["client"].get("/api/treino/tokens?dir=data/tokens/wiki").get_json()["tokens"]
    assert um["vocab_size"] == 16384
    assert ambiente["client"].get("/api/treino/tokens?dir=/etc").status_code == 400


# ------------------------------------------------------------------ validações de disparo


def test_runs_listam_e_curva_le_log(ambiente):
    run = ambiente["raiz"] / "runs" / "alguncoisa"
    run.mkdir()
    (run / "log.jsonl").write_text('{"step": 1, "train_loss": 9.9}\n{"step": 2, "train_loss": 9.5'
                                   '}\n{"step": 2, "val_loss": 9.1}\n')
    runs = ambiente["client"].get("/api/treino/runs").get_json()["runs"]
    assert runs[0]["nome"] == "alguncoisa" and runs[0]["steps"] == 2

    curva = ambiente["client"].get("/api/treino/runs/alguncoisa").get_json()["registros"]
    assert len(curva) == 3
    assert ambiente["client"].get("/api/treino/runs/../runs").status_code in (400, 404)
    assert ambiente["client"].get("/api/treino/runs/nao-existe").status_code == 404


def test_progresso_do_run_diz_quanto_falta(ambiente):
    """A barra do /treino precisa de três coisas: o tamanho do plano (max_steps), o custo
    do step e o andamento. O plano vem do `evento=inicio` que o train.py grava no log —
    por isso um run disparado fora da UI também sabe responder "quanto falta"."""
    raiz, c = ambiente["raiz"], ambiente["client"]
    tok = raiz / "data" / "tokens" / "tinytok"
    tok.mkdir(parents=True)
    (tok / "meta.json").write_text(json.dumps({"train_tokens": 1_000_000, "vocab_size": 64}))
    run = raiz / "runs" / "complano"
    run.mkdir()
    (run / "log.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"step": 0, "t": 0.0, "evento": "inicio", "max_steps": 100,
         "tokens_por_step": 10_000, "tokens_dir": str(tok), "train_tokens": 1_000_000},
        {"step": 10, "t": 100.0, "train_loss": 9.9, "toks_s": 9000, "tokens_seen": 100_000},
        {"step": 30, "t": 300.0, "train_loss": 8.5, "toks_s": 9500, "tokens_seen": 300_000},
    ]) + "\n")

    p = c.get("/api/treino/runs/complano").get_json()["progresso"]
    assert p["step"] == 30 and p["max_steps"] == 100
    assert abs(p["pct"] - 30.0) < 1e-9
    assert p["tokens_total"] == 1_000_000 and p["tokens_seen"] == 300_000
    # 100k tokens/step ... taxa medida no log: 20 steps em 200 s = 0,1 step/s → faltam 700 s
    assert abs(p["eta_s"] - 700.0) < 1.0
    assert abs(p["epocas"] - 1.0) < 1e-9            # 100 steps × 10k = o dataset inteiro
    assert p["dataset_tokens"] == 1_000_000
    # a lista de runs também traz o progresso (a coluna da tabela)
    lista = {r["nome"]: r for r in c.get("/api/treino/runs").get_json()["runs"]}
    assert lista["complano"]["progresso"]["pct"] == 30.0

    # run sem plano (log antigo, só loss) não inventa % nem ETA
    velho = raiz / "runs" / "veloplano"
    velho.mkdir()
    (velho / "log.jsonl").write_text('{"step": 5, "t": 10.0, "train_loss": 7.0}\n')
    p2 = c.get("/api/treino/runs/veloplano").get_json()["progresso"]
    assert p2["step"] == 5 and p2["pct"] is None and p2["eta_s"] is None


def test_progresso_de_run_retomado_conta_a_sessao(ambiente):
    """`--resume` num plano absoluto (5.248 → 6.488): o % do plano diria 81%, mas quem quer
    saber quanto falta precisa da SESSÃO. Tokens/épocas também são da sessão — somar o que o
    checkpoint já viu misturaria corpora diferentes."""
    raiz, c = ambiente["raiz"], ambiente["client"]
    tok = raiz / "data" / "tokens" / "tinytok2"
    tok.mkdir(parents=True)
    (tok / "meta.json").write_text(json.dumps({"train_tokens": 650_000_000, "vocab_size": 64}))
    run = raiz / "runs" / "retomado"
    run.mkdir()
    (run / "log.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"step": 5248, "t": 0.0, "evento": "inicio", "max_steps": 6488,
         "tokens_por_step": 524_288, "tokens_dir": str(tok), "train_tokens": 650_000_000},
        {"step": 5260, "t": 60.0, "train_loss": 3.0, "toks_s": 100_000,
         "tokens_seen": 2_757_754_880},
        {"step": 5280, "t": 160.0, "train_loss": 2.9, "toks_s": 100_000,
         "tokens_seen": 2_768_240_640},
    ]) + "\n")

    p = c.get("/api/treino/runs/retomado").get_json()["progresso"]
    assert p["step_inicio"] == 5248 and p["max_steps"] == 6488
    assert abs(p["pct"] - 81.38) < 0.01                 # do plano absoluto
    assert abs(p["pct_sessao"] - 100 * 32 / 1240) < 1e-6    # desta sessão: 2,6%
    assert p["tokens_seen"] == 32 * 524_288             # só o que a sessão andou
    assert p["tokens_total"] == 1240 * 524_288          # o que falta a sessão inteira
    assert abs(p["epocas"] - 1.0) < 1e-3                # ~1 passada no corpus, não 5,2
    # 20 steps em 100 s = 0,2 step/s → 1.208 restantes = 6.040 s
    assert abs(p["eta_s"] - 6040.0) < 1.0


def test_pretreino_recusa_nome_ruim_run_protegido_e_out_ocupado(ambiente):
    c, tokens = ambiente["client"], ambiente["tokens_dir"]()
    payloads = [
        ({"out": "../fuga"}, "inválido"),
        ({"out": "pt-120m"}, "protegido"),
        ({"out": "pt-120m-BACKUP-2026-10-30"}, "protegido"),
        ({"out": "novo", "tokens_dir": "data/tokens/nao-existe"}, "meta.json"),
    ]
    for corpo, esperado in payloads:
        r = c.post("/api/treino/pretreino", json=corpo)
        assert r.status_code in (400, 403) and esperado in r.get_json()["erro"], corpo

    ocupado = ambiente["raiz"] / "runs" / "ocupado"
    ocupado.mkdir()
    (ocupado / "ckpt.pt").write_bytes(b"x")
    r = c.post("/api/treino/pretreino", json={"out": "ocupado", "tokens_dir": str(tokens)})
    assert r.status_code == 409 and "retomar" in r.get_json()["erro"]
    # retomar=true mas sem ckpt de verdade também é barrado
    vazio = ambiente["raiz"] / "runs" / "semtree"
    vazio.mkdir()
    (vazio / "algo").write_text("x")
    r = c.post("/api/treino/pretreino", json={"out": "semtree", "tokens_dir": str(tokens),
                                              "retomar": True})
    assert r.status_code == 400 and "ckpt.pt" in r.get_json()["erro"]


def test_sft_validida_base_tokenizer_e_data_dir(ambiente):
    c = ambiente["client"]
    base = ambiente["raiz"] / "runs" / "basefake" / "final"
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}")
    tok = ambiente["raiz"] / "tokenizer"
    (tok / "tokenizer.json").write_text("{}")
    dd = ambiente["raiz"] / "data" / "sft"
    dd.mkdir(parents=True)
    (dd / "train.jsonl").write_text("")

    r = c.post("/api/treino/sft", json={"out": "x1"})                       # base default não existe
    assert r.status_code == 400 and "base" in r.get_json()["erro"]
    r = c.post("/api/treino/sft", json={"out": "x2", "base": "data"})       # base sem config.json
    assert r.status_code == 400 and "base" in r.get_json()["erro"]
    r = c.post("/api/treino/sft", json={"out": "x3", "base": "runs/basefake/final",
                                        "tokenizer": "tokenizer", "data_dir": "data"})
    assert r.status_code == 400 and "data_dir" in r.get_json()["erro"]
    r = c.post("/api/treino/sft", json={"out": "pt-120m", "base": "runs/basefake/final",
                                        "tokenizer": "tokenizer", "data_dir": "data/sft"})
    assert r.status_code == 403


def test_tokenizador_tokenizacao_validam_sem_gastar_processo(ambiente):
    c = ambiente["client"]
    r = c.post("/api/treino/tokenizador", json={"id": "../mal"})
    assert r.status_code == 400
    r = c.post("/api/treino/tokenizador", json={"id": "ok", "jsonl": "data/nao.jsonl"})
    assert r.status_code == 400
    exp = ambiente["raiz"] / "tokenizer" / "velho"
    exp.mkdir()
    (exp / "tokenizer.json").write_text("{}")
    r = c.post("/api/treino/tokenizador", json={"id": "velho"})
    assert r.status_code == 409 and "já existe" in r.get_json()["erro"]

    r = c.post("/api/treino/tokenizacao", json={"out": "x", "tokenizer": "nao-existe"})
    assert r.status_code == 400
    r = c.post("/api/treino/tokenizacao", json={"out": "x", "tokenizer": "velho",
                                                "input": "jsonl", "jsonl": "data/nao.jsonl"})
    assert r.status_code == 400


def test_fila_ocupada_devolve_429(ambiente):
    # simula job rodando sem subprocess: a fila decide, não o filho
    ambiente["runner"].current = Job(id="z", tipo="pretreino", titulo="fantasma", argv=[])
    r = ambiente["client"].post("/api/treino/pretreino",
                                json={"out": "qualquer", "tokens_dir": "data/tokens"})
    assert r.status_code == 429 and "fantasma" in r.get_json()["erro"]
    ambiente["runner"].current = None


def test_job_rodando_bloqueia_carga_de_modelo_no_chat(ambiente):
    ambiente["runner"].current = Job(id="z", tipo="sft", titulo="sft x", argv=[])
    srv = ambiente["client"].application.config["PINGO"]
    with pytest.raises(RuntimeError, match="treino"):
        srv.carregar("qualquer")
    ambiente["runner"].current = None


# ------------------------------------------------------------------ ponta a ponta (real, CPU)


def test_pretreino_tiny_de_verdade_pela_api(ambiente):
    """A API dispara o scripts/train.py de verdade (subprocess CPU, sem CUDA) e a gente
    assiste pelo status/log/runs — é exatamente o fluxo do botão da página."""
    tokens = ambiente["tokens_dir"]()
    corpo = {"out": "api-tiny", "tokens_dir": str(tokens), "device": "cpu", "tiny": True,
             "max_steps": 3, "ctx": 32, "batch_size": 4, "accum_steps": 2, "warmup_steps": 1,
             "eval_steps": 2, "epochs": 0}
    r = ambiente["client"].post("/api/treino/pretreino", json=corpo)
    assert r.status_code == 202, r.get_json()
    job = esperar_fim(ambiente["client"])
    assert job["state"] == "concluido" and job["rc"] == 0, job

    out = ambiente["raiz"] / "runs" / "api-tiny"
    assert (out / "final" / "model.safetensors").exists()
    log = ambiente["client"].get("/api/treino/log?depois=0").get_json()
    assert any("params" in l for l in log["linhas"]) and log["cursor"] == len(log["linhas"])
    runs = ambiente["client"].get("/api/treino/runs").get_json()["runs"]
    assert runs[0]["nome"] == "api-tiny" and runs[0]["final"] is True
    curva = ambiente["client"].get("/api/treino/runs/api-tiny").get_json()["registros"]
    assert any(rc.get("val_loss") is not None for rc in curva)
    # o argv gravado no registro é o que a CLI veria
    assert "--tokens-dir" in job["argv"] and "--max-steps" in job["argv"]


def test_epocas_viram_max_steps_pelo_meta(ambiente):
    tokens = ambiente["tokens_dir"](nome="epoch", train_tokens=9200)
    corpo = {"out": "epoch-teste", "tokens_dir": str(tokens), "device": "cpu", "tiny": True,
             "epochs": 2, "ctx": 32, "batch_size": 4, "accum_steps": 2}
    r = ambiente["client"].post("/api/treino/pretreino", json=corpo)
    assert r.status_code == 202, r.get_json()
    argv = r.get_json()["job"]["argv"]
    # 2 épocas × 9200 / (4·32·2=256) = 72 steps
    assert argv[argv.index("--max-steps") + 1] == "72"
    ambiente["client"].post("/api/treino/parar", json={})   # pode estar rodando: para já
    esperar_fim(ambiente["client"])
