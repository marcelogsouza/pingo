"""Testes do JobRunner (treino-3) — subprocesses minúsculos em CPU, sem tocar em runs/ real.

O runner é o único jeito de a UI começar um job, então o teste cobre o contrato inteiro:
fila de um-por-vez, SIGTERM no "parar", classificação do fim (concluido/falhou/
interrompido), persistência em jobs.json e a **caça ao órfão** no boot (a webui morreu
com um filho vivo — o treino fantasma não pode sobreviver ao restart).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pingo.jobs import Job, JobOcupado, JobRunner, hw


def esperar_fim(runner: JobRunner, timeout: float = 15.0) -> dict:
    t0 = time.time()
    while runner.rodando() and time.time() - t0 < timeout:
        time.sleep(0.05)
    assert not runner.rodando(), "o job não terminou dentro do prazo"
    return runner.current


def test_job_bom_conclui_guarda_saida_e_persiste(tmp_path):
    script = tmp_path / "oi.py"
    script.write_text("print('linha um')\nprint('linha dois')\n")
    jp = tmp_path / "runs" / "jobs.json"
    runner = JobRunner(jobs_path=jp)

    runner.iniciar("qa", "diz oi", [str(script)])
    job = esperar_fim(runner)
    assert job.state == "concluido" and job.rc == 0
    assert runner.saida() == ["linha um", "linha dois"]

    registro = json.loads(jp.read_text())
    assert registro[-1]["state"] == "concluido" and registro[-1]["pid"] > 0
    assert runner.estado()["job"]["id"] == job.id
    assert runner.estado()["historico"][0]["id"] == job.id     # mais recente primeiro


def test_fila_e_de_um_por_vez_e_libera_no_fim(tmp_path):
    script = tmp_path / "dorme.py"
    script.write_text("import time\ntime.sleep(5)\n")
    runner = JobRunner(jobs_path=tmp_path / "jobs.json")

    runner.iniciar("pretreino", "primeiro", [str(script)])
    with pytest.raises(JobOcupado):
        runner.iniciar("pretreino", "segundo", [str(script)])
    assert runner.current.titulo == "primeiro"

    assert runner.parar()
    esperar_fim(runner)
    runner.iniciar("pretreino", "segundo", [str(script)])      # a fila abriu de novo
    assert runner.current.titulo == "segundo"
    runner.parar()
    esperar_fim(runner)


def test_parar_manda_sigterm_e_o_estado_fica_interrompido(tmp_path):
    script = tmp_path / "loop.py"
    script.write_text("import time\nwhile True:\n    print('tick', flush=True)\n    time.sleep(0.05)\n")
    runner = JobRunner(jobs_path=tmp_path / "jobs.json")
    runner.iniciar("sft", "loop", [str(script)])
    t0 = time.time()
    while len(runner.saida()) < 3 and time.time() - t0 < 10:
        time.sleep(0.05)
    assert runner.parar() is True
    assert runner.current.state == "parando"                   # pedido registrado na hora
    job = esperar_fim(runner)
    assert job.state == "interrompido" and job.rc in (-15, 130)
    assert runner.parar() is False                             # sem job rodando, nada a parar


def test_job_que_falha_fica_falhou_com_rc(tmp_path):
    script = tmp_path / "boom.py"
    script.write_text("import sys\nsys.exit(3)\n")
    runner = JobRunner(jobs_path=tmp_path / "jobs.json")
    runner.iniciar("tokenizacao", "explode", [str(script)])
    job = esperar_fim(runner)
    assert job.state == "falhou" and job.rc == 3


def test_boot_caça_o_órfão_e_reescreve_o_registro(tmp_path):
    # o órfão de verdade é um `scripts/train.py` nosso: é isso que a caça reconhece
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    alvo = scripts / "train.py"
    alvo.write_text("import time; time.sleep(60)\n")
    órfão = subprocess.Popen([sys.executable, str(alvo)])
    jp = tmp_path / "jobs.json"
    jp.write_text(json.dumps([{"id": "antigo", "tipo": "pretreino", "titulo": "webui morreu",
                               "argv": [], "state": "rodando", "pid": órfão.pid,
                               "started": time.time(), "ended": None, "rc": None}]))

    runner = JobRunner(jobs_path=jp, caçar=True)   # o processo da webui caça; um teste, não
    assert órfão.wait(timeout=10) in (-15, 130)               # o SIGTERM chegou
    rec = json.loads(jp.read_text())[0]
    assert rec["state"] == "interrompido" and "órfão" in rec["motivo"]
    assert runner.current is None and not runner.rodando()


def test_boot_não_mata_processo_alheio(tmp_path):
    """O registro pode apontar para um pid reciclado — ou, pior, o `jobs.json` real pode ter
    um treino VIVO e um teste chamar `create_app()` sem `jobs=` (foi o que matou o run da
    wiki no step 74 em 2026-11-03). A caça só SIGTERMa script de treino DESTE repo.
    """
    alheio = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    jp = tmp_path / "jobs.json"
    jp.write_text(json.dumps([{"id": "x", "tipo": "pretreino", "titulo": "outro processo",
                               "argv": [], "state": "rodando", "pid": alheio.pid,
                               "started": time.time(), "ended": None, "rc": None}]))
    try:
        runner = JobRunner(jobs_path=jp, caçar=True)
        time.sleep(0.5)
        assert alheio.poll() is None                          # continua vivo
        rec = json.loads(jp.read_text())[0]
        assert rec["state"] == "interrompido" and "não é do projeto" in rec["motivo"]
        assert not runner.rodando()
    finally:
        alheio.kill()
        alheio.wait(timeout=10)


def test_saida_do_filho_chega_antes_do_fim(tmp_path):
    """stdout do filho em pipe é block-buffered por padrão (despeja a cada ~8 KB): o painel
    de log da UI passaria minutos vazio no meio de um treino e despejaria tudo no fim.
    `bufsize=1` no Popen só vale p/ o lado do pai — o filho precisa de PYTHONUNBUFFERED=1.
    """
    script = tmp_path / "demorado.py"
    script.write_text(
        "import time\n"
        "print('primeira linha')      # sem flush: no pipe cru só sairia ao terminar\n"
        "time.sleep(6)\n"
    )
    runner = JobRunner(jobs_path=tmp_path / "jobs.json")
    runner.iniciar("qa", "fala cedo", [str(script)])
    t0 = time.time()
    while time.time() - t0 < 3.0 and "primeira linha" not in runner.saida():
        time.sleep(0.05)
    assert "primeira linha" in runner.saida(), "a linha só chegou no fim — stdout bufferizado"
    runner.parar()
    esperar_fim(runner)


def test_log_records_tolera_linha_pela_metade(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "log.jsonl").write_text(
        '{"step": 1, "train_loss": 9.9}\n{"step": 2, "val'      # cortada no flush
    )
    runner = JobRunner(jobs_path=tmp_path / "jobs.json")
    runner.current = Job(id="x", tipo="pretreino", titulo="t", argv=[], out=str(out))
    assert runner.log_records() == [{"step": 1, "train_loss": 9.9}]


def test_hw_devolve_números_sanos(tmp_path):
    h = hw()
    assert 0.0 <= h["cpu_pct"] <= 100.0 * (1 + 1e-6)
    assert h["ram"]["usada_gb"] > 0 and h["ram"]["pct"] > 0
    assert h["disco"]["livre_gb"] >= 0
    assert h["gpu"] is None or (isinstance(h["gpu"], list) and
                                all({"nome", "usada_mb", "total_mb"} <= g.keys()
                                    for g in h["gpu"]))
