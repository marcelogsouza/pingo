"""JobRunner (treino-3) — a interface /treino NÃO reimplementa o treino: ela supervisiona.

Cada job é um subprocess do script de sempre (`train.py`, `train_sft.py`,
`prepare_tokens.py`, `train_tokenizer.py`) — mesma esteira da CLI, mesma regra de
**um escritor por saída**: a fila daqui é o único jeito de um job da UI começar, e só
um pode rodar por vez (a VRAM é uma só e o AGENTS.md manda). O "parar" manda SIGTERM
— desde o treino-1 os scripts checkpointam e saem limpos (rc 130).

Estado:
- `rodando` → `parando` (SIGTERM pedido) → `interrompido` | `concluido` | `falhou`;
- saída do filho em um anel de memória (o log bruto da tela) + `log.jsonl` do `--out`
  parseado para a curva de loss;
- registro persistente em `runs/jobs.json` (sobrevive ao restart da webui). Se o
  servidor morreu com um job vivo, no boot o runner **caça o órfão** (SIGTERM) — nada
  de treino fantasma comendo a GPU depois de um Ctrl+C no servidor.
"""

from __future__ import annotations

import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ESTADOS_FIM = ("interrompido", "concluido", "falhou")


class JobOcupado(RuntimeError):
    """Já existe um job rodando — a fila é de um só (escritor único + VRAM única)."""


@dataclass
class Job:
    id: str
    tipo: str                       # pretreino | sft | tokenizacao | tokenizador | qa
    titulo: str
    argv: list[str]
    out: str | None = None          # diretório do run (p/ ler log.jsonl/ckpt)
    precisa_gpu: bool = True
    state: str = "rodando"          # rodando | parando | interrompido | concluido | falhou
    started: float = field(default_factory=time.time)
    ended: float | None = None
    rc: int | None = None
    pid: int | None = None

    def resumo(self) -> dict:
        return {"id": self.id, "tipo": self.tipo, "titulo": self.titulo,
                "argv": self.argv, "out": self.out, "precisa_gpu": self.precisa_gpu,
                "state": self.state, "started": self.started, "ended": self.ended,
                "rc": self.rc, "pid": self.pid}


class JobRunner:
    def __init__(self, jobs_path: Path | None = None, ring: int = 400):
        self.jobs_path = Path(jobs_path) if jobs_path else ROOT / "runs" / "jobs.json"
        self._lock = threading.Lock()
        self.current: Job | None = None
        self._proc: subprocess.Popen | None = None
        self._saida: collections.deque[str] = collections.deque(maxlen=ring)
        self.historico: list[dict] = []
        self._caçar_órfãos()
        self._carregar()

    # --------------------------------------------------------------- boot
    def _carregar(self) -> None:
        if self.jobs_path.exists():
            try:
                self.historico = json.loads(self.jobs_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.historico = []

    def _caçar_órfãos(self) -> None:
        """Job marcado `rodando` de uma webui que morreu: SIGTERM e encerra o registro.

        Os scripts treinam em subprocess proprio — sobrevivem ao servidor. Sem isto,
        um restart da UI deixaria um treino fantasma segurando a GPU para sempre.
        """
        if not self.jobs_path.exists():
            return
        try:
            registros = json.loads(self.jobs_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        mexido = False
        for rec in registros:
            if rec.get("state") != "rodando":
                continue
            pid = rec.get("pid")
            if pid and os.path.exists(f"/proc/{pid}"):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            rec.update(state="interrompido", ended=time.time(),
                       motivo="webui reiniciou — órfão encerrado")
            mexido = True
        if mexido:
            self.jobs_path.write_text(json.dumps(registros, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    def _persistir(self) -> None:
        self.jobs_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.jobs_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.historico[-200:], ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.jobs_path)

    # --------------------------------------------------------------- ciclo
    def rodando(self) -> bool:
        return bool(self.current and self.current.state in ("rodando", "parando"))

    def iniciar(self, tipo: str, titulo: str, argv: list[str], out: str | None = None,
                precisa_gpu: bool = True, env: dict[str, str] | None = None) -> Job:
        """Dispara o subprocess; `argv` começa com o script (o runner põe `sys.executable`).

        `JobOcupado` se a fila está ocupada — um escritor por saída, uma VRAM por vez.
        """
        with self._lock:
            if self.rodando():
                raise JobOcupado(f"job {self.current.titulo} está {self.current.state}")
            job = Job(id=uuid.uuid4().hex[:8], tipo=tipo, titulo=titulo, argv=list(argv),
                      out=out, precisa_gpu=precisa_gpu)
            meio = dict(os.environ)
            if env:
                meio.update(env)
            # stdout do filho em linha por linha; stderr junto (o print do treino é o log)
            proc = subprocess.Popen([sys.executable, *argv],
                                    cwd=str(ROOT), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, env=meio)
            job.pid = proc.pid
            self._proc = proc
            self.current = job
            self._saida.clear()
            self.historico.append(job.resumo())
            self._persistir()
            threading.Thread(target=self._vigiar, args=(job, proc), daemon=True).start()
            return job

    def _vigiar(self, job: Job, proc) -> None:
        for linha in proc.stdout:
            self._saida.append(linha.rstrip("\n"))
        rc = proc.wait()
        with self._lock:
            job.rc = rc
            job.ended = time.time()
            if job.state == "parando" or rc == 130:
                job.state = "interrompido"
            else:
                job.state = "concluido" if rc == 0 else "falhou"
            for rec in reversed(self.historico):
                if rec.get("id") == job.id:
                    rec.update(job.resumo())
                    break
            self._persistir()

    def parar(self) -> bool:
        """SIGTERM no filho — desde o treino-1 ele checkpointa e sai limpo."""
        with self._lock:
            if not (self.current and self.current.state == "rodando" and self._proc):
                return False
            self.current.state = "parando"
            self._proc.terminate()
            return True

    # --------------------------------------------------------------- leitura
    def saida(self, depois: int = 0) -> list[str]:
        """As linhas do anel desde o índice `depois` (o cliente SSE manda o cursor)."""
        linhas = list(self._saida)
        return linhas[depois:]

    def log_records(self, limite: int = 1000) -> list[dict]:
        """Registros do `log.jsonl` do job atual (a curva de loss) — tolerante a linha no meio."""
        out = self.current.out if self.current else None
        if not out:
            return []
        arq = Path(out) / "log.jsonl"
        if not arq.exists():
            return []
        registros = []
        with arq.open(encoding="utf-8", errors="replace") as fh:
            for linha in fh:
                try:
                    registros.append(json.loads(linha))
                except json.JSONDecodeError:
                    continue          # linha escrita pela metade entre flushes
        return registros[-limite:]

    def estado(self) -> dict:
        """Tudo que a tela do /treino precisa numa chamada (job atual + saída + curva)."""
        job = self.current
        return {"rodando": self.rodando(),
                "job": job.resumo() if job else None,
                "historico": self.historico[-20:][::-1]}


# ------------------------------------------------------------------ hardware

_GPU_TTL = 2.0
_gpu_cache: tuple[float, object] = (0.0, None)


def hw() -> dict:
    """Uma amostra de CPU/RAM/disco/GPU (o painel do /treino). Barato e sem estado no filho.

    Com o Strata/professor servindo, o `nvidia-smi` mostra a verdade que o AGENTS.md
    pede para não ignorar: se a VRAM estiver cheia, job pesado não deveria começar.
    """
    global _gpu_cache
    import psutil

    agora = time.time()
    if agora - _gpu_cache[0] > _GPU_TTL:
        gpu = None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,"
                                     "utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4)
            if out.returncode == 0:
                gpu = []
                for linha in out.stdout.strip().splitlines():
                    nome, usada, total, util, temp = [c.strip() for c in linha.split(",")]
                    gpu.append({"nome": nome, "usada_mb": int(usada), "total_mb": int(total),
                                "usado_pct": int(util), "temp_c": int(temp)})
        except (OSError, subprocess.TimeoutExpired, ValueError):
            gpu = None
        _gpu_cache = (agora, gpu)
    vm = psutil.virtual_memory()
    disco = psutil.disk_usage(str(ROOT))
    return {
        "t": round(agora, 1),
        "cpu_pct": psutil.cpu_percent(interval=0.05),
        "ram": {"usada_gb": round(vm.used / 2**30, 2), "total_gb": round(vm.total / 2**30, 2),
                "pct": vm.percent},
        "disco": {"livre_gb": round(disco.free / 2**30, 1), "total_gb": round(disco.total / 2**30, 1),
                  "pct": disco.percent},
        "gpu": _gpu_cache[1],
    }
