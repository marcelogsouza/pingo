#!/usr/bin/env python
"""WebUI local (Flask) — a ferramenta de teste do Pingo treinado.

Serve uma página de chat e uma API de streaming (SSE) por cima do `pingo.generate`:

- **Dois modelos na mesma UI**: o SFT (`runs/sft-120m-ep1/final`, instrução) e o base
  (`runs/pt-120m/final`, que só continua texto). Trocar de modelo descarrega o anterior —
  a VRAM é compartilhada com o treino.
- **Streaming token a token**: `generate_stream()` emite cada id; aqui ele é decodificado
  em incrementos (segurando byte multibyte incompleto) e mandado como evento `delta`.
- **Prompt igual ao do treino**: `pingo.sft.render` em espírito — `<|user|>Contexto: …\\n\\n
  Pergunta: …<|assistant|>`, parando no `<|end|>`. Sem contexto, só `Pergunta: …`.
- **Uma geração por vez**: um `threading.Lock` não bloqueante devolve 429 se você mandar
  outra pergunta enquanto o modelo está no meio de uma resposta.
- **Modo didático em `/lab`** (o curso, passo a passo): o prompt vira tokens coloridos com
  os ids, a superfície byte-level e o vetor inicial de cada id (passo ①), e a frase vira uma
  **matriz desenhada** com as estatísticas da tabela inteira e a "lida de volta" tied (passo
  ②) — lógica em `pingo.didatic`, que lê a tabela de embeddings do `.safetensors` na CPU.
  Os passos ①–② não tocam na GPU nem no lock de geração: dá para abrir o laboratório
  enquanto uma resposta é gerada no chat. O passo ③ (RoPE + atenção com Q/K **reais**)
  carrega o modelo de propósito no device do servidor, respeita o lock de geração (429 se o
  chat está falando) e limita a frase a 64 tokens. O passo ④ (RMSNorm + SwiGLU) pega
  carona no MESMO forward: a captura do ③ já entrega o bloco inteiro — normas antes/depois,
  γ e as 2.304 portas do MLP do token escolhido. O passo ⑤ (lm head) pega a mesma carona:
  a norm final e os **logits** daquele forward (o `lm_head` é a própria tabela do passo ①
  — tied — lida de volta com o vetor final de cada posição).

Uso:
  uv run python scripts/webui.py                      # 0.0.0.0:8010 (LAN + localhost)
  uv run python scripts/webui.py --host 127.0.0.1     # só esta máquina
  uv run python scripts/webui.py --device cpu --port 8080
  uv run python scripts/webui.py --sft runs/sft-120m-ep1/final --base runs/pt-120m/final

A porta padrão é **8010** (a 8000 desta máquina vive ocupada) e o host padrão é `0.0.0.0`:
o chat responde na rede local (ex.: `http://192.168.0.101:8010`). **Não há autenticação** —
qualquer um na mesma rede pode usar o modelo; `--host 127.0.0.1` fecha de novo. O servidor é
o de desenvolvimento do Flask: serve para LAN doméstica, não para a internet.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from flask import Flask, Response, jsonify, request, send_file
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pingo.didatic import (LIDA_PADRAO, MAX_ATTN_TOKENS, MAX_LAB_CHARS,  # noqa: E402
                           MAX_LAB_TOKENS, TabelaEmbeddings, TOPK_LMHEAD, VIZINHOS_PADRAO,
                           buscar, ficha_bloco, ficha_lmhead, tokenizar)
from pingo.generate import generate_stream  # noqa: E402
from pingo.model import PingoLM  # noqa: E402
from pingo.sft import ASSISTANT_TOKEN, END_TOKEN, USER_TOKEN, chat_token_ids  # noqa: E402

MAX_PROMPT_CHARS = 4000
MAX_CONTEXT_CHARS = 6000
PAGINA_LAB = ROOT / "web" / "lab.html"


# ---------------------------------------------------------------- modelos


@dataclass(frozen=True)
class ModelSpec:
    """Um diretório de checkpoint que a UI pode servir."""

    id: str
    label: str
    path: Path
    instrucao: bool


@dataclass
class Loaded:
    spec: ModelSpec
    model: PingoLM
    tokenizer: Tokenizer
    end_id: int | None


class Server:
    """Carrega um modelo por vez, sob demanda (lazy) e com troca segura."""

    def __init__(self, specs: list[ModelSpec], device: str = "auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" \
            else device
        self.specs = {s.id: s for s in specs if s.path.exists()}
        self._loaded: Loaded | None = None
        self._tokenizadores: dict[str, Tokenizer] = {}
        self._tabelas: dict[str, TabelaEmbeddings] = {}
        self._captura: tuple[tuple, dict] | None = None   # última captura de atenção (③)
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()

    # -- tokenizer e tabela de embeddings (o que o laboratório usa: CPU) --
    def arquivo_tokenizador(self, spec: ModelSpec) -> Path:
        """O tokenizer do checkpoint; o base não traz o seu — cai no do projeto."""
        proprio = spec.path / "tokenizer.json"
        return proprio if proprio.exists() else ROOT / "tokenizer" / "tokenizer.json"

    def _le_tokenizador(self, model_id: str) -> Tokenizer:
        """Sem lock: quem chama já está dentro de `self._load_lock`."""
        tok = self._tokenizadores.get(model_id)
        if tok is None:
            tok = Tokenizer.from_file(str(self.arquivo_tokenizador(self.specs[model_id])))
            self._tokenizadores[model_id] = tok
        return tok

    def tokenizador_de(self, model_id: str) -> Tokenizer:
        """Só o tokenizer — o laboratório tokeniza sem pôr modelo nenhum na GPU."""
        with self._load_lock:
            if self._loaded and self._loaded.spec.id == model_id:
                return self._loaded.tokenizer
            return self._le_tokenizador(model_id)

    def tabela_de(self, model_id: str) -> TabelaEmbeddings:
        """A `embed.weight` do checkpoint na CPU (~58 MB) com normas e cossenos, cacheada.

        Lê direto do `model.safetensors`, sem construir o modelo: por isso o modo didático
        funciona com o professor servindo na VRAM. A checagem de vocab é obrigatória — o
        base tem 16.384 tokens e o SFT 16.387 (os 3 de chat são `added_tokens`).
        """
        with self._load_lock:
            tabela = self._tabelas.get(model_id)
            if tabela is None:
                tabela = TabelaEmbeddings.do_checkpoint(self.specs[model_id].path)
                tokenizer = self._le_tokenizador(model_id)
                if tabela.pesos.shape[0] != tokenizer.get_vocab_size():
                    raise ValueError(
                        f"{model_id}: a tabela de embeddings tem {tabela.pesos.shape[0]} linhas "
                        f"e o tokenizer tem {tokenizer.get_vocab_size()} tokens")
                tabela.vocabulario(tokenizer)
                self._tabelas[model_id] = tabela
            return tabela

    # -- captura de atenção/bloco/lm head (③–⑤): a ÚNICA rota do lab que carrega o modelo --
    def captura_atencao(self, model_id: str, ids: list[int]) -> dict:
        """Um forward com `capture`: Q/K pré/pós-RoPE, pesos de atenção, o bloco inteiro,
        a norm final e os **logits** daquele forward.

        Diferente do resto do lab (que vive só do `.safetensors` na CPU), ③, ④ e ⑤ mostram
        números **reais** — então carregamos o modelo na GPU (com o aviso do professor
        valendo) e obedecemos ao `gen_lock` (a rota devolve 429 se o chat está gerando).
        Guarda só a última captura (chave = modelo+ids) — trocou o texto, refaz o forward;
        o ④ e o ⑤ vieram de carona no mesmo forward do ③ (por isso os três falam com o
        mesmo cached). Os logits do ⑤ são os que o **próprio forward produziu**
        (`out.logits`), não uma recomputação — é a nota real que o lm head tied deu a cada
        token do vocab.
        """
        chave = (model_id, tuple(ids))
        if self._captura is not None and self._captura[0] == chave:
            return self._captura[1]
        carregado = self.carregar(model_id)          # o lock de carga é do próprio carregar
        modelo = carregado.model
        with torch.no_grad():
            cap: list[dict] = [{} for _ in range(modelo.config.num_hidden_layers)]
            out = modelo(torch.tensor([ids], device=next(modelo.parameters()).device),
                         capture=cap)
        chaves = ("w", "q_lin", "q", "k_lin", "k",                               # ③
                  "x0", "n1p", "n1", "a", "x1", "n2p", "n2", "g", "u", "s",      # ④
                  "m", "x2")
        saida = {
            "heads": modelo.config.num_attention_heads,
            "head_dim": modelo.config.head_dim,
            "theta": float(modelo.config.rope_theta),
            "logits": out.logits[0].float().cpu(),                        # ⑤ [T, vocab]
            "fin": {k: cap[-1][k][0].float().cpu() for k in ("nfinp", "nfin")},  # ⑤ norm final
            "g_fin": modelo.norm.weight.detach().float().cpu(),           # ⑤ o γ da norm final
            "camadas": [{**{k: c[k][0].float().cpu() for k in chaves},          # sem a dimensão B
                         "g_attn": blk.norm_attn.weight.detach().float().cpu(),  # o γ de cada norma
                         "g_mlp": blk.norm_mlp.weight.detach().float().cpu()}
                        for c, blk in zip(cap, modelo.blocks)],
        }
        self._captura = (chave, saida)
        return saida

    # -- ciclo de vida -------------------------------------------------
    def carregar(self, model_id: str) -> Loaded:
        with self._load_lock:
            if self._loaded and self._loaded.spec.id == model_id:
                return self._loaded
            self.descarregar()
            spec = self.specs[model_id]
            tokenizer = self._le_tokenizador(model_id)
            model = PingoLM.from_pretrained(spec.path, torch_dtype=torch.float32)
            model = model.to(self.device).eval()
            # só o SFT sabe parar no <|end|>; no base ele não significa nada
            end_id = chat_token_ids(tokenizer).get(END_TOKEN) if spec.instrucao else None
            self._loaded = Loaded(spec, model, tokenizer, end_id)
            return self._loaded

    def descarregar(self) -> None:
        if self._loaded is not None:
            self._loaded = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    # -- estado --------------------------------------------------------
    def info(self) -> dict:
        carregado = self._loaded.spec.id if self._loaded else None
        return {
            "device": self.device,
            "carregado": carregado,
            "vram_gb": (round(torch.cuda.memory_allocated() / 2**30, 2)
                        if self.device == "cuda" else None),
            "modelos": [{"id": s.id, "label": s.label, "instrucao": s.instrucao,
                         "carregado": s.id == carregado,
                         "caminho": str(s.path.relative_to(ROOT)) if s.path.is_relative_to(ROOT)
                         else str(s.path)}
                        for s in sorted(self.specs.values(), key=lambda s: not s.instrucao)],
        }

    def ocupado(self) -> bool:
        """True se já há uma geração em andamento (o lock é por requisição)."""
        return self._gen_lock.locked()

    @property
    def gen_lock(self) -> threading.Lock:
        return self._gen_lock


# ---------------------------------------------------------------- prompt/decodificação


def montar_prompt(texto: str, contexto: str = "", instrucao: bool = True) -> str:
    """Prompt de chat (SFT) ou continuação crua (base) — igual ao formato do treino."""
    texto, contexto = (texto or "").strip(), (contexto or "").strip()
    if not instrucao:
        return f"{contexto}\n\n{texto}" if contexto else texto
    if contexto:
        user = f"Contexto: {contexto}\n\nPergunta: {texto}"
    else:
        user = f"Pergunta: {texto}"
    return f"{USER_TOKEN}{user}{ASSISTANT_TOKEN}"


class DeltaDecoder:
    """Decodifica ids em incrementos de texto seguros (sem quebrar caractere multibyte)."""

    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.ids: list[int] = []
        self.enviado = ""

    def push(self, tok_id: int) -> str:
        self.ids.append(tok_id)
        texto = self.tok.decode(self.ids, skip_special_tokens=False)
        if not texto.startswith(self.enviado):
            return ""  # o id novo mudou o prefixo (byte incompleto): espera o próximo
        novo = texto[len(self.enviado):]
        if novo.endswith("\ufffd"):
            return ""
        self.enviado = texto
        return novo


# ---------------------------------------------------------------- API


def _num(dados: dict, chave: str, padrao, lo, hi, inteiro: bool = False):
    try:
        valor = float(dados.get(chave, padrao))
    except (TypeError, ValueError):
        valor = float(padrao)
    valor = max(lo, min(hi, valor))
    return int(valor) if inteiro else valor


def params_de(dados: dict) -> dict:
    """Clampa os hiperparâmetros vindos do navegador (nunca confie no cliente)."""
    return {
        "temperature": _num(dados, "temperature", 0.0, 0.0, 2.0),
        "top_p": _num(dados, "top_p", 1.0, 0.05, 1.0),
        "top_k": _num(dados, "top_k", 0, 0, 200, inteiro=True),
        "repetition_penalty": _num(dados, "repetition_penalty", 1.15, 1.0, 2.0),
        "max_new_tokens": _num(dados, "max_new_tokens", 128, 1, 512, inteiro=True),
        "seed": _num(dados, "seed", -1, -1, 2**31 - 1, inteiro=True),
    }


def sse(evento: str, dados: dict) -> str:
    return f"event: {evento}\ndata: {json.dumps(dados, ensure_ascii=False)}\n\n"


def _modelo_pedido(srv: "Server", dados: dict) -> tuple[str | None, str | None, int]:
    """`(modelo_id, erro, status)` para o que veio do navegador — vale para todas as rotas."""
    if not srv.specs:
        return None, "nenhum checkpoint encontrado em runs/ — treine o modelo", 503
    model_id = dados.get("modelo") or next(iter(srv.specs))
    if model_id not in srv.specs:
        return None, f"modelo desconhecido: {model_id}", 400
    return model_id, None, 200


def montar_prompt_lab(texto: str, instrucao: bool = True) -> str:
    """O invólucro de chat **sem** o rótulo `Pergunta:` — `<|user|>texto<|assistant|>`.

    Não é o prompt do treino (esse é `montar_prompt`, e o chat continua usando ele): o lab
    quer mostrar os dois tokens de chat cercando o texto do aluno, e `Pergunta:` é um detalhe
    do dataset, não da arquitetura. No **base** não há invólucro — `<|user|>` nem existe no
    vocab dele (16.384), e tokenizá-lo produziria `<`, `|`, `user`…, o que confundiria.
    """
    return f"{USER_TOKEN}{texto}{ASSISTANT_TOKEN}" if instrucao else texto


def create_app(server: Server | None = None) -> Flask:
    app = Flask(__name__)
    app.config["PINGO"] = server or Server(
        [ModelSpec("sft", "Pingo SFT (instrução)", ROOT / "runs" / "sft-120m-ep1" / "final", True),
         ModelSpec("base", "Pingo base (pré-treino)", ROOT / "runs" / "pt-120m" / "final", False)])
    app.json.ensure_ascii = False

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/logo.png")
    def logo():
        return send_file(ROOT / "logo.png", mimetype="image/png")

    @app.get("/api/status")
    def status():
        return jsonify(app.config["PINGO"].info())

    @app.post("/api/chat")
    def chat():
        srv: Server = app.config["PINGO"]
        dados = request.get_json(silent=True) or {}
        texto = (dados.get("mensagem") or "").strip()
        contexto = (dados.get("contexto") or "").strip()
        if not texto:
            return jsonify({"erro": "mensagem vazia"}), 400
        if len(texto) > MAX_PROMPT_CHARS or len(contexto) > MAX_CONTEXT_CHARS:
            return jsonify({"erro": f"texto acima de {MAX_PROMPT_CHARS} caracteres"}), 400
        model_id, erro, status = _modelo_pedido(srv, dados)
        if erro:
            return jsonify({"erro": erro}), status
        if not srv.gen_lock.acquire(blocking=False):
            return jsonify({"erro": "o modelo já está gerando outra resposta"}), 429

        try:
            carregado = srv.carregar(model_id)       # pode levar ~1 s (disco → GPU)
            prompt = montar_prompt(texto, contexto, carregado.spec.instrucao)
            ids = carregado.tokenizer.encode(prompt, add_special_tokens=False).ids
        except Exception as exc:  # noqa: BLE001 — devolve o erro para a UI, não 500 mudo
            srv.gen_lock.release()  # nada foi gerado ainda: o gerador não vai rodar
            return jsonify({"erro": f"falha ao carregar o modelo: {exc}"}), 500
        ctx_max = carregado.model.config.max_position_embeddings
        if len(ids) >= ctx_max:
            srv.gen_lock.release()
            return jsonify({"erro": f"prompt maior que o contexto do modelo ({ctx_max} tokens)"}), 400

        return Response(stream(carregado, ids, params_de(dados), srv),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # -- modo didático (passo 1: do texto ao id e ao vetor inicial) ------
    # Nada aqui põe modelo na GPU: tokeniza com o tokenizer do checkpoint e lê a tabela de
    # embeddings do .safetensors na CPU. Por isso o lab não usa o lock de geração.

    @app.get("/lab")
    def lab():
        if not PAGINA_LAB.exists():
            return jsonify({"erro": f"página do laboratório ausente: {PAGINA_LAB}"}), 500
        return send_file(PAGINA_LAB, mimetype="text/html")

    @app.post("/api/lab/tokenize")
    def lab_tokenize():
        srv: Server = app.config["PINGO"]
        dados = request.get_json(silent=True) or {}
        texto = dados.get("texto") or ""          # não strip: espaço também é token
        model_id, erro, status = _modelo_pedido(srv, dados)
        if erro:
            return jsonify({"erro": erro}), status
        if not texto.strip():
            return jsonify({"erro": "texto vazio"}), 400
        if len(texto) > MAX_LAB_CHARS:
            return jsonify({"erro": f"texto acima de {MAX_LAB_CHARS} caracteres"}), 400
        instrucao = srv.specs[model_id].instrucao
        prompt = montar_prompt_lab(texto, instrucao)
        try:
            saida = tokenizar(srv.tokenizador_de(model_id), prompt)
        except Exception as exc:  # noqa: BLE001 — a UI mostra o motivo, não um 500 mudo
            return jsonify({"erro": f"falha ao tokenizar: {exc}"}), 500
        saida["prompt"] = prompt
        saida["modelo"] = model_id
        saida["rotulo"] = srv.specs[model_id].label
        saida["chat"] = instrucao               # sem invólucro de chat no modelo base
        return jsonify(saida)

    @app.get("/api/lab/embedding")
    def lab_embedding():
        srv: Server = app.config["PINGO"]
        model_id, erro, status = _modelo_pedido(srv, request.args)
        if erro:
            return jsonify({"erro": erro}), status
        try:
            tok_id = int(request.args.get("id", ""))
        except ValueError:
            return jsonify({"erro": "falta o id do token (inteiro)"}), 400
        k = int(_num(request.args, "k", VIZINHOS_PADRAO, 1, 50, inteiro=True))
        try:
            ficha = srv.tabela_de(model_id).ficha(tok_id, k=k)
        except (IndexError, KeyError, FileNotFoundError, ValueError) as exc:
            return jsonify({"erro": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify(ficha)

    # -- passo ②: a frase como matriz, a tabela inteira, e a tabela lida de volta --

    @app.post("/api/lab/matriz")
    def lab_matriz():
        srv: Server = app.config["PINGO"]
        dados = request.get_json(silent=True) or {}
        model_id, erro, status = _modelo_pedido(srv, dados)
        if erro:
            return jsonify({"erro": erro}), status
        try:
            ids = [int(x) for x in dados.get("ids") or []][:MAX_LAB_TOKENS]
        except (TypeError, ValueError):
            return jsonify({"erro": "ids tem que ser uma lista de inteiros"}), 400
        if not ids:
            return jsonify({"erro": "mande os ids da frase"}), 400
        try:
            saida = srv.tabela_de(model_id).matriz(ids)
        except (IndexError, KeyError, FileNotFoundError, ValueError) as exc:
            return jsonify({"erro": f"{type(exc).__name__}: {exc}"}), 400
        saida["modelo"] = model_id
        return jsonify(saida)

    @app.get("/api/lab/stats")
    def lab_stats():
        srv: Server = app.config["PINGO"]
        model_id, erro, status = _modelo_pedido(srv, request.args)
        if erro:
            return jsonify({"erro": erro}), status
        try:
            saida = srv.tabela_de(model_id).estatisticas()
        except (KeyError, FileNotFoundError, ValueError) as exc:
            return jsonify({"erro": f"{type(exc).__name__}: {exc}"}), 400
        saida["modelo"] = model_id
        return jsonify(saida)

    @app.get("/api/lab/lida")
    def lab_lida():
        srv: Server = app.config["PINGO"]
        model_id, erro, status = _modelo_pedido(srv, request.args)
        if erro:
            return jsonify({"erro": erro}), status
        try:
            tok_id = int(request.args.get("id", ""))
        except ValueError:
            return jsonify({"erro": "falta o id do token (inteiro)"}), 400
        k = int(_num(request.args, "k", LIDA_PADRAO, 1, 50, inteiro=True))
        try:
            saida = srv.tabela_de(model_id).lida_de_volta(tok_id, k=k)
        except (IndexError, KeyError, FileNotFoundError, ValueError) as exc:
            return jsonify({"erro": f"{type(exc).__name__}: {exc}"}), 400
        saida["modelo"] = model_id
        return jsonify(saida)

    # -- passos ③–④: RoPE + atenção e o resto do bloco (o lab carrega o modelo — com lock) --

    @app.post("/api/lab/attn")
    def lab_attn():
        srv: Server = app.config["PINGO"]
        dados = request.get_json(silent=True) or {}
        model_id, erro, status = _modelo_pedido(srv, dados)
        if erro:
            return jsonify({"erro": erro}), status
        try:
            ids = [int(x) for x in (dados.get("ids") or [])][:MAX_ATTN_TOKENS]
        except (TypeError, ValueError):
            return jsonify({"erro": "ids tem que ser uma lista de inteiros"}), 400
        if not ids:
            return jsonify({"erro": "mande os ids da frase"}), 400
        if not srv.gen_lock.acquire(blocking=False):
            return jsonify({"erro": "o modelo está gerando no chat — espere a resposta acabar"}), 429
        try:
            cap = srv.captura_atencao(model_id, ids)
            camada = int(_num(dados, "camada", 0, 0, 999, inteiro=True))
            n_camadas = len(cap["camadas"])
            if not 0 <= camada < n_camadas:
                return jsonify({"erro": f"camada fora do intervalo (0..{n_camadas - 1})"}), 400
            saida = {"modelo": model_id, "camada": camada, "n_camadas": n_camadas,
                     "heads": cap["heads"], "n": len(ids),
                     "pesos": [[[round(float(v), 5) for v in linha] for linha in cabeca]
                               for cabeca in cap["camadas"][camada]["w"]]}
        except Exception as exc:  # noqa: BLE001 — um forward real falha de muitos jeitos
            return jsonify({"erro": f"{type(exc).__name__}: {exc}"}), 500
        finally:
            srv.gen_lock.release()
        return jsonify(saida)

    @app.get("/api/lab/rope")
    def lab_rope():
        srv: Server = app.config["PINGO"]
        model_id, erro, status = _modelo_pedido(srv, request.args)
        if erro:
            return jsonify({"erro": erro}), status
        try:
            ids = [int(x) for x in (request.args.get("ids") or "").replace(" ", "").split(",")
                   if x][:MAX_ATTN_TOKENS]
        except ValueError:
            return jsonify({"erro": "ids tem que ser inteiros separados por vírgula"}), 400
        if not ids:
            return jsonify({"erro": "mande os ids da frase"}), 400
        camada = int(_num(request.args, "camada", 0, 0, 999, inteiro=True))
        head = int(_num(request.args, "head", 0, 0, 999, inteiro=True))
        i = int(_num(request.args, "i", 0, 0, 999, inteiro=True))
        if not srv.gen_lock.acquire(blocking=False):
            return jsonify({"erro": "o modelo está gerando no chat — espere a resposta acabar"}), 429
        try:
            cap = srv.captura_atencao(model_id, ids)
            if not (0 <= camada < len(cap["camadas"]) and 0 <= head < cap["heads"]
                    and 0 <= i < len(ids)):
                return jsonify({"erro": "camada/head/i fora do intervalo"}), 400
            c, hd, theta = cap["camadas"][camada], cap["head_dim"], cap["theta"]
            valores = lambda t: [round(float(x), 5) for x in t.tolist()]
            saida = {"modelo": model_id, "camada": camada, "head": head, "i": i,
                     "pos": i, "head_dim": hd, "theta": theta,
                     "inv_freq": [round(theta ** (-2.0 * j / hd), 8)
                                  for j in range(hd // 2)],
                     "q_lin": valores(c["q_lin"][head][i]), "q": valores(c["q"][head][i]),
                     "k_lin": valores(c["k_lin"][head][i]), "k": valores(c["k"][head][i])}
        except Exception as exc:  # noqa: BLE001
            return jsonify({"erro": f"{type(exc).__name__}: {exc}"}), 500
        finally:
            srv.gen_lock.release()
        return jsonify(saida)

    # -- passo ④: o resto do bloco (RMSNorm + SwiGLU) — carona no mesmo forward do ③ --

    @app.post("/api/lab/bloco")
    def lab_bloco():
        srv: Server = app.config["PINGO"]
        dados = request.get_json(silent=True) or {}
        model_id, erro, status = _modelo_pedido(srv, dados)
        if erro:
            return jsonify({"erro": erro}), status
        try:
            ids = [int(x) for x in (dados.get("ids") or [])][:MAX_ATTN_TOKENS]
        except (TypeError, ValueError):
            return jsonify({"erro": "ids tem que ser uma lista de inteiros"}), 400
        if not ids:
            return jsonify({"erro": "mande os ids da frase"}), 400
        camada = int(_num(dados, "camada", 0, 0, 999, inteiro=True))
        tok_i = int(_num(dados, "i", 0, 0, MAX_ATTN_TOKENS - 1, inteiro=True))
        if not srv.gen_lock.acquire(blocking=False):
            return jsonify({"erro": "o modelo está gerando no chat — espere a resposta acabar"}), 429
        try:
            cap = srv.captura_atencao(model_id, ids)
            if not 0 <= camada < len(cap["camadas"]):
                return jsonify({"erro": f"camada fora do intervalo (0..{len(cap['camadas']) - 1})"}), 400
            saida = ficha_bloco(cap["camadas"][camada], ids,
                                srv.tabela_de(model_id).id_para_token, tok_i)
            saida.update({"modelo": model_id, "camada": camada,
                          "n_camadas": len(cap["camadas"])})
        except (IndexError, KeyError, FileNotFoundError, ValueError) as exc:
            return jsonify({"erro": f"{type(exc).__name__}: {exc}"}), 400
        except Exception as exc:  # noqa: BLE001 — um forward real falha de muitos jeitos
            return jsonify({"erro": f"{type(exc).__name__}: {exc}"}), 500
        finally:
            srv.gen_lock.release()
        return jsonify(saida)

    # -- passo ⑤: lm head — carona no MESMO forward: norm final + logits reais --

    @app.post("/api/lab/lmhead")
    def lab_lmhead():
        srv: Server = app.config["PINGO"]
        dados = request.get_json(silent=True) or {}
        model_id, erro, status = _modelo_pedido(srv, dados)
        if erro:
            return jsonify({"erro": erro}), status
        try:
            ids = [int(x) for x in (dados.get("ids") or [])][:MAX_ATTN_TOKENS]
        except (TypeError, ValueError):
            return jsonify({"erro": "ids tem que ser uma lista de inteiros"}), 400
        if not ids:
            return jsonify({"erro": "mande os ids da frase"}), 400
        tok_i = int(_num(dados, "i", 0, 0, MAX_ATTN_TOKENS - 1, inteiro=True))
        k = int(_num(dados, "k", TOPK_LMHEAD, 1, 50, inteiro=True))
        if not srv.gen_lock.acquire(blocking=False):
            return jsonify({"erro": "o modelo está gerando no chat — espere a resposta acabar"}), 429
        try:
            cap = srv.captura_atencao(model_id, ids)
            ultimo = cap["camadas"][-1]
            saida = ficha_lmhead({"x2": ultimo["x2"], "logits": cap["logits"],
                                  "g_fin": cap["g_fin"], **cap["fin"]},
                                 ids, srv.tabela_de(model_id).id_para_token, tok_i, k=k)
            saida.update({"modelo": model_id, "n_camadas": len(cap["camadas"])})
        except (IndexError, KeyError, FileNotFoundError, ValueError) as exc:
            return jsonify({"erro": f"{type(exc).__name__}: {exc}"}), 400
        except Exception as exc:  # noqa: BLE001 — um forward real falha de muitos jeitos
            return jsonify({"erro": f"{type(exc).__name__}: {exc}"}), 500
        finally:
            srv.gen_lock.release()
        return jsonify(saida)

    @app.get("/api/lab/lookup")
    def lab_lookup():
        srv: Server = app.config["PINGO"]
        model_id, erro, status = _modelo_pedido(srv, request.args)
        if erro:
            return jsonify({"erro": erro}), status
        consulta = (request.args.get("q") or "").strip()
        if not consulta:
            return jsonify({"erro": "diga um id ou uma palavra"}), 400
        limite = int(_num(request.args, "n", 20, 1, 50, inteiro=True))
        try:
            achados = buscar(srv.tokenizador_de(model_id), consulta, limite=limite)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"erro": f"{type(exc).__name__}: {exc}"}), 500
        return jsonify({"consulta": consulta, "modelo": model_id, "achados": achados})

    return app


def stream(carregado: Loaded, ids: list[int], params: dict,
           srv: Server) -> Iterator[str]:
    """Gerador SSE: meta → deltas → done (sempre solta o lock no fim)."""
    t0 = time.perf_counter()
    dec = DeltaDecoder(carregado.tokenizer)
    n, primeiro = 0, None
    parada = "max_tokens"
    try:
        yield sse("meta", {"modelo": carregado.spec.id, "rotulo": carregado.spec.label,
                           "instrucao": carregado.spec.instrucao,
                           "prompt_tokens": len(ids), "params": params})
        for tok in generate_stream(
                carregado.model, ids,
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"], top_k=params["top_k"] or None,
                top_p=params["top_p"], repetition_penalty=params["repetition_penalty"],
                stop_token=carregado.end_id if carregado.spec.instrucao else None,
                seed=None if params["seed"] < 0 else params["seed"]):
            if primeiro is None:
                primeiro = time.perf_counter()
            n += 1
            if carregado.end_id is not None and tok == carregado.end_id:
                parada = "end_token"
                break
            novo = dec.push(tok)
            if novo:
                yield sse("delta", {"t": novo})
        segundos = time.perf_counter() - t0
        ttft = (primeiro - t0) if primeiro else 0.0
        yield sse("done", {"tokens": n, "segundos": round(segundos, 3),
                           "ttft": round(ttft, 3),
                           "tok_s": round(n / segundos, 1) if segundos > 0 else 0.0,
                           "parada": parada})
    except GeneratorExit:  # cliente apertou "parar" ou fechou a aba
        raise
    except Exception as exc:  # noqa: BLE001
        yield sse("erro", {"erro": f"{type(exc).__name__}: {exc}"})
    finally:
        if srv.gen_lock.locked():
            srv.gen_lock.release()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0",
                   help="0.0.0.0 = LAN; 127.0.0.1 = só esta máquina")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--sft", type=Path, default=ROOT / "runs" / "sft-120m-ep1" / "final")
    p.add_argument("--base", type=Path, default=ROOT / "runs" / "pt-120m" / "final")
    return p.parse_args(argv)


def ip_da_lan() -> str | None:
    """IP da interface que fala com o resto da rede (o UDP não chega a sair da máquina)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))   # só escolhe a rota; nenhum pacote é enviado
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    server = Server([ModelSpec("sft", "Pingo SFT (instrução)", args.sft, True),
                     ModelSpec("base", "Pingo base (pré-treino)", args.base, False)],
                    device=args.device)
    if not server.specs:
        sys.exit("nenhum checkpoint de SFT nem de base em runs/")
    print(f"Pingo WebUI — device {server.device} — modelos: {', '.join(server.specs)}")
    print(f"  local:      http://127.0.0.1:{args.port}")
    if args.host in ("0.0.0.0", "::"):
        lan = ip_da_lan()
        print(f"  na rede:    http://{lan}:{args.port}" if lan
              else "  na rede:    (sem interface de rede detectada)")
        print("  atenção:    sem autenticação — qualquer um na mesma rede usa o modelo")
    else:
        print(f"  escutando:  {args.host}:{args.port} (só este host)")
    create_app(server).run(host=args.host, port=args.port, threaded=True,
                           debug=False, use_reloader=False)


# ---------------------------------------------------------------- página


PAGE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pingo — WebUI</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>💧</text></svg>">
<style>
  :root{
    --bg:#0a0c10; --bg2:#0e1117; --panel:#12161d; --panel2:#161b24; --line:#1f2632;
    --line2:#2a3342; --text:#e7ebf2; --muted:#8d98aa; --dim:#5f6b7d;
    --accent:#3b7bfb; --accent2:#5b93ff; --accent-dim:#1b2b4d;
    --green:#3ddc84; --yellow:#ffd166; --red:#ff6b6b;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0;background:var(--bg);color:var(--text);
    font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  body::before{
    content:"";position:fixed;inset:-20% 0 auto 0;height:520px;pointer-events:none;
    background:radial-gradient(60% 100% at 50% 0%,rgba(59,123,251,.16),transparent 70%);
  }
  .wrap{position:relative;max-width:900px;margin:0 auto;padding:0 20px 190px}
  header{
    position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:14px;
    padding:14px 0;background:linear-gradient(var(--bg) 65%,transparent);
    backdrop-filter:blur(6px);
  }
  .brand{display:flex;align-items:center;gap:10px;min-width:0}
  .brand svg{width:30px;height:30px;flex:none;filter:drop-shadow(0 2px 8px rgba(59,123,251,.45))}
  .brand b{font-size:17px;letter-spacing:.2px}
  .brand span{color:var(--dim);font-size:12.5px;white-space:nowrap}
  .spacer{flex:1}
  .models{display:flex;gap:4px;padding:4px;background:var(--panel);border:1px solid var(--line);
    border-radius:11px}
  .models button{
    border:0;background:transparent;color:var(--muted);font:inherit;font-size:12.5px;
    padding:6px 11px;border-radius:8px;cursor:pointer;white-space:nowrap;transition:.15s;
  }
  .models button:hover{color:var(--text)}
  .models button.on{background:var(--accent-dim);color:#cfe0ff;box-shadow:inset 0 0 0 1px #2b4d8f}
  .models button:disabled{opacity:.45;cursor:not-allowed}
  .lablink{display:inline-flex;align-items:center;gap:6px;text-decoration:none;color:var(--muted);
    font-size:12.5px;border:1px solid var(--line);background:var(--panel);padding:8px 12px;
    border-radius:10px;transition:.15s;white-space:nowrap}
  .lablink:hover{color:var(--text);border-color:var(--line2);transform:translateY(-1px)}
  #chat{display:flex;flex-direction:column;gap:18px;padding-top:10px}
  .msg{display:flex;gap:12px;animation:in .25s ease}
  @keyframes in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  .msg .ava{
    width:28px;height:28px;flex:none;border-radius:9px;display:grid;place-items:center;
    font-size:13px;font-weight:600;background:var(--panel2);border:1px solid var(--line);
    color:var(--muted)
  }
  .msg.user .ava{background:var(--accent-dim);border-color:#2b4d8f;color:#cfe0ff}
  .bubble{min-width:0;max-width:100%}
  .who{font-size:11.5px;color:var(--dim);margin-bottom:3px;letter-spacing:.4px;
    text-transform:uppercase}
  .text{white-space:pre-wrap;overflow-wrap:anywhere}
  .msg.assistant .text{background:var(--panel);border:1px solid var(--line);
    padding:12px 15px;border-radius:4px 13px 13px 13px}
  .msg.user .text{background:linear-gradient(180deg,#1d3564,#18294b);border:1px solid #274377;
    padding:12px 15px;border-radius:13px 4px 13px 13px}
  .foot{display:flex;align-items:center;gap:10px;margin-top:6px;font:11.5px/1 var(--mono);
    color:var(--dim)}
  .foot button{background:none;border:0;color:var(--dim);font:inherit;cursor:pointer;
    padding:2px 4px;border-radius:5px}
  .foot button:hover{color:var(--text);background:var(--panel2)}
  .cursor::after{
    content:"";display:inline-block;width:7px;height:15px;margin-left:2px;
    background:var(--accent2);vertical-align:-2px;animation:blink 1s steps(2) infinite;
  }
  @keyframes blink{50%{opacity:0}}
  .err{color:#ffb4b4;background:#2a1518;border:1px solid #5a2a2f;padding:10px 13px;
    border-radius:10px;font-size:14px}
  /* vazio */
  #empty{display:flex;flex-direction:column;align-items:center;text-align:center;
    gap:10px;padding:64px 0 30px;color:var(--muted)}
  #empty svg{width:74px;height:74px;opacity:.95}
  #empty h1{font-size:20px;margin:6px 0 0;color:var(--text);font-weight:600}
  #empty p{margin:0;font-size:13.5px;max-width:430px}
  .chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:16px}
  .chip{background:var(--panel);border:1px solid var(--line);color:var(--muted);
    padding:8px 13px;border-radius:999px;font-size:12.5px;cursor:pointer;transition:.15s}
  .chip:hover{color:var(--text);border-color:var(--line2);transform:translateY(-1px)}
  /* composer */
  .dock{position:fixed;left:0;right:0;bottom:0;z-index:30;padding:14px 20px 16px;
    background:linear-gradient(transparent,var(--bg) 28%)}
  .composer{max-width:860px;margin:0 auto;background:var(--panel);border:1px solid var(--line2);
    border-radius:16px;box-shadow:0 14px 40px rgba(0,0,0,.45);overflow:hidden}
  .composer:focus-within{border-color:#33507f;box-shadow:0 14px 40px rgba(0,0,0,.5),0 0 0 3px rgba(59,123,251,.12)}
  .ctx{margin:0;border-bottom:1px solid var(--line);background:var(--bg2)}
  .ctx summary{cursor:pointer;list-style:none;padding:9px 14px;font-size:12px;color:var(--dim);
    display:flex;align-items:center;gap:7px;user-select:none}
  .ctx summary::-webkit-details-marker{display:none}
  .ctx summary:hover{color:var(--muted)}
  .ctx textarea{width:100%;border:0;background:transparent;color:var(--text);resize:vertical;
    min-height:74px;padding:0 14px 12px;font:13px/1.55 var(--mono);outline:none}
  .row{display:flex;align-items:flex-end;gap:10px;padding:10px 10px 10px 18px}
  textarea#q{flex:1;border:0;background:transparent;color:var(--text);resize:none;
    font:inherit;line-height:1.5;max-height:180px;padding:6px 0;outline:none}
  textarea#q::placeholder{color:var(--dim)}
  .send{width:38px;height:38px;flex:none;border-radius:11px;border:0;cursor:pointer;
    background:var(--accent);color:#fff;display:grid;place-items:center;transition:.15s}
  .send:hover{background:var(--accent2)}
  .send:disabled{background:var(--panel2);color:var(--dim);cursor:not-allowed}
  .send.stop{background:#3a1d22;color:#ffb4b4;border:1px solid #5a2a2f}
  .params{border-top:1px solid var(--line);display:none;gap:20px;flex-wrap:wrap;
    padding:12px 18px 14px}
  .params.open{display:flex}
  .param{display:flex;flex-direction:column;gap:5px;min-width:128px}
  .param label{font-size:11px;color:var(--dim);letter-spacing:.4px;text-transform:uppercase;
    display:flex;justify-content:space-between;gap:8px}
  .param label b{color:var(--muted);font:11px var(--mono)}
  input[type=range]{-webkit-appearance:none;height:3px;border-radius:3px;background:var(--line2);
    outline:none;width:100%}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;
    border-radius:50%;background:var(--accent2);cursor:pointer;border:2px solid var(--panel)}
  .bar{max-width:860px;margin:8px auto 0;display:flex;align-items:center;gap:12px;
    font:11px var(--mono);color:var(--dim);padding:0 4px}
  .bar button{background:none;border:0;color:var(--dim);font:inherit;cursor:pointer;
    padding:3px 6px;border-radius:5px}
  .bar button:hover{color:var(--text);background:var(--panel2)}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--green);display:inline-block;
    box-shadow:0 0 8px var(--green)}
  .dot.idle{background:var(--dim);box-shadow:none}
  kbd{font:11px var(--mono);background:var(--panel2);border:1px solid var(--line2);
    border-bottom-width:2px;padding:1px 5px;border-radius:4px;color:var(--muted)}
  @media (max-width:640px){
    .brand span{display:none} .wrap{padding:0 12px 200px} .dock{padding:10px 12px 12px}
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <svg viewBox="0 0 32 32" aria-hidden="true">
        <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#5b93ff"/><stop offset="1" stop-color="#1f4bb8"/>
        </linearGradient></defs>
        <path d="M16 3C11 10 6 14.2 6 20a10 10 0 0 0 20 0c0-5.8-5-10-10-17z" fill="url(#g)"/>
        <ellipse cx="12.6" cy="20.4" rx="2" ry="2.6" fill="#0a0c10" opacity=".82"/>
        <ellipse cx="19.4" cy="20.4" rx="2" ry="2.6" fill="#0a0c10" opacity=".82"/>
        <circle cx="13.3" cy="19.5" r=".7" fill="#fff"/><circle cx="20.1" cy="19.5" r=".7" fill="#fff"/>
      </svg>
      <div><b>Pingo</b> <span>· 120M em português</span></div>
    </div>
    <div class="spacer"></div>
    <a class="lablink" href="/lab" title="Ver o modelo por dentro: tokenização, cores, embeddings">🔬 laboratório</a>
    <div class="models" id="models"></div>
  </header>

  <main id="chat"></main>
  <div id="empty">
    <svg viewBox="0 0 32 32" aria-hidden="true">
      <defs><linearGradient id="g2" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#5b93ff"/><stop offset="1" stop-color="#1f4bb8"/>
      </linearGradient></defs>
      <path d="M16 3C11 10 6 14.2 6 20a10 10 0 0 0 20 0c0-5.8-5-10-10-17z" fill="url(#g2)"/>
      <ellipse cx="12.6" cy="20.4" rx="2" ry="2.6" fill="#0a0c10" opacity=".82"/>
      <ellipse cx="19.4" cy="20.4" rx="2" ry="2.6" fill="#0a0c10" opacity=".82"/>
      <circle cx="13.3" cy="19.5" r=".7" fill="#fff"/><circle cx="20.1" cy="19.5" r=".7" fill="#fff"/>
    </svg>
    <h1>Pergunte ao Pingo</h1>
    <p id="sub">Modelo de 120M parâmetros treinado do zero em notícias em português.
      Cole um trecho no <em>Contexto</em> para ele responder com base no texto.</p>
    <div class="chips" id="chips"></div>
  </div>
</div>

<div class="dock">
  <div class="bar">
    <span id="status"><span class="dot idle"></span> <span id="statustext">conectando…</span></span>
    <div style="flex:1"></div>
    <button id="toggleparams" title="Parâmetros de amostragem">parâmetros</button>
    <button id="clear" title="Nova conversa">nova conversa</button>
  </div>
  <div class="composer">
    <details class="ctx" id="ctxbox">
      <summary>▸ Contexto (opcional) — cole o trecho que o Pingo deve ler</summary>
      <textarea id="ctx" placeholder="Cole aqui o texto de apoio. O modelo recebe: Contexto: … / Pergunta: …"></textarea>
    </details>
    <div class="row">
      <textarea id="q" rows="1" placeholder="Escreva sua pergunta…"></textarea>
      <button class="send" id="send" title="Enviar (Enter)">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 19V5M5 12l7-7 7 7"/></svg>
      </button>
    </div>
    <div class="params" id="params">
      <div class="param"><label>temperatura <b id="v-temp">0</b></label>
        <input type="range" id="p-temp" min="0" max="1.5" step="0.05" value="0"></div>
      <div class="param"><label>top-p <b id="v-topp">1</b></label>
        <input type="range" id="p-topp" min="0.1" max="1" step="0.05" value="1"></div>
      <div class="param"><label>top-k <b id="v-topk">0</b></label>
        <input type="range" id="p-topk" min="0" max="100" step="5" value="0"></div>
      <div class="param"><label>penalidade <b id="v-rep">1.15</b></label>
        <input type="range" id="p-rep" min="1" max="1.5" step="0.05" value="1.15"></div>
      <div class="param"><label>máx. tokens <b id="v-max">128</b></label>
        <input type="range" id="p-max" min="16" max="512" step="16" value="128"></div>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const EXAMPLES = [
  {q:"Quanto a companhia planeja investir até 2028?",
   c:"A Companhia Energética de Minas Gerais anunciou um plano de investimentos de 12 bilhões de reais até 2028, com foco em redes de distribuição no interior do estado."},
  {q:"O que o ministro disse sobre a reforma?", c:""},
  {q:"O que é aprendizado de máquina?", c:""}
];
const state = {modelos:[], modelo:null, busy:false, params:{
  temperature:0, top_p:1, top_k:0, repetition_penalty:1.15, max_new_tokens:128}};

function fmt(n){ return n>=1000 ? (n/1000).toFixed(1).replace('.0','')+'k' : String(n); }

/* ---------------------------------------------------------------- modelos */
async function carregarModelos(){
  const r = await fetch('/api/status'); const s = await r.json();
  state.modelos = s.modelos;
  const box = $('#models'); box.innerHTML = '';
  if(!s.modelos.length){
    box.innerHTML = '<button disabled>nenhum checkpoint</button>';
    $('#statustext').textContent = 'sem modelo em runs/';
    return;
  }
  if(!state.modelo) state.modelo = (s.modelos.find(m=>m.instrucao) || s.modelos[0]).id;
  for(const m of s.modelos){
    const b = document.createElement('button');
    b.textContent = m.instrucao ? 'SFT' : 'base';
    b.title = m.label;
    b.classList.toggle('on', m.id === state.modelo);
    b.onclick = () => { state.modelo = m.id; [...box.children].forEach(x=>x.classList.remove('on'));
      b.classList.add('on'); atualizarSub(); };
    box.appendChild(b);
  }
  $('#statustext').innerHTML = `<span class="dot"></span> ${s.device === 'cuda' ? 'GPU' : 'CPU'} pronta`;
  atualizarSub();
}
function atual(){
  return state.modelos.find(m=>m.id === state.modelo) || {instrucao:true};
}
function atualizarSub(){
  const inst = atual().instrucao;
  $('#sub').innerHTML = inst
    ? 'Modelo de 120M parâmetros treinado do zero em notícias em português. Cole um trecho no <em>Contexto</em> para ele responder com base no texto.'
    : 'Este é o modelo <b>base</b>: ele só continua o texto, não segue instruções. Escreva o começo de uma frase e ele segue.';
  $('#q').placeholder = inst ? 'Escreva sua pergunta…' : 'Escreva o começo do texto…';
}

/* ---------------------------------------------------------------- render */
function addMsg(papel, texto){
  $('#empty').style.display = 'none';
  const el = document.createElement('div');
  el.className = 'msg ' + papel;
  el.innerHTML = `<div class="ava">${papel === 'user' ? 'você' : '💧'}</div>
    <div class="bubble"><div class="who">${papel === 'user' ? 'você' : 'pingo'}</div>
    <div class="text"></div><div class="foot"></div></div>`;
  el.querySelector('.text').textContent = texto || '';
  $('#chat').appendChild(el);
  return el;
}
function pertoDoFim(){ return window.innerHeight + window.scrollY >= document.body.scrollHeight - 160; }
function rolar(forcar){ if(forcar || pertoDoFim()) window.scrollTo({top:document.body.scrollHeight}); }

/* ---------------------------------------------------------------- envio */
let ctrl = null;
async function enviar(){
  if(state.busy) return;
  const q = $('#q').value.trim(); if(!q) return;
  const contexto = $('#ctx').value.trim();
  addMsg('user', q);
  $('#q').value = ''; autoGrow();
  const alvo = addMsg('assistant', '');
  alvo.querySelector('.text').classList.add('cursor');
  rolar(true);
  setBusy(true);

  ctrl = new AbortController();
  const t0 = performance.now();
  let recebido = '', meta = null, stats = null;
  try{
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({mensagem:q, contexto, modelo:state.modelo, ...state.params}),
      signal: ctrl.signal});
    if(!r.ok){
      const erro = await r.json().catch(()=>({erro:`HTTP ${r.status}`}));
      throw new Error(erro.erro || `HTTP ${r.status}`);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream:true});
      let i;
      while((i = buf.indexOf('\\n\\n')) >= 0){
        const bloco = buf.slice(0, i); buf = buf.slice(i + 2);
        const ev = (bloco.match(/^event: (.*)$/m) || [])[1];
        const dados = JSON.parse((bloco.match(/^data: (.*)$/m) || ['', '{}'])[1]);
        if(ev === 'meta'){ meta = dados; }
        else if(ev === 'delta'){
          recebido += dados.t;
          const t = alvo.querySelector('.text');
          t.textContent = recebido; rolar();
        }
        else if(ev === 'done'){ stats = dados; }
        else if(ev === 'erro'){ throw new Error(dados.erro); }
      }
    }
  }catch(e){
    if(e.name !== 'AbortError'){
      alvo.querySelector('.text').classList.add('err');
      alvo.querySelector('.text').textContent = '⚠ ' + e.message;
    }
  }finally{
    alvo.querySelector('.text').classList.remove('cursor');
    const seg = (performance.now() - t0) / 1000;
    const s = stats || {};
    const partes = [];
    if(meta) partes.push(meta.rotulo);
    if(s.tokens) partes.push(`${s.tokens} tokens`);
    partes.push(`${seg.toFixed(1)} s`);
    if(s.tok_s) partes.push(`${s.tok_s} tok/s`);
    if(s.ttft) partes.push(`1º token ${(s.ttft*1000).toFixed(0)} ms`);
    const foot = alvo.querySelector('.foot');
    foot.innerHTML = `<span>${partes.join(' · ')}</span>`;
    const copiar = document.createElement('button');
    copiar.textContent = 'copiar';
    copiar.onclick = () => { navigator.clipboard.writeText(recebido);
      copiar.textContent = 'copiado!'; setTimeout(()=>copiar.textContent='copiar', 1200); };
    foot.appendChild(copiar);
    setBusy(false);
    ctrl = null;
    rolar();
    $('#q').focus();
  }
}

function setBusy(b){
  state.busy = b;
  const btn = $('#send');
  btn.classList.toggle('stop', b);
  btn.disabled = !b && !state.modelos.length;
  btn.innerHTML = b
    ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>'
    : '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
  btn.title = b ? 'Parar' : 'Enviar (Enter)';
  document.querySelectorAll('.models button').forEach(x=>x.disabled = b);
  $('#statustext').innerHTML = b
    ? '<span class="dot"></span> gerando…'
    : `<span class="dot"></span> ${state.modelos.length ? 'pronta' : 'sem modelo'}`;
}
$('#send').onclick = () => {
  if(state.busy){ if(ctrl) ctrl.abort(); return; }
  enviar();
};
$('#q').addEventListener('keydown', e => {
  if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); enviar(); }
});
function autoGrow(el){
  const t = $('#q'); t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight, 180) + 'px';
}
$('#q').addEventListener('input', autoGrow);
$('#clear').onclick = () => {
  $('#chat').innerHTML = ''; $('#empty').style.display = 'flex';
  $('#ctx').value = ''; $('#q').value = ''; autoGrow(); $('#q').focus();
};
$('#toggleparams').onclick = () => $('#params').classList.toggle('open');

/* params */
const mapa = {'p-temp':['temperature','v-temp',v=>v], 'p-topp':['top_p','v-topp',v=>v],
  'p-topk':['top_k','v-topk',v=>v], 'p-rep':['repetition_penalty','v-rep',v=>v],
  'p-max':['max_new_tokens','v-max',v=>v]};
for(const [id,[chave,label,conv]] of Object.entries(mapa)){
  const el = $('#' + id), out = $('#' + label);
  const salvo = localStorage.getItem('pingo-' + chave);
  if(salvo !== null){ el.value = salvo; }
  state.params[chave] = conv(el.value);
  out.textContent = el.value;
  el.oninput = () => {
    state.params[chave] = conv(el.value); out.textContent = el.value;
    localStorage.setItem('pingo-' + chave, el.value);
  };
}

/* exemplos */
const chips = $('#chips');
EXAMPLES.forEach((ex, i) => {
  const b = document.createElement('button');
  b.className = 'chip'; b.textContent = ex.q.length > 46 ? ex.q.slice(0,44) + '…' : ex.q;
  b.onclick = () => {
    $('#q').value = ex.q; autoGrow();
    if(ex.c){ $('#ctx').value = ex.c; $('#ctxbox').open = true; }
    $('#q').focus();
  };
  chips.appendChild(b);
});

carregarModelos().catch(e => { $('#statustext').textContent = 'servidor fora do ar'; });
$('#q').focus();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
