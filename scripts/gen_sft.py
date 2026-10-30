#!/usr/bin/env python
"""SFT-1 — gera o dataset de instruções (pergunta/resposta) com o professor local.

O professor é um Qwen3.5-4B servido em `llama.cpp` (`/v1/chat/completions`, 8 slots).
Cada chamada recebe **um trecho** de um doc do OSCAR (split `train`, nunca `val`) e devolve
um par em JSON estrito (`response_format` = `json_schema` + `strict`), com o campo
`evidencia` — cópia literal do trecho — que é a única verificação automática barata de que
o professor não inventou. O texto do professor é descartado; só o JSON é usado.

Pipeline por doc:

1. **filtro de fonte** (duro, bem mais que o do pré-treino): tamanho, nº de parágrafos,
   dígitos/palavra, maiúsculas, links; limpeza de boilerplate (redes sociais, menus).
2. **chunking** em parágrafos inteiros, até `--chunk-chars` (~2000 chars ≈ 500 tokens do
   nosso tokenizer), com mínimos p/ o trecho ser autossuficiente.
3. **professor** (8 em paralelo): tipo sorteado — `extracao` 50%, `explicacao` 30%,
   `resumo` 15%, `recusa` 5%. O professor pode ainda responder `invalido` (trecho que não
   é texto legível em pt): é um **veto de qualidade** de graça, que descarta o trecho.
4. **validação local**: JSON ok, evidência literal no trecho, tamanhos, pt, dedup exata de
   pergunta. Falhou → reamostra (até `--tries`).
5. **JSONL** com procedência (doc_id, url, split, teacher, prompt_version, seed, tipo) —
   uma linha por par, escrita incrementalmente (`--resume` continua de onde parou).

Uso:
  uv run python scripts/gen_sft.py --max-examples 20 --dry-run          # só amostra trechos
  uv run python scripts/gen_sft.py --max-examples 300 --out data/sft/pilot-v1.jsonl
  uv run python scripts/gen_sft.py --max-examples 30000 --workers 8 \\
      --out data/sft/sft-v1.jsonl --resume
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

PROMPT_VERSION = "v2"
DEFAULT_TEACHER_MODEL = "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL"
DEFAULT_TEACHER_URL = "http://localhost:8001/v1/chat/completions"

# proporção de tipos de exemplo (o resto é recusa, controlado por --recusa-frac)
TIPO_WEIGHTS = {"extracao": 0.50, "explicacao": 0.30, "resumo": 0.15}

# recusa: a frase depende de o exemplo levar o trecho junto (ancorado) ou não
RECUSA_ANCORADO = "Isso não consta no texto."
RECUSA_LIVRE = "Não tenho informações suficientes para responder isso."
RECUSA_PREFIXO = {True: "isso não consta", False: "não tenho informa"}

TIPO_DESCRICAO = {
    "extracao": "faça uma pergunta cuja resposta seja um fato explícito do trecho (nomes, "
                "números, datas, lugares).",
    "explicacao": "faça uma pergunta tipo 'por quê' ou 'como', cuja resposta exija juntar "
                  "duas ou três informações do trecho.",
    "resumo": "faça uma pergunta sobre a ideia central do trecho; a resposta deve resumir "
              "o trecho em 1 a 2 frases.",
    "recusa": "invente uma pergunta objetiva e plausível sobre o MESMO assunto do trecho, "
              "mas cuja resposta NÃO esteja no trecho (exigiria informação de fora).",
}

MODO_DESCRICAO = {
    True: "ANCORADO: a pergunta será feita junto com o TRECHO, que o leitor terá em mãos — "
          "ela pode se referir ao conteúdo ('segundo o texto', 'no caso descrito').",
    False: "SEM CONTEXTO: a pergunta será feita ao modelo SEM o TRECHO; ela precisa ser "
           "autossuficiente e NUNCA citar 'texto', 'trecho', 'documento' ou 'artigo'. "
           "Use o trecho só para saber do que falar: o assunto (entidade, evento, tema) "
           "precisa estar explícito na pergunta.",
}

SYSTEM_PROMPT = """Você monta dados de treino em português a partir de trechos de páginas web.
Responda SOMENTE com o objeto JSON pedido, sem comentários.

Regras:
- pergunta: uma única pergunta em português, clara, terminada em '?'.
- resposta: em português, 1 a 3 frases completas, usando APENAS informações do TRECHO.
- evidencia: cópia LITERAL de uma ou duas frases do TRECHO que sustentam a resposta.
  Copie caractere por caractere — nunca parafraseie, nunca resuma.
- tipo: exatamente o tipo pedido.
- status: 'ok' normalmente; 'invalido' se o TRECHO não for texto legível em português
  (lista de links, menu de site, texto embaralhado, propaganda, spam).
- O MODO de cada pedido diz se a pergunta vai ou não acompanhada do trecho: siga-o à risca."""

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "invalido"]},
        "tipo": {"type": "string", "enum": ["extracao", "explicacao", "resumo", "recusa"]},
        "pergunta": {"type": "string"},
        "resposta": {"type": "string"},
        "evidencia": {"type": "string"},
    },
    "required": ["status", "tipo", "pergunta", "resposta", "evidencia"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------- filtro de fonte

# blocklist de fonte (o professor veta muitos, mas não todos: spam adulto e sites de
# pirataria passavam por serem "texto legível"). Tudo é tunável pela CLI.
DENY_URL_DEFAULT = (r"(?i)(porn|xxx|nsfw|hentai|escort|onlyfans|putaria|sexo\b|crack|"
                    r"keygen|warez|torrent|nifentas|bet[0-9]{2}|casino|tigrinho|slot[s]?\b|"
                    r"assistir|dublado|legendado|filmes?online|megafilmes|superflix|"
                    r"series?online|baixar|gratisbaixar)")
DENY_TEXT_DEFAULT = (r"(?i)(v[ií]deos? picantes?|conte[uú]do adulto expl[ií]cito|"
                     r"maiores de 18 anos|apenas maiores de idade)")

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WORD_RE = re.compile(r"[^\W\d_]{2,}")
DIGIT_RE = re.compile(r"\d")
BOILER_RE = re.compile(
    r"(?i)^(?:facebook|twitter|whatsapp|instagram|linkedin|youtube|tiktok|telegram|"
    r"compartilh\w*|assine\b|assinar\b|publicidade|anúncio|propaganda|leia mais|leia também|"
    r"voltar|início|home|menu|pesquisar|buscar|cookie\w*|todos os direitos|política de "
    r"privacidade|termos de uso|siga-nos|siga o|newsletter|artigo em|ver mais|"
    r"©|copyright|<|>)"
)
PT_STOP = {
    "a", "o", "as", "os", "um", "uma", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "por", "para", "com", "que", "se", "não", "e", "é", "foi", "são", "ao",
    "aos", "à", "às", "como", "mais", "mas", "ou", "quando", "qual", "quais", "seu", "sua",
    "seus", "suas", "ele", "ela", "eles", "elas", "isso", "este", "esta", "esse", "essa",
    "entre", "sobre", "depois", "antes", "também", "já", "há", "ser", "tem", "tinha", "ter",
}


ACCENT_RE = re.compile(r"[áàâãéêíóôõúüç]", re.IGNORECASE)


def is_pt(text: str, min_signals: int = 2) -> bool:
    """Heurística barata de português: conta palavras funcionais + palavras acentuadas.

    Uma frase curta pode ter só uma palavra funcional ("Quanto a companhia investiu?"), por
    isso as duas pistas somam em vez de exigir várias ocorrências de uma só.
    """
    words = re.findall(r"[^\W\d_]+", text.lower())
    stops = sum(1 for w in words if w in PT_STOP)
    accents = sum(1 for w in words if ACCENT_RE.search(w))
    return stops + accents >= min_signals


def normalize_ws(text: str) -> str:
    """NFC + whitespace colapsado — base das comparações de evidência."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def domain_of(url: str) -> str:
    """Domínio normalizado (sem `www.`) — usado no teto por fonte."""
    netloc = urllib.parse.urlparse(url).netloc.lower()
    netloc = netloc.split("@")[-1].split(":")[0]
    return netloc[4:] if netloc.startswith("www.") else netloc


@dataclass
class Chunk:
    doc_id: str
    url: str
    split: str
    dump: str
    idx: int
    text: str


@dataclass
class Config:
    min_chars: int = 2000
    max_chars: int = 60_000
    min_paras: int = 4
    min_para_chars: int = 60
    max_digits_per_word: float = 0.25
    max_upper_ratio: float = 0.30
    max_urls: int = 4
    chunk_chars: int = 2000
    chunk_min_chars: int = 700
    max_per_doc: int = 3
    domain_cap_frac: float = 0.05
    domain_cap_min: int = 3
    ancorado_frac: float = 0.5
    recusa_frac: float = 0.05
    deny_url_regex: str = DENY_URL_DEFAULT
    deny_text_regex: str = DENY_TEXT_DEFAULT
    tries: int = 3
    workers: int = 8
    max_tokens: int = 400
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 1.5
    timeout: float = 180.0
    teacher_model: str = DEFAULT_TEACHER_MODEL
    teacher_url: str = DEFAULT_TEACHER_URL
    seed: int = 13


# ---------------------------------------------------------------- documentos


def clean_paragraphs(content: str, min_para_chars: int = 60) -> list[str]:
    """Quebra o doc em parágrafos úteis: tira boilerplate, linhas curtíssimas e links."""
    paras: list[str] = []
    for raw in content.split("\n"):
        line = " ".join(raw.split())
        if len(line) < min_para_chars:
            continue
        if BOILER_RE.match(line):
            continue
        if len(URL_RE.findall(line)) > 2:  # linha que é só lista de links
            continue
        paras.append(line)
    return paras


def doc_reject_reason(content: str, cfg: Config, url: str = "") -> str | None:
    """Filtro duro de fonte. Devolve o motivo da rejeição ou None se o doc serve."""
    if cfg.deny_url_regex and url and re.search(cfg.deny_url_regex, url):
        return "fonte_bloqueada"
    if cfg.deny_text_regex and re.search(cfg.deny_text_regex, content):
        return "conteudo_bloqueado"
    n = len(content)
    if n < cfg.min_chars:
        return "doc_curto"
    if n > cfg.max_chars:
        return "doc_longo"
    paras = clean_paragraphs(content, cfg.min_para_chars)
    if len(paras) < cfg.min_paras:
        return "poucos_paragrafos"
    words = WORD_RE.findall(content) or [""]
    if len(DIGIT_RE.findall(content)) / len(words) > cfg.max_digits_per_word:
        return "muitos_digitos"
    letters = [c for c in content if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > cfg.max_upper_ratio:
        return "muitas_maiusculas"
    if len(URL_RE.findall(content)) > cfg.max_urls:
        return "muitos_links"
    return None


def _split_sentences(par: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", par) if s]


def make_chunks(paras: list[str], cfg: Config) -> list[str]:
    """Junta parágrafos inteiros em trechos de até `chunk_chars` (nunca corta frase)."""
    units: list[str] = []
    for par in paras:
        if len(par) <= cfg.chunk_chars:
            units.append(par)
        else:  # parágrafo gigante: quebra em frases
            units.extend(_split_sentences(par))

    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for u in units:
        if size and size + len(u) + 1 > cfg.chunk_chars:
            if size >= cfg.chunk_min_chars:
                chunks.append(" ".join(cur))
            cur, size = [], 0
        if len(u) > cfg.chunk_chars:  # frase maior que o chunk: fica inteira, sozinha
            chunks.append(u)
            continue
        cur.append(u)
        size += len(u) + 1
    if size >= cfg.chunk_min_chars:
        chunks.append(" ".join(cur))
    return chunks


def iter_docs(raw_dir: Path, cfg: Config) -> Iterator[tuple[Chunk, list[str]]]:
    """Itera docs do `split=train` (nunca `val`), round-robin entre as parts.

    Round-robin de row groups entre as parts (em ordem embaralhada pela seed) para que os
    primeiros exemplos já venham de páginas diferentes, e não de um único shard.
    """
    parts = sorted(raw_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"nenhum parquet em {raw_dir}")
    random.Random(cfg.seed).shuffle(parts)
    readers = [pq.ParquetFile(p) for p in parts]
    rgs: list[list[int]] = [list(range(f.num_row_groups)) for f in readers]
    while any(rgs):
        for i, f in enumerate(readers):
            if not rgs[i]:
                continue
            rg = rgs[i].pop(0)
            table = f.read_row_group(rg, columns=["id", "content", "url", "split", "dump"])
            for row in table.to_pylist():
                if row["split"] != "train":
                    continue
                content = row["content"] or ""
                if doc_reject_reason(content, cfg, row["url"] or "") is not None:
                    continue
                paras = clean_paragraphs(content, cfg.min_para_chars)
                for idx, text in enumerate(make_chunks(paras, cfg)[: cfg.max_per_doc]):
                    yield Chunk(row["id"], row["url"] or "", row["split"], row["dump"] or "",
                                idx, text), paras


# ---------------------------------------------------------------- professor


def pick_tipo(rng: random.Random, recusa_frac: float) -> str:
    if rng.random() < recusa_frac:
        return "recusa"
    tipos = list(TIPO_WEIGHTS)
    return rng.choices(tipos, weights=[TIPO_WEIGHTS[t] for t in tipos], k=1)[0]


def tipo_descricao(tipo: str, ancorado: bool) -> str:
    desc = TIPO_DESCRICAO[tipo]
    if tipo == "recusa":
        alvo = RECUSA_ANCORADO if ancorado else RECUSA_LIVRE
        desc += (f" Em 'resposta' escreva exatamente: {alvo.rstrip('.')}."
                 " Em 'evidencia' devolva string vazia.")
    return desc


def build_body(cfg: Config, chunk: str, tipo: str, ancorado: bool, seed: int) -> dict:
    user = (f"MODO: {MODO_DESCRICAO[ancorado]}\n"
            f"TIPO DE EXEMPLO: {tipo}\nTarefa: {tipo_descricao(tipo, ancorado)}\n\n"
            f"TRECHO:\n{chunk}")
    return {
        "model": cfg.teacher_model,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "presence_penalty": cfg.presence_penalty,
        "max_tokens": cfg.max_tokens,
        "seed": seed,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        # JSON garantido: o formato COMPLETO é obrigatório (json_object+schema é ignorado)
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "qa", "schema": QA_SCHEMA, "strict": True}},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def post_json(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ask_teacher(cfg: Config, chunk: str, tipo: str, ancorado: bool,
                seed: int) -> tuple[dict | None, dict, str]:
    """Uma chamada ao professor. Devolve (json, usage, erro)."""
    body = build_body(cfg, chunk, tipo, ancorado, seed)
    try:
        resp = post_json(cfg.teacher_url, body, cfg.timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, {}, f"http:{type(exc).__name__}"
    usage = resp.get("usage") or {}
    content = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    try:
        return json.loads(content), usage, ""
    except json.JSONDecodeError:
        return None, usage, "json_invalido"


# ---------------------------------------------------------------- validação


FONTE_RE = re.compile(r"(?i)\b(trecho|texto|documento|artigo|mat[ée]ria|fonte)\b")


def validate(qa: dict | None, chunk: str, tipo_pedido: str,
             ancorado: bool = True) -> tuple[str, str, str, str]:
    """Valida o par. Devolve (motivo_ou_vazio, pergunta, resposta, evidencia).

    `ancorado` muda duas regras: exemplos SEM contexto não podem citar a fonte na pergunta
    (ninguém teria o trecho em mãos) e a recusa tem outra frase.
    """
    if qa is None:
        return "json_invalido", "", "", ""
    status = qa.get("status", "ok")
    if status == "invalido":
        return "fonte_invalida", "", "", ""
    pergunta = " ".join(str(qa.get("pergunta", "")).split())
    resposta = " ".join(str(qa.get("resposta", "")).split())
    evidencia = str(qa.get("evidencia", "")).strip()
    tipo = qa.get("tipo", "")

    if tipo != tipo_pedido:
        return "tipo_diferente", pergunta, resposta, evidencia
    if not pergunta.endswith("?") or not (15 <= len(pergunta) <= 300):
        return "pergunta_ruim", pergunta, resposta, evidencia
    if not (10 <= len(resposta) <= 900):
        return "resposta_ruim", pergunta, resposta, evidencia
    if not is_pt(pergunta) or not is_pt(resposta):
        return "nao_pt", pergunta, resposta, evidencia
    if not ancorado and FONTE_RE.search(pergunta):
        return "menciona_fonte", pergunta, resposta, evidencia
    if normalize_ws(resposta).lower() in normalize_ws(pergunta).lower():
        return "resposta_copia_pergunta", pergunta, resposta, evidencia

    if tipo == "recusa":
        if not resposta.lower().startswith(RECUSA_PREFIXO[ancorado]):
            return "recusa_ruim", pergunta, resposta, evidencia
        if evidencia:
            return "recusa_com_evidencia", pergunta, resposta, evidencia
        return "", pergunta, resposta, ""

    # respondível: a evidência TEM que ser cópia literal do trecho
    ev = normalize_ws(evidencia).strip(" \"'“”‘’-–—")
    if len(ev) < 20:
        return "evidencia_vazia", pergunta, resposta, evidencia
    if ev not in normalize_ws(chunk):
        return "evidencia_nao_literal", pergunta, resposta, evidencia
    return "", pergunta, resposta, ev


# ---------------------------------------------------------------- orquestração


@dataclass
class Stats:
    accepted: int = 0
    rejected: int = 0
    chunks: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    reasons: dict = field(default_factory=dict)
    tipos: dict = field(default_factory=dict)
    modos: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    def bump(self, name: str) -> None:
        self.reasons[name] = self.reasons.get(name, 0) + 1


def load_done_keys(out_path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((rec.get("doc_id", ""), int(rec.get("chunk_idx", -1))))
    return done


@dataclass
class ChunkResult:
    records: list[dict]
    failures: list[str]
    completion_tokens: int
    prompt_tokens: int


def process_chunk(chunk: Chunk, cfg: Config, seen_questions: set[str],
                  lock: threading.Lock) -> ChunkResult:
    """Gera (e valida) os pares de um trecho. Nunca levanta: devolve o resultado + motivos."""
    rng = random.Random(f"{cfg.seed}:{chunk.doc_id}:{chunk.idx}")
    failures: list[str] = []
    completion_tokens = prompt_tokens = 0
    for attempt in range(cfg.tries):
        tipo = pick_tipo(rng, cfg.recusa_frac)
        ancorado = rng.random() < cfg.ancorado_frac
        seed = rng.randrange(2**31)
        qa, usage, err = ask_teacher(cfg, chunk.text, tipo, ancorado, seed)
        completion_tokens += usage.get("completion_tokens", 0)
        prompt_tokens += usage.get("prompt_tokens", 0)
        if err:
            failures.append(err)
            continue
        motivo, pergunta, resposta, evidencia = validate(qa, chunk.text, tipo, ancorado)
        if motivo:
            failures.append(motivo)
            continue
        key = normalize_ws(pergunta).lower()
        with lock:
            if key in seen_questions:
                failures.append("pergunta_duplicada")
                continue
            seen_questions.add(key)
        record = {
            "doc_id": chunk.doc_id, "url": chunk.url, "split": chunk.split,
            "dump": chunk.dump, "chunk_idx": chunk.idx, "chunk_chars": len(chunk.text),
            "chunk_text": chunk.text,
            "teacher": cfg.teacher_model, "prompt_version": PROMPT_VERSION,
            "seed": seed, "attempt": attempt, "tipo": qa.get("tipo", tipo),
            "ancorado": ancorado,
            "pergunta": pergunta, "resposta": resposta, "evidencia": evidencia,
        }
        return ChunkResult([record], failures, completion_tokens, prompt_tokens)
    return ChunkResult([], failures, completion_tokens, prompt_tokens)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "sft" / "sft-v1.jsonl")
    p.add_argument("--max-examples", type=int, default=300,
                   help="nº de pares aceitos antes de parar (piloto: 100–300)")
    p.add_argument("--resume", action="store_true", help="pula trechos já gravados no --out")
    p.add_argument("--dry-run", action="store_true", help="só mostra trechos (sem professor)")
    p.add_argument("--workers", type=int, default=8, help="chamadas simultâneas (slots=8)")
    p.add_argument("--tries", type=int, default=3, help="tentativas por trecho")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--ancorado-frac", type=float, default=0.5,
                   help="fração de exemplos COM o trecho junto (mini-RAG); o resto é sem contexto")
    p.add_argument("--recusa-frac", type=float, default=0.05)
    p.add_argument("--deny-url-regex", default=DENY_URL_DEFAULT)
    p.add_argument("--deny-text-regex", default=DENY_TEXT_DEFAULT)
    p.add_argument("--chunk-chars", type=int, default=2000)
    p.add_argument("--chunk-min-chars", type=int, default=700)
    p.add_argument("--max-per-doc", type=int, default=3)
    p.add_argument("--domain-cap-frac", type=float, default=0.05,
                   help="teto de exemplos por domínio = max(min, frac * --max-examples)")
    p.add_argument("--domain-cap-min", type=int, default=3)
    p.add_argument("--min-chars", type=int, default=2000)
    p.add_argument("--min-paras", type=int, default=4)
    p.add_argument("--max-digits-per-word", type=float, default=0.25)
    p.add_argument("--max-upper-ratio", type=float, default=0.30)
    p.add_argument("--max-urls", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--teacher-url", default=DEFAULT_TEACHER_URL)
    p.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    p.add_argument("--timeout", type=float, default=180.0)
    return p.parse_args(argv)


def cfg_from_args(args: argparse.Namespace) -> Config:
    return Config(
        min_chars=args.min_chars, min_paras=args.min_paras,
        max_digits_per_word=args.max_digits_per_word,
        max_upper_ratio=args.max_upper_ratio, max_urls=args.max_urls,
        chunk_chars=args.chunk_chars, chunk_min_chars=args.chunk_min_chars,
        max_per_doc=args.max_per_doc, domain_cap_frac=args.domain_cap_frac,
        domain_cap_min=args.domain_cap_min, ancorado_frac=args.ancorado_frac,
        recusa_frac=args.recusa_frac, deny_url_regex=args.deny_url_regex,
        deny_text_regex=args.deny_text_regex, tries=args.tries,
        workers=args.workers, max_tokens=args.max_tokens, timeout=args.timeout,
        teacher_model=args.teacher_model, teacher_url=args.teacher_url, seed=args.seed,
    )


def write_stats(path: Path, stats: Stats, args: argparse.Namespace) -> None:
    elapsed = max(time.time() - stats.started, 1e-6)
    payload = {
        "accepted": stats.accepted, "rejected_chunks": stats.rejected,
        "chunks": stats.chunks, "reasons": stats.reasons, "tipos": stats.tipos,
        "modos": stats.modos,
        "completion_tokens": stats.completion_tokens,
        "prompt_tokens": stats.prompt_tokens,
        "elapsed_s": round(elapsed, 1),
        "tok_s_agregado": round(stats.completion_tokens / elapsed, 1),
        "examples_min": round(stats.accepted * 60 / elapsed, 1),
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items() if k != "out"},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = cfg_from_args(args)
    if args.dry_run:
        shown = 0
        for chunk, _paras in iter_docs(args.raw_dir, cfg):
            print(f"--- doc {chunk.doc_id[:8]} chunk {chunk.idx} ({len(chunk.text)} chars)"
                  f" {chunk.url[:70]}")
            print(chunk.text[:600].replace("\n", " "))
            shown += 1
            if shown >= args.max_examples:
                break
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(args.out) if args.resume else set()
    if not args.resume and args.out.exists():
        sys.exit(f"{args.out} já existe — use --resume ou outro --out")
    stats = Stats()
    lock = threading.Lock()
    seen_questions: set[str] = set()
    domain_counts: dict[str, int] = {}
    domain_cap = max(cfg.domain_cap_min, int(cfg.domain_cap_frac * args.max_examples))
    if args.resume and args.out.exists():  # mantém os tetos ao continuar um run
        for line in args.out.open(encoding="utf-8"):
            try:
                dom = domain_of(json.loads(line)["url"])
            except (json.JSONDecodeError, KeyError):
                continue
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
    if done:
        print(f"resume: {len(done)} trechos já no arquivo, pulando")

    doc_iter = iter_docs(args.raw_dir, cfg)
    pending: dict[Future, Chunk] = {}
    bar = tqdm(total=args.max_examples, unit="par", desc="sft")
    out_fh = args.out.open("a" if args.resume else "w", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            exhausted = False
            while bar.n < args.max_examples:
                # mantém a fila cheia sem carregar o corpus na memória
                while not exhausted and len(pending) < cfg.workers * 3:
                    try:
                        chunk, _ = next(doc_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    if (chunk.doc_id, chunk.idx) in done:
                        continue
                    pending[pool.submit(process_chunk, chunk, cfg, seen_questions, lock)] = chunk
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    chunk = pending.pop(fut)
                    res = fut.result()
                    with lock:
                        stats.chunks += 1
                        stats.completion_tokens += res.completion_tokens
                        stats.prompt_tokens += res.prompt_tokens
                        for motivo in res.failures:
                            stats.bump(motivo)
                        if not res.records:
                            stats.rejected += 1
                        for rec in res.records:
                            if bar.n >= args.max_examples:
                                break
                            dom = domain_of(rec["url"])
                            if domain_counts.get(dom, 0) >= domain_cap:
                                stats.bump("dominio_no_teto")
                                continue
                            domain_counts[dom] = domain_counts.get(dom, 0) + 1
                            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            stats.accepted += 1
                            stats.tipos[rec["tipo"]] = stats.tipos.get(rec["tipo"], 0) + 1
                            modo = "ancorado" if rec["ancorado"] else "livre"
                            stats.modos[modo] = stats.modos.get(modo, 0) + 1
                            bar.update(1)
                        out_fh.flush()
                bar.set_postfix(rej=stats.rejected,
                                tok=stats.completion_tokens,
                                falhas=len(stats.reasons))
    except KeyboardInterrupt:
        print("\ninterrompido — o que já foi aceito está no arquivo")
    finally:
        bar.close()
        out_fh.close()
        write_stats(args.out.with_suffix(".stats.json"), stats, args)
    print(json.dumps({k: v for k, v in json.loads(
        args.out.with_suffix(".stats.json").read_text()).items() if k != "config"},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
