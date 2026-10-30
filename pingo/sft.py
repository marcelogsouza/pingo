"""Primitivas compartilhadas do dataset de SFT.

`scripts/gen_sft.py` (gera os pares com o professor) e `scripts/validate_sft.py` (revalida,
deduplica e separa train/val/heldout) precisam concordar sobre o que é *evidência literal*,
*recusa*, *texto em português* e *exemplo válido*. Se cada script tivesse a sua cópia, o
validador poderia aprovar o que o gerador rejeitou (ou vice-versa) — por isso as regras
moram aqui.

Nada aqui depende de GPU, rede ou do professor: dá para testar tudo em CPU.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------- taxonomia

TIPOS = ("extracao", "explicacao", "resumo", "recusa")
TIPO_WEIGHTS = {"extracao": 0.50, "explicacao": 0.30, "resumo": 0.15}

# recusa: a frase depende de o exemplo levar o trecho junto (ancorado) ou não
RECUSA_ANCORADO = "Isso não consta no texto."
RECUSA_LIVRE = "Não tenho informações suficientes para responder isso."
RECUSA_PREFIXO = {True: "isso não consta", False: "não tenho informa"}

# pergunta de exemplo SEM contexto não pode citar a fonte (ninguém teria o trecho em mãos)
FONTE_RE = re.compile(r"(?i)\b(trecho|texto|documento|artigo|mat[ée]ria|fonte)\b")


# ---------------------------------------------------------------- texto


PT_STOP = {
    "a", "o", "as", "os", "um", "uma", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "por", "para", "com", "que", "se", "não", "e", "é", "foi", "são", "ao",
    "aos", "à", "às", "como", "mais", "mas", "ou", "quando", "qual", "quais", "seu", "sua",
    "seus", "suas", "ele", "ela", "eles", "elas", "isso", "este", "esta", "esse", "essa",
    "entre", "sobre", "depois", "antes", "também", "já", "há", "ser", "tem", "tinha", "ter",
    # pronomes/interrogativos: segura frase curta sem acento ("Quanto custa o plano?")
    "quanto", "quantos", "quantas", "quem", "onde", "porque", "porquê", "cujo", "cuja",
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
    import urllib.parse

    netloc = urllib.parse.urlparse(url).netloc.lower()
    netloc = netloc.split("@")[-1].split(":")[0]
    return netloc[4:] if netloc.startswith("www.") else netloc


def evidence_in_chunk(evidencia: str, chunk: str) -> bool:
    """A evidência é cópia literal do trecho? (comparação tolerante só a whitespace/aspas)"""
    ev = normalize_ws(evidencia).strip(" \"'“”‘’-–—")
    return len(ev) >= 20 and ev in normalize_ws(chunk)


# ---------------------------------------------------------------- contrato do exemplo

REQUIRED_FIELDS = ("doc_id", "url", "split", "chunk_idx", "chunk_text", "tipo", "ancorado",
                   "pergunta", "resposta", "evidencia")


def example_reject_reason(rec: dict, min_chunk: int = 200, max_chunk: int = 6000) -> str:
    """Revalida um exemplo já gravado. Devolve o motivo da rejeição ou "" se está ok.

    Repete de propósito as checagens do gerador: o dataset é artefato de longo prazo e o
    validador não deve confiar em quem escreveu o JSONL.
    """
    if not isinstance(rec, dict):
        return "nao_e_objeto"
    faltando = [k for k in REQUIRED_FIELDS if k not in rec]
    if faltando:
        return "campo_ausente:" + ",".join(faltando)
    if rec["split"] != "train":
        return "split_nao_train"  # o val do pré-treino não pode vazar para o SFT
    if not (isinstance(rec["doc_id"], str) and rec["doc_id"]):
        return "doc_id_ruim"
    chunk = rec["chunk_text"] or ""
    if not (min_chunk <= len(chunk) <= max_chunk):
        return "chunk_tamanho"
    pergunta = (rec["pergunta"] or "").strip()
    resposta = (rec["resposta"] or "").strip()
    tipo = rec["tipo"]
    ancorado = bool(rec["ancorado"])
    if tipo not in TIPOS:
        return "tipo_invalido"
    if not pergunta.endswith("?") or not (15 <= len(pergunta) <= 300):
        return "pergunta_ruim"
    if not (10 <= len(resposta) <= 900):
        return "resposta_ruim"
    if not is_pt(pergunta) or not is_pt(resposta):
        return "nao_pt"
    if not ancorado and FONTE_RE.search(pergunta):
        return "sem_contexto_cita_fonte"
    if tipo == "recusa":
        if not resposta.lower().startswith(RECUSA_PREFIXO[ancorado]):
            return "recusa_ruim"
        if (rec["evidencia"] or "").strip():
            return "recusa_com_evidencia"
        return ""
    if not evidence_in_chunk(rec["evidencia"] or "", chunk):
        return "evidencia_nao_literal"
    if normalize_ws(resposta).lower() in normalize_ws(pergunta).lower():
        return "resposta_copia_pergunta"
    return ""
