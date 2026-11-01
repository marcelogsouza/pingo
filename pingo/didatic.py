"""Modo didático (passos 1–2) — do texto aos ids, ao vetor de cada id e à tabela inteira.

O curso do Pingo precisa mostrar o que o modelo *vê* antes de qualquer conta: o BPE
byte-level fatiando a frase em pedaços (com o id de cada um), o vetor que cada id carrega
na tabela de embeddings, e a tabela inteira de uma vez — normas, histograma e o produto
escalar que um `lm_head` tied leria de volta daquele vetor (passo ②).

Duas decisões estruturam o módulo:

1. **Nada de GPU.** A tabela de embeddings é lida do `model.safetensors` direto na CPU
   (58 MB). O laboratório funciona com o professor servindo na VRAM e em qualquer máquina.
2. **Superfície ≠ texto.** Num BPE byte-level o token é uma sequência de *bytes*: `ĠatÃ©`
   é o que o vocab tem, ` até` é o que o humano lê. As duas aparecem na tela — é ali que o
   byte-level costuma fazer sentido. Os `offsets` do `encode()` são posições de **caractere**
   do texto original, então é por eles que se recupera o texto humano (nunca por
   `decode([id])`, que produz `\ufffd` em byte incompleto).

Tudo aqui é puro e testeiro em CPU: nada importa Flask, nada toca em checkpoint alheio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ limites

MAX_LAB_CHARS = 4_000      # teto do texto que o navegador pode mandar
MAX_LAB_TOKENS = 512       # teto de chips desenhados (acima disso, avisa e corta)
VIZINHOS_PADRAO = 8        # k da lista "tokens vizinhos"
BUSCA_PADRAO = 20          # limites do lookup no vocab
FAIXA_BINS = 128           # colunas da linha da matriz (896 = 7 × 128; funciona p/ qualquer dim)
MATRIZ_LINHAS = 96         # teto de linhas da matriz — tela nenhuma lê 500 linhas de uma vez
LIDA_PADRAO = 8            # k da "tabela lida de volta" (passo ②)

# Nomes que são especiais por construção no projeto (além dos `added_tokens` do backend).
# No fixture de teste os tokens de chat entram como vocab comum — daí a lista explícita.
NOMES_ESPECIAIS = ("<|bos|>", "<|eos|>", "<|pad|>", "<|user|>",
                   "<|assistant|>", "<|end|>", "<unk>")

CATEGORIAS = ("especial", "linha", "bytes", "pontuação", "número", "início", "continuação")


# ------------------------------------------------------------------ byte-level

def _monta_alfabeto() -> dict[int, str]:
    """Alfabeto do ByteLevel (o mesmo do GPT-2 e do `tokenizers`): byte → caractere.

    Bytes imprimíveis ficam com o próprio código; os demais sobem para a região de
    `U+0100`, que nunca colide com texto real. É assim que `Ġ` é espaço e `Ã§` são os dois
    bytes UTF-8 de `ç`.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_PARA_CHAR = _monta_alfabeto()
_CHAR_PARA_BYTE = {c: bytes([b]) for b, c in BYTE_PARA_CHAR.items()}


def superficie_para_bytes(superficie: str) -> bytes | None:
    """A superfície do token (o que o BPE reporta) → os bytes que ela representa.

    `None` se algum caractere não pertence ao alfabeto byte-level (não acontece na saída
    de um BPE byte-level; acontece em tokens especiais como `<|user|>`? não — esses são ASCII).
    """
    try:
        return b"".join(_CHAR_PARA_BYTE[c] for c in superficie)
    except KeyError:
        return None


def bytes_para_superficie(bruto: bytes) -> str:
    """O inverso: `b" priced"` → `ĠpreÃ§o`. É como o vocab escreve o token."""
    return "".join(BYTE_PARA_CHAR[b] for b in bruto)


def texto_legivel(superficie: str) -> str:
    """A superfície byte-level → o que um humano lê (`ĠpreÃ§o` → ` preço`).

    Byte no meio de um caractere UTF-8 vira `U+FFFD`: é honesto, é exatamente o que o
    token isolado é (um pedaço de palavra).
    """
    bruto = superficie_para_bytes(superficie)
    if bruto is None:
        return superficie
    return bruto.decode("utf-8", "replace")


# ------------------------------------------------------------------ categorias

def especial_ids(tokenizer) -> set[int]:
    """Ids especiais: os `added_tokens` do backend + os nomes conhecidos do projeto."""
    ids = set(tokenizer.get_added_tokens_decoder().keys())
    for nome in NOMES_ESPECIAIS:
        tid = tokenizer.token_to_id(nome)
        if tid is not None:
            ids.add(tid)
    return ids


def categoria(superficie: str, texto: str, especial: bool = False) -> str:
    """Classe do token, do mais específico para o mais geral (é o que guia a cor na tela).

    - `especial`  — `<|user|>`, `<|eos|>`… (o que o SFT ensinou, não é texto)
    - `linha`     — carrega o byte 0x0A (quebra de linha)
    - `bytes`     — **um único byte** ≥ 0x80: o byte-fallback puro (o vocab não tem o caractere)
    - `pontuação` — só símbolos, sem letra nem dígito
    - `número`    — dígitos sem letras (`202`, `1,5` viram vários desses)
    - `início`    — começa com espaço byte-level (`ĠQuanto`): começa palavra
    - `continuação` — o resto (`gunt`, `Ã§Ã£o`)
    """
    if especial:
        return "especial"
    bruto = superficie_para_bytes(superficie)
    if bruto is None:
        bruto = superficie.encode("utf-8", "replace")
    if b"\n" in bruto:
        return "linha"
    if len(bruto) == 1 and bruto[0] >= 0x80:
        return "bytes"
    nu = texto.strip()
    if nu and not any(c.isalnum() for c in nu):
        return "pontuação"
    if any(c.isdigit() for c in nu) and not any(c.isalpha() for c in nu):
        return "número"
    if bruto[:1] == b" ":
        return "início"
    return "continuação"


# ------------------------------------------------------------------ tokenização

def tokenizar(tokenizer, prompt: str, limite: int = MAX_LAB_TOKENS) -> dict[str, Any]:
    """Tokeniza `prompt` (o texto que o modelo de fato recebe) para a tela do curso.

    Devolve o contrato JSON que o front-end desenha. Por token:

    - `texto`      — fatia do prompt pelos `offsets` (o humano lê ` até`)
    - `superficie` — o que o vocab guarda (`ĠatÃ©`)
    - `bytes`      — os bytes dessa superfície, em hex (`20 61 74 c3 a9`)
    - `kind`       — a categoria (ver `categoria`)
    - `acentuado`  — tem byte ≥ 0x80 na superfície (o custo do português acentuado)
    - `repetido`   — índice da primeira vez que esse mesmo id apareceu (ou None)
    """
    enc = tokenizer.encode(prompt, add_special_tokens=False)
    especiais = especial_ids(tokenizer)
    cortado_em = len(enc.ids) if len(enc.ids) > limite else None

    tokens: list[dict[str, Any]] = []
    visto: dict[int, int] = {}
    por_categoria = {c: 0 for c in CATEGORIAS}
    acentuados, n_bytes = 0, 0
    for i, (superficie, tok_id, (ini, fim)) in enumerate(
            zip(enc.tokens, enc.ids, enc.offsets)):
        if i >= limite:
            break
        texto = prompt[ini:fim]
        bruto = superficie_para_bytes(superficie)
        hexbytes = " ".join(f"{b:02x}" for b in bruto) if bruto is not None else ""
        acentuado = bool(bruto) and any(b >= 0x80 for b in bruto)
        kind = categoria(superficie, texto, tok_id in especiais)
        por_categoria[kind] += 1
        acentuados += int(acentuado)
        n_bytes += len(bruto) if bruto else len(texto.encode("utf-8"))
        primeira = visto.setdefault(tok_id, i)   # o índice da primeira vez que este id apareceu
        tokens.append({
            "i": i,
            "id": tok_id,
            "texto": texto,
            "superficie": superficie,
            "bytes": hexbytes,
            "offset": [ini, fim],
            "kind": kind,
            "acentuado": acentuado,
            "repetido": primeira if primeira != i else None,
        })

    n = len(tokens)
    chars = len(prompt)
    return {
        "tokens": tokens,
        "resumo": {
            "chars": chars,
            "bytes": n_bytes,
            "n_tokens": n,
            "chars_por_token": round(chars / n, 2) if n else 0.0,
            "por_categoria": por_categoria,
            "acentuados": acentuados,
            "especiais": por_categoria["especial"],
            "repetidos": n - len({t["id"] for t in tokens}),
            "distintos": len({t["id"] for t in tokens}),
            "vocab": int(tokenizer.get_vocab_size()),
            "cortado_em": cortado_em,
        },
    }


# ------------------------------------------------------------------ embeddings


@dataclass
class TabelaEmbeddings:
    """A tabela `embed.weight` na CPU + o que a tela precisa dela (normas e cossenos).

    É a mesma matriz que o `lm_head` usa (tied embeddings) — por isso o passo 1 do curso
    já pode anunciar que o vetor do token é metade do modelo inteiro.
    """

    pesos: torch.Tensor        # [V, D] fp32, CPU
    unit: torch.Tensor         # pesos por unidade (para cosseno em 1 matmul)
    normas: torch.Tensor       # [V]
    id_para_token: dict[int, str]
    chave: str = "embed.weight"

    @classmethod
    def do_checkpoint(cls, pasta: Path) -> "TabelaEmbeddings":
        """Lê só a tabela de embeddings do `model.safetensors` — sem modelo, sem GPU."""
        from safetensors import safe_open

        arq = Path(pasta) / "model.safetensors"
        if not arq.exists():
            raise FileNotFoundError(f"não achei {arq}")
        with safe_open(str(arq), framework="pt") as f:
            if cls.chave not in f.keys():
                raise KeyError(f"{cls.chave} não está em {arq} (tem {len(f.keys())} tensores)")
            pesos = f.get_tensor(cls.chave).to("cpu", torch.float32)
        if pesos.dim() != 2:
            raise ValueError(f"{cls.chave} não é [vocab, dim]: {tuple(pesos.shape)}")
        return cls(pesos=pesos, unit=F.normalize(pesos, dim=1),
                   normas=pesos.norm(dim=1), id_para_token={})

    def vocabulario(self, tokenizer) -> "TabelaEmbeddings":
        """Indexa `id → token` a partir do tokenizer (chamado uma vez por carga)."""
        self.id_para_token = {i: t for t, i in tokenizer.get_vocab().items()}
        return self

    def token(self, tok_id: int) -> str:
        return self.id_para_token.get(tok_id, f"<id {tok_id}>")

    def ficha(self, tok_id: int, k: int = VIZINHOS_PADRAO) -> dict[str, Any]:
        """O vetor do token + estatísticas + os vizinhos mais próximos por cosseno.

        Os vizinhos são o ouro didático do passo 1: antes de qualquer camada, o vetor de
        `Ġinvestir` já mora perto de `Ġapostar` e `Ġgastar` — a semântia está na geometria.
        """
        if not 0 <= tok_id < self.pesos.shape[0]:
            raise IndexError(f"id {tok_id} fora do vocab (0..{self.pesos.shape[0] - 1})")
        v = self.pesos[tok_id]
        sims = self.unit @ self.unit[tok_id]
        k = max(1, min(int(k), 50))
        topk = torch.topk(sims, min(k + 1, sims.numel()))
        vizinhos = [{"id": int(j), "token": self.token(int(j)),
                     "texto": texto_legivel(self.token(int(j))),
                     "cos": round(float(s), 4), "norma": round(float(self.normas[int(j)]), 3)}
                    for s, j in zip(topk.values.tolist(), topk.indices.tolist())
                    if int(j) != tok_id][:k]
        valores = [round(float(x), 4) for x in v.tolist()]
        maior = max(range(len(valores)), key=lambda j: abs(valores[j]))
        bruto = superficie_para_bytes(self.token(tok_id))
        return {
            "id": int(tok_id),
            "token": self.token(tok_id),
            "texto": texto_legivel(self.token(tok_id)),
            "bytes": " ".join(f"{b:02x}" for b in bruto) if bruto else "",
            "dim": int(self.pesos.shape[1]),
            "valores": valores,
            "norma": round(float(self.normas[tok_id]), 4),
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4),
            "std": round(float(v.std()), 5),
            "escala": round(float(self.pesos.abs().max()), 4),   # escala fixa p/ todas as fichas
            "norma_media": round(float(self.normas.mean()), 4),
            "dim_mais_ativa": {"j": maior, "valor": valores[maior]},
            "vizinhos": vizinhos,
        }

    # -- passo ②: a frase como matriz, a tabela inteira, e a tabela lida de volta --

    def faixa(self, tok_id: int, bins: int = FAIXA_BINS) -> list[float]:
        """O vetor resumido a `bins` valores — a linha desenhada da matriz.

        Sobrevive em cada grupo o valor de **maior módulo**, não a média: o ponto da
        matriz são os picos; a média os apagaria e toda linha sairia cinza. (Em 896 os
        grupos de 7 dividem exato, mas o código vale para qualquer `dim`.)
        """
        v = self.pesos[tok_id]
        n = int(v.numel())
        b = max(1, min(int(bins), n))
        bordas = [i * n // b for i in range(b + 1)]
        saida = []
        for i in range(b):
            pedaco = v[bordas[i]:bordas[i + 1]]
            j = int(pedaco.abs().argmax())
            saida.append(round(float(pedaco[j]), 4))
        return saida

    def matriz(self, ids: list[int], bins: int = FAIXA_BINS) -> dict[str, Any]:
        """A frase como matriz de números: uma linha por token — a imagem central do ②.

        As linhas param em `MATRIZ_LINHAS` (`cortado_em` avisa com o total real): um
        prompt de laboratório tem ~40 tokens, e o aluno não leria 500 linhas de cores.
        """
        b = max(1, min(int(bins), int(self.pesos.shape[1])))
        linhas = []
        for tok_id in ids[:MATRIZ_LINHAS]:
            if not 0 <= tok_id < self.pesos.shape[0]:
                raise IndexError(f"id {tok_id} fora do vocab (0..{self.pesos.shape[0] - 1})")
            v = self.pesos[tok_id]
            linhas.append({"id": int(tok_id), "token": self.token(int(tok_id)),
                           "texto": texto_legivel(self.token(int(tok_id))),
                           "norma": round(float(self.normas[tok_id]), 3),
                           "std": round(float(v.std()), 5),
                           "faixa": self.faixa(tok_id, b)})
        return {"dim": int(self.pesos.shape[1]), "bins": b,
                "escala": round(float(self.pesos.abs().max()), 4),
                "cortado_em": len(ids) if len(ids) > MATRIZ_LINHAS else None,
                "linhas": linhas}

    def estatisticas(self, bins: int = 48, extremos: int = 6) -> dict[str, Any]:
        """A tabela inteira de uma vez: a distribuição das normas e seus extremos.

        O histograma mostra que norma é tudo, menos uniforme; os extremos dão nome aos
        números (o byte raro de maior norma, o token de chat mal treinado de menor).
        """
        n = self.normas
        maior = float(n.max())
        hist = torch.histc(n, bins=bins, min=0.0, max=maior)
        bordas = torch.linspace(0.0, maior, bins + 1)

        def quem(i: int) -> dict[str, Any]:
            return {"id": int(i), "token": self.token(int(i)),
                    "texto": texto_legivel(self.token(int(i))),
                    "norma": round(float(n[int(i)]), 3)}

        k = max(1, min(int(extremos), int(n.numel())))
        topo = torch.topk(n, k)
        base = torch.topk(n, k, largest=False)
        return {
            "vocab": int(n.numel()), "dim": int(self.pesos.shape[1]),
            "norma": {"media": round(float(n.mean()), 4),
                      "mediana": round(float(torch.quantile(n, 0.5)), 4),
                      "min": round(float(n.min()), 4), "max": round(maior, 4),
                      "std": round(float(n.std()), 4)},
            "histograma": [{"desde": round(float(bordas[i]), 3),
                            "ate": round(float(bordas[i + 1]), 3), "cont": int(hist[i])}
                           for i in range(bins)],
            "maiores": [quem(int(i)) for i in topo.indices.tolist()],
            "menores": [quem(int(i)) for i in base.indices.tolist()],
            "norma_vetor_medio": round(float(self.pesos.mean(dim=0).norm()), 4),
        }

    def lida_de_volta(self, tok_id: int, k: int = LIDA_PADRAO) -> dict[str, Any]:
        """O que um `lm_head` tied leria deste vetor: nota = W·v para o vocab inteiro.

        Os logits *reais* só nascem depois das 12 camadas (passo ⑤) — aqui é o vetor
        **inicial** multiplicado pela própria tabela, e serve para uma lição que a tela
        mostra sozinha: a nota do lm_head é produto escalar, **inclui a norma** do
        vizinho; o cosseno é só direção. As duas listas vêm ordenadas de forma
        diferente — é exatamente essa a diferença que o aluno precisa ver.
        """
        if not 0 <= tok_id < self.pesos.shape[0]:
            raise IndexError(f"id {tok_id} fora do vocab (0..{self.pesos.shape[0] - 1})")
        k = max(1, min(int(k), 50))
        logits = self.pesos @ self.pesos[tok_id]     # [V] — a nota antes de qualquer softmax
        sims = self.unit @ self.unit[tok_id]         # cosseno com todo o vocab

        def quem(i: int) -> dict[str, Any]:
            return {"id": int(i), "token": self.token(int(i)),
                    "texto": texto_legivel(self.token(int(i))),
                    "pontos": round(float(logits[int(i)]), 4),
                    "cos": round(float(sims[int(i)]), 4),
                    "norma": round(float(self.normas[int(i)]), 3)}

        top_dot = torch.topk(logits, min(k, logits.numel()))
        por_pontos = [quem(int(j)) for j in top_dot.indices.tolist()]
        top_cos = torch.topk(sims, min(k + 1, sims.numel()))
        por_cos = [quem(int(j)) for _, j in zip(top_cos.values.tolist(), top_cos.indices.tolist())
                   if int(j) != tok_id][:k]
        return {"id": int(tok_id), "token": self.token(int(tok_id)),
                "texto": texto_legivel(self.token(int(tok_id))),
                "norma": round(float(self.normas[tok_id]), 4),
                "pontos_consigo": round(float(logits[tok_id]), 4),
                "por_pontos": por_pontos, "por_cos": por_cos}


# ------------------------------------------------------------------ busca no vocab


def buscar(tokenizer, consulta: str, limite: int = BUSCA_PADRAO) -> list[dict[str, Any]]:
    """`q` é um id (número) ou um trecho de texto → o que existe no vocab com aquilo.

    A consulta é comparada nas três formas — o texto humano, a superfície byte-level
    (`ção` → `Ã§Ã£o`) e o id — porque é assim que o aluno navega no vocab: às vezes ele tem
    a palavra, às vezes o número, às vezes o desenho esquisito que a tela mostrou.
    """
    consulta = (consulta or "").strip()
    if not consulta:
        return []
    achados: list[dict[str, Any]] = []
    if consulta.isdigit():
        tok = tokenizer.id_to_token(int(consulta))
        if tok is not None:
            achados.append(_registro(int(consulta), tok))
    alvo = consulta.lower()
    alvo_na_superficie = bytes_para_superficie(consulta.encode("utf-8")).lower()
    exatos, contem = [], []
    for tok, tid in tokenizer.get_vocab().items():
        superficie, legivel = tok.lower(), texto_legivel(tok).lower()
        if superficie == alvo or legivel == alvo:
            exatos.append((tok, tid))
        elif alvo in superficie or alvo in legivel or alvo_na_superficie in superficie:
            contem.append((tok, tid))
    for tok, tid in sorted(exatos, key=lambda x: x[1]) + sorted(contem, key=lambda x: x[1])[:limite]:
        achados.append(_registro(tid, tok))
        if len(achados) >= limite:
            break
    return achados


def _registro(tid: int, superficie: str) -> dict[str, Any]:
    bruto = superficie_para_bytes(superficie) or b""
    return {"id": tid, "token": superficie, "texto": texto_legivel(superficie),
            "bytes": " ".join(f"{b:02x}" for b in bruto)}
