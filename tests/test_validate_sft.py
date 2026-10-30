"""Testes do SFT-2 (`scripts/validate_sft.py`) e das regras compartilhadas (`pingo/sft.py`).

O que precisa ser verdade para o dataset valer alguma coisa:

- a evidência é cópia literal do trecho (senão o professor inventou);
- recusa vem na frase certa do modo (ancorado x sem contexto);
- quase-duplicata de pergunta/trecho é barrada (MinHash + LSH), mas texto só *parecido*
  não é;
- o split é **por documento** — vazamento de doc entre train/val/heldout é erro fatal.
"""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("pingo_validate_sft",
                                               ROOT / "scripts" / "validate_sft.py")
v = importlib.util.module_from_spec(_spec)
sys.modules["pingo_validate_sft"] = v
_spec.loader.exec_module(v)

from pingo.sft import (  # noqa: E402
    RECUSA_ANCORADO, RECUSA_LIVRE, evidence_in_chunk, example_reject_reason, is_pt,
)

CHUNK = ("A Companhia Energética de Minas Gerais anunciou um plano de investimentos de 12 "
         "bilhões de reais até 2028, com foco em redes de distribuição no interior do estado. "
         "O programa prevê a contratação de 3 mil eletricistas em regime de tempo integral.")
EVID = "um plano de investimentos de 12 bilhões de reais até 2028"


def rec(**over) -> dict:
    base = {
        "doc_id": "doc1", "url": "https://exemplo.com.br/a", "split": "train",
        "chunk_idx": 0, "chunk_text": CHUNK, "chunk_chars": len(CHUNK),
        "teacher": "qwen", "prompt_version": "v2", "seed": 1, "attempt": 0,
        "tipo": "extracao", "ancorado": True,
        "pergunta": "Quanto a companhia planeja investir até 2028?",
        "resposta": "A companhia planeja investir 12 bilhões de reais até 2028.",
        "evidencia": EVID,
    }
    base.update(over)
    return base


# ------------------------------------------------------------------ pingo/sft


def test_evidence_in_chunk_tolera_so_whitespace():
    assert evidence_in_chunk(EVID, CHUNK)
    assert evidence_in_chunk(EVID.replace(" ", "\n  "), CHUNK)
    assert evidence_in_chunk(f'"{EVID}"', CHUNK)  # aspas de citação
    assert not evidence_in_chunk("15 bilhões de reais até 2030", CHUNK)


def test_is_pt_aceita_frase_curta_legitima():
    assert is_pt("Quanto custa o plano?")
    assert not is_pt("Where is the investment plan for the company?")


@pytest.mark.parametrize("over,motivo", [
    ({}, ""),
    ({"split": "val"}, "split_nao_train"),
    ({"pergunta": "sem interrogação"}, "pergunta_ruim"),
    ({"resposta": "curta"}, "resposta_ruim"),
    ({"evidencia": "texto que não está no trecho em lugar nenhum"}, "evidencia_nao_literal"),
    ({"tipo": "outro"}, "tipo_invalido"),
    ({"ancorado": False, "pergunta": "Segundo o texto, quanto será investido até 2028?"},
     "sem_contexto_cita_fonte"),
    ({"chunk_text": "curto"}, "chunk_tamanho"),
    ({"pergunta": "What is the investment plan of this company?"}, "nao_pt"),
])
def test_example_reject_reason(over, motivo):
    assert example_reject_reason(rec(**over)) == motivo


def test_recusa_por_modo_no_contrato():
    anc = rec(tipo="recusa", resposta=RECUSA_ANCORADO, evidencia="")
    assert example_reject_reason(anc) == ""
    assert example_reject_reason({**anc, "ancorado": False}) == "recusa_ruim"
    livre = rec(tipo="recusa", resposta=RECUSA_LIVRE, evidencia="", ancorado=False,
                pergunta="Quem dirigiu o filme indicado ao Oscar em 1998?")
    assert example_reject_reason(livre) == ""
    assert example_reject_reason({**livre, "evidencia": EVID}) == "recusa_com_evidencia"


def test_campo_ausente():
    r = rec()
    del r["chunk_text"]
    assert example_reject_reason(r) == "campo_ausente:chunk_text"


# ------------------------------------------------------------------ MinHash / LSH


def test_shingles_e_jaccard():
    a = v.shingles("o gato subiu no telhado de zinco ontem à noite")
    b = v.shingles("o gato subiu no telhado de zinco ontem à noite")
    c = v.shingles("a economia brasileira cresceu no trimestre passado")
    assert v.jaccard(a, b) == 1.0
    assert v.jaccard(a, c) == 0.0


def test_lsh_encontra_quase_duplicata_de_pergunta():
    d = v.NearDedup(q_threshold=0.8, chunk_threshold=0.85)
    assert d.is_duplicate("Qual foi o valor do plano de investimentos da Cemig até 2028?",
                          CHUNK) == ""
    # mesma pergunta com 1 palavra trocada → quase-duplicata
    assert d.is_duplicate("Qual foi o valor do plano de investimentos da Cemig até 2029?",
                          "outro trecho completamente diferente sobre energia solar") == "pergunta"
    # pergunta diferente e trecho diferente → aceita
    assert d.is_duplicate("Quem é o presidente da Argentina desde 2023?",
                          "A inflação acumulada no país vizinho superou as expectativas "
                          "do mercado e o banco central elevou a taxa básica de juros.") == ""


def test_lsh_barra_trecho_republicado():
    """Site espelho: trecho de ~500 palavras com 2 palavras trocadas (como na vida real)."""
    longo = " ".join([
        "A companhia anunciou um plano de investimentos bilionários para o período.",
        CHUNK,
        "O diretor de operações afirmou que a maior parte dos recursos vai para a rede.",
        "O programa também prevê treinamento de equipes e modernização de subestações.",
    ] * 6)
    espelho = longo.replace("interior", "litoral").replace("3 mil", "3.500")
    d = v.NearDedup(q_threshold=0.9, chunk_threshold=0.85)
    assert d.is_duplicate("Quanto a companhia vai investir segundo o plano?", longo) == ""
    assert d.is_duplicate("Qual é o número de eletricistas previsto no programa?",
                          espelho) == "trecho"


def test_lsh_nao_indexa_rejeitados_no_meio():
    """Um duplicado rejeitado não pode deslocar os índices das assinaturas guardadas."""
    d = v.NearDedup(q_threshold=0.8, chunk_threshold=0.85)
    pergunta = "Quantos eletricistas serão contratados pelo programa da companhia mineira?"
    assert d.is_duplicate(pergunta, CHUNK) == ""
    assert d.is_duplicate(pergunta, "texto de outro documento bem diferente daqui") == "pergunta"
    # nova pergunta comparada contra a lista real: não pode estourar IndexError
    assert d.is_duplicate("Qual é a capital da França segundo o texto mais antigo?",
                          "Paris é a capital e a maior cidade da França.") == ""


# ------------------------------------------------------------------ split e ponta a ponta


def test_split_por_documento_e_deterministico():
    a = [v.split_of(f"doc{i}", 13, 0.05, 0.05) for i in range(2000)]
    b = [v.split_of(f"doc{i}", 13, 0.05, 0.05) for i in range(2000)]
    assert a == b
    frac = a.count("train") / len(a)
    assert 0.85 < frac < 0.95  # ~90% no treino
    assert set(a) == {"train", "val", "heldout"}


def _escreve_jsonl(path: Path, recs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _dataset(n_docs: int = 40, por_doc: int = 2) -> list[dict]:
    """Docs sintéticos de assuntos distintos (senão o dedup os barra, e com razão)."""
    rng = random.Random(0)
    vocabulario = [f"tema{i}" for i in range(400)]
    recs = []
    for d in range(n_docs):
        for c in range(por_doc):
            unicas = rng.sample(vocabulario, 40)
            texto = CHUNK + " " + " ".join(unicas)
            recs.append(rec(
                doc_id=f"doc{d}", chunk_idx=c,
                pergunta=f"Como o assunto {' '.join(unicas[:3])} afeta o plano de 2028?",
                chunk_text=texto,
                evidencia=" ".join(unicas[:6]),
            ))
    return recs


def test_main_ponta_a_ponta(tmp_path):
    inp = tmp_path / "cru.jsonl"
    recs = _dataset()
    recs.append(rec(doc_id="ruim", evidencia="inventado fora do trecho por completo"))
    recs.append(rec(doc_id="doc0", chunk_idx=0))         # repetição exata (artefato de --resume)
    _escreve_jsonl(inp, recs)
    with inp.open("a", encoding="utf-8") as fh:
        fh.write('{"quebrado": ')                       # linha truncada (arquivo em escrita)

    out = tmp_path / "final"
    v.main(["--in", str(inp), "--out-dir", str(out), "--seed", "13"])

    stats = json.loads((out / "stats.json").read_text())
    assert stats["linhas_lidas"] == len(recs) + 1
    assert stats["motivos"]["json_invalido"] == 1
    assert stats["motivos"]["evidencia_nao_literal"] == 1
    assert stats["motivos"]["duplicado_exato"] == 1
    assert stats["vazamento_docs"] == {"train_val": 0, "train_heldout": 0, "val_heldout": 0}

    docs = {}
    total = 0
    for nome in ("train", "val", "heldout"):
        linhas = [json.loads(x) for x in (out / f"{nome}.jsonl").read_text().splitlines()]
        total += len(linhas)
        for r in linhas:
            docs.setdefault(r["doc_id"], set()).add(nome)
    assert total == stats["aceitos"] == len(recs) - 2  # fora: 1 ruim + 1 duplicado
    assert all(len(s) == 1 for s in docs.values())  # nenhum doc em dois splits
    assert stats["splits"]["train"]["exemplos"] > stats["splits"]["val"]["exemplos"]
