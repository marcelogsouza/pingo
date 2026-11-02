"""Testes do QA de dataset (treino-2) — o portão do /treino, em CPU e sem dados reais.

O que não pode quebrar: o contrato (JSONL com `content`) é aprovado/reprovado com
motivos legíveis; duplicatas usam o MESMO hash do `process_data.py` (senão a UI aprova
o que o filtro joga fora); e o relatório é JSON-safe para o front.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("pingo_qa_dataset", ROOT / "scripts" / "qa_dataset.py")
qa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qa)


def _jsonl(tmp_path: Path, linhas: list) -> Path:
    p = tmp_path / "d.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for l in linhas:
            fh.write((l + "\n") if isinstance(l, str)
                     else json.dumps(l, ensure_ascii=False) + "\n")
    return p


DOC = "O governo anunciou nesta terça-feira um pacote de investimentos em infraestrutura. " * 10


def test_dataset_bom_e_aprovado(tmp_path):
    p = _jsonl(tmp_path, [{"content": DOC}, {"content": DOC + " fim", "url": "x"},
                          {"content": "Terceiro documento com conteúdo razoável. " * 12}])
    r = qa.analisar(p)
    assert r["aprovado"] is True
    assert r["linhas"] == 3 and r["content_ok"] == 3 and r["problemas"] == []
    assert r["comprimento"]["min"] > 300  # as réguas de comprimento vêm do process_data


def test_content_ausente_reprova_e_sugere_o_campo_candidato(tmp_path):
    p = _jsonl(tmp_path, [{"text": DOC, "titulo": "x"}, {"content": DOC}])
    r = qa.analisar(p)
    assert r["aprovado"] is False
    assert any("content" in p for p in r["problemas"])
    assert r["detalhe"]["campo_candidato"] == "text"
    assert r["detalhe"]["content_ausente"][0]["linha"] == 1


def test_json_quebrado_linha_vazia_e_content_ruim_aparecem_com_motivo(tmp_path):
    p = _jsonl(tmp_path, ["{isso não é json", "", {"content": DOC},
                          {"content": "   "}, {"content": 42}, "[1,2,3]"])
    r = qa.analisar(p)
    assert r["aprovado"] is False
    assert r["content_ok"] == 1
    assert r["detalhe"]["vazias"] == 1
    assert r["detalhe"]["json_erros"] and r["detalhe"]["json_erros"][0]["linha"] == 1
    assert r["detalhe"]["content_vazio"] == 1
    assert r["detalhe"]["content_nao_string"] == 1
    assert any("JSON" in p for p in r["problemas"])


def test_duplicata_e_aviso_nao_problema_usa_o_hash_do_process_data(tmp_path):
    p = _jsonl(tmp_path, [{"content": DOC},
                          {"content": DOC},                       # exata
                          {"content": "  ".join(DOC.split())},    # whitespace colapsado = a mesma
                          {"content": "Outro documento de tamanho suficiente para não ser curto. " * 9}])
    r = qa.analisar(p)
    assert r["duplicatas"] == 2 and r["content_ok"] == 4
    assert r["aprovado"] is True                                  # duplicata é aviso, não reprovação
    assert any("repetido" in a for a in r["avisos"])


def test_dado_curto_ou_simbolico_gera_aviso_mas_aprova(tmp_path):
    p = _jsonl(tmp_path, [{"content": "curto demais"},
                          {"content": "#%&%#%&" * 80},
                          {"content": "a\n" * 200}])
    r = qa.analisar(p)
    assert r["aprovado"] is True and len(r["avisos"]) >= 3


def test_amostras_sao_sorteadas_e_o_relatorio_e_json_seguro(tmp_path):
    p = _jsonl(tmp_path, [{"content": f"Documento número {i}. " + DOC} for i in range(100)])
    r = qa.analisar(p, sample=5)
    assert len(r["amostras"]) == 5
    assert all(a["linha"] <= 100 and a["chars"] > 300 for a in r["amostras"])
    json.dumps(r, ensure_ascii=False)                             # a UI consome cru


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        qa.analisar(tmp_path / "nao-existe.jsonl")
