---
tags: [dados, wiki]
status: pronto
data: 2026-11-03
---

# Dataset Wikipedia PT (Passo 14b — o "corpus limpo" para re-treinar)

Depois de medir a [[QA do dataset com LLM|raspa do OSCAR]] (~45% dos docs aprovados com marca
de web), abrimos um caminho paralelo: **re-treinar sobre Wikipedia pt** e comparar. Não
substitui o OSCAR ainda — é o experimento de qualidade, com o OSCAR de volta depois.

## Fonte

- [`TucanoBR/wikipedia-PT`](https://huggingface.co/datasets/TucanoBR/wikipedia-PT) (TucanoBR,
  nov/2024): **1.103.446 artigos**, campo único `text`, licença **cc-by-sa-3.0** (attribution
  devida em model card). Texto já extraído (sem HTML), ~2,59 GB.
- Já está em cache arrow em `~/.cache/huggingface/datasets/TucanoBR___wikipedia-pt/` — não
  precisa baixar de novo.

## Ingest — `scripts/ingest_wikipedia.py`

```bash
uv run python scripts/ingest_wikipedia.py            # → data/wiki/pt.jsonl + pt-val.jsonl + pt.meta.json
uv run python scripts/ingest_wikipedia.py --limit 5000 --out data/wiki/pt-smoke.jsonl
```

Reaproveita **as MESMAS réguas** do filtro OSCAR (`process_data.py`: `SHORT_LINE_CHARS`,
`SYMBOLS`, `norm_key` blake2b) e o MESMO split por hash (`% 100 < val_pct`). Converte o campo
`text` → `content`, que é o contrato da QA do `/treino` e do `prepare_tokens --input jsonl`.

> **O `<out>.jsonl` já é o lado train** (exclui os docs do val, que vão para `<out>-val.jsonl`).
> Binarizar re-fatiando esse arquivo por hash deixa o val vazio — ver [[Gotchas]] (aceite do
> passo: `prepare_tokens` detecta o irmão `-val.jsonl` e aborta se o val vier vazio).

### A régua de linhas curtas mudou: 0.9, não 0.6

Medido em amostra de 4.000 artigos (`SHORT_LINE_CHARS` = 30):

| ratio | retenção |
|---|---|
| 0,6 (régua OSCAR) | **33,0%** |
| 0,7 | 67,9% |
| 0,8 | 85,6% |
| **0,9 (default do ingest)** | **97,0%** |
| 0,95 | 99,1% |

Motivo: na wiki, **título de seção é linha curta legítima** ("História", "Ver também", "Referências").
Amostra de reprovados aos 0,6: 61–68% das linhas curtas, artigos perfeitamente bons.

Consequência operacional: o `qa_dataset.py` tem 0,6 embutido e acusa **492.254** documentos com
"maioria de linhas curtas" no corpus wiki — falso positivo. Ele agora aceita
`--short-line-ratio 0.9` (o `/treino` continua com 0,6).

## Contas do corpus

| | docs | bytes |
|---|---|---|
| vistos no HF | 1.103.446 | ~2,59 GB |
| **aprovados** | **853.359** (77,3%) | |
| ├ train (`pt.jsonl`) | 836.266 | 2.439.019.969 |
| └ val (`pt-val.jsonl`) | 17.093 (2,00%) | 49.202.401 (1,98% dos chars) |
| descarte `len` (<300 / >100k chars) | 226.094 | |
| descarte `short_lines` (régua 0,9) | 21.986 | |
| descarte `symbols` | 1.945 | |
| duplicatas exatas (blake2b) | 62 | |

QA (`scripts/qa_dataset.py --short-line-ratio 0.9`): **APROVADO** nos dois lados — 0 duplicatas,
0 linhas fora do contrato; chars/doc p50 1.196 · p90 6.193 · max 99.892 · média 2.790.

## Tokens — `data/tokens/wiki-16k/`

```bash
uv run python scripts/prepare_tokens.py --input jsonl --jsonl data/wiki/pt.jsonl \
    --tokenizer tokenizer --out data/tokens/wiki-16k --val-pct 2 --n-process 8
#   val: data/wiki/pt-val.jsonl (split já feito pelo ingest)
```

**Tokenizer default congelado (16.384)** — mesmo vocab do `tokenizer/tokenizer.json`, então os
*embeddings* e o checkpoint do `runs/pt-120m` são **compatíveis**: dá para continuar o treino
OSCAR na wiki, além de partir do zero.

- `train-0000..0650.bin` — **650.248.473 tokens** (651 shards de 1M)
- `val-0000.bin` — **13.117.734 tokens**
- 3,75 chars/token nos dois lados (conta coerente)

## O que isso muda na economia do treino

O run do Pingo-120M consumiu **2,75B tokens em 5.248 steps** (524.288 tokens/step). A wiki
inteira tem **663M tokens = 24% disso**:

- 1 época de wiki = **1.265 steps** (~1,8 h na RTX, no mesmo batch do run atual);
- 2 épocas = 2.530 steps (~3,6 h) — e aí a comparação com o run OSCAR é "mesmo nº de steps,
  corpus 4× menor e limpo".

Dois experimentos valem a pena (mesmo vocab, `runs/pt-120m` intocado): **do zero** na wiki e
**continuando** o checkpoint do OSCAR na wiki (`--resume` copiado para `runs/pt-120m-wiki`).
Avaliação: ppl no `val-0000.bin` da wiki + a régua de sempre (`scripts/eval_sft.py`).
**Exige GPU** — o Strata segura 31,4 GB dos 32,6; combinar antes de disparar.

## Ver também

- [[Filtro de qualidade]] · [[QA do dataset com LLM]] · [[Interface de treino]] · [[Gotchas]]
