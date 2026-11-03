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

- 1 época de wiki = **1.240 steps** (650M de train ÷ 524.288 tokens/step; ~1,8 h na RTX, no
  mesmo batch do run atual);
- 2 épocas = 2.480 steps (~3,6 h) — e aí a comparação com o run OSCAR é "mesmo nº de steps,
  corpus 4× menor e limpo".

Dois experimentos valem a pena (mesmo vocab, `runs/pt-120m` intocado): **do zero** na wiki e
**continuando** o checkpoint do OSCAR na wiki (`--resume` copiado para `runs/pt-120m-wiki`).
Avaliação: ppl no `val-0000.bin` da wiki + a régua de sempre (`scripts/eval_sft.py`).
**Exige GPU** — o Strata segura 31,4 GB dos 32,6; combinar antes de disparar.

Disparados pelo `/treino` (fila de 1 job — um depois do outro), em 2026-11-03. O
`runs/pt-120m-wiki/ckpt.pt` é **cópia** do `runs/pt-120m/ckpt.pt` (step 5.248); o run (b) precisa
de `max_steps` explícito porque o default é 1 época (1.240 < 5.248 ⇒ o `while step < max_steps`
nem entraria). **`compile: true` não é enfeite**: sem a flag o run cai de 102k para 56k tok/s.

```bash
# (a) do zero: 1 época = 1.240 steps (650M ÷ 524.288 tokens-por-step), ~1,7 h
curl -s -X POST localhost:8010/api/treino/pretreino -H 'Content-Type: application/json' \
  -d '{"out":"wiki-120m","tokens_dir":"data/tokens/wiki-16k","epochs":1,"compile":true}'
# (b) continua o OSCAR na wiki: 5.248 + 1.240 = 6.488 steps, ~1,7 h
curl -s -X POST localhost:8010/api/treino/pretreino -H 'Content-Type: application/json' \
  -d '{"out":"pt-120m-wiki","retomar":true,"tokens_dir":"data/tokens/wiki-16k","max_steps":6488,"compile":true}'
```

Os dois com a receita do run OSCAR (bs 16 × ctx 2048 × accum 16 = 524.288 tokens/step, lr 3e-4
→ min-lr 3e-5, warmup 400, wd 0.1). No (b), no passo 5.248 esse cronograma está em ~5,9e-5 e
decai a 3e-5: adaptação, não re-treino.

### O que deu errado no caminho (e ficou consertado)

1. **Sem `--compile`**: 56k tok/s em vez de 102k (o ckpt do OSCAR tem `compile: True` — foi o
   primeiro lugar onde olhar). Relançado.
2. **A suíte de testes matou o run** no step 74: um teste chama `create_app()` sem `jobs=` e o
   `JobRunner` default aponta para o `runs/jobs.json` real, cujo `__init__` caça órfãos por
   SIGTERM. Duas travas agora (caça só no processo da webui + só em script do projeto) e os
   testes de WebUI injetam um runner de tmp. Ver [[Gotchas]].
3. **Log da UI vazio**: stdout do filho em pipe é block-buffered; `PYTHONUNBUFFERED=1` no
   `JobRunner` resolve. Depois disso, o `/treino` ganhou **barra de progresso com ETA**
   ([[Interface de treino]]).

## Resultado do run (a) — do zero na wiki (2026-11-03)

1.240 steps, 650.117.120 tokens (1,0 época), **6.071 s ≈ 1 h 41** a 101k tok/s com `--compile`.
Val loss no val da wiki (`val-0000.bin`, 13,1M tokens), eval do próprio treino a cada 250 steps:

| step | val loss | ppl |
|---|---|---|
| 250 | 4,631 | 102,6 |
| 500 | 3,531 | 34,2 |
| 750 | 3,058 | 21,3 |
| **1.000** | **2,878** | **17,78** |

O `final/` nasce quando o run conclui (`config.json` + `model.safetensors`).

**A comparação que importa** (mesmo harness `train.evaluate`, bs 16 × ctx 2048): o modelo do
OSCAR medido no val da wiki, no arranque do run (b), dá **3,0759 / 21,7**. O run do zero, com
**650M tokens limpos** (4× menos), chega a **17,78** — 18% melhor no domínio da wiki. A
hipótese do corpus limpo se sustenta: o problema do Pingo não era falta de tokens, era o
OSCAR.

## Baseline cruzado (2026-11-03, CPU, antes de qualquer re-treino)

O Pingo-120M do OSCAR (`runs/pt-120m/best.pt`, step 4.872) avaliado no MESMO harness
(`train.evaluate`, batch 4 × ctx 1024 × 8 iters, rng fixo) nos dois val:

| val | val loss | ppl |
|---|---|---|
| OSCAR (`data/tokens`, 6 shards) | 2,621 | **13,75** |
| **wiki (`data/tokens/wiki-16k`)** | 3,207 | **24,72** |

O estilo enciclopédico é **fora de distribuição** para o modelo do OSCAR — 1,8× pior ppl no
corpus que agora queremos explorar. (Não comparar com o 11,67 do relatório do run: lá é
ctx 2048 / mais iters; aqui a comparação é entre corpora, mesmo harness.) É o número-alvo do
**wiki-4**: o run da wiki deve derrubar esse 24,72, e o `--resume` deve fazê-lo rápido.

## Ver também

- [[Filtro de qualidade]] · [[QA do dataset com LLM]] · [[Interface de treino]] · [[Gotchas]]
