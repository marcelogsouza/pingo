---
tags: [pingo, referencia, comandos]
atualizado: 2026-10-30
---

# Comandos e cheatsheet

Tudo via `uv run`. Ordem = ordem do projeto.

## Sanidade, sempre

```bash
uv run pytest -q                                    # 103 testes, ~13 s
uv run python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv   # antes de qualquer CUDA
```

## Etapa 2 — dados

```bash
uv run hf auth login                       # 1x; aceitar o gate do dataset na página do Hub
./scripts/download_data.sh                 # 100 parts, ~42,7 GB (Windows: .bat)
uv run python scripts/process_data.py      # → data/raw/*.parquet + stats.json
```

Só um teste rápido: `uv run python scripts/process_data.py --limit-shards 2`

## Etapa 3 — tokenizer

```bash
uv run python scripts/train_tokenizer.py                                  # 16k, 1 GB de amostra
uv run python scripts/train_tokenizer.py --vocab 2000 --sample-chars 50_000_000   # smoke
```

## Etapa 4 — tokens

```bash
uv run python scripts/prepare_tokens.py                                   # 2,75B train + val todo
uv run python scripts/prepare_tokens.py --train-tokens 10_000_000 --out data/tokens_smoke
wc -l data/prepare_tokens.log && cat data/tokens/meta.json
```

## Etapas 5–6 — modelo e pré-treino

```bash
# dry-run em CPU (config tiny)
uv run python scripts/train.py --tiny --ctx 256 --max-steps 20 --device cpu --out runs/dryrun-cpu

# run real (GPU!) — cuidado: sobrescreve runs/pt-120m/ se não houver --resume
uv run python scripts/train.py --compile
uv run python scripts/train.py --compile --resume          # retomar
```

## Etapa 7 — avaliação do base

```bash
uv run python scripts/sample.py --out runs/pt-120m/eval.json
uv run python scripts/sample.py --no-bench --max-new-tokens 60            # só ppl + amostras
uv run python scripts/sample.py --prompt "Era uma vez" --temperature 0.9 --top-p 0.95
uv run python scripts/sample.py --model runs/smoke-120m/final --no-ppl    # comparar runs
```

## SFT — pipeline completo

```bash
# 0 · o professor tem de estar no ar (o USUÁRIO sobe; não subimos)
curl -s localhost:8001/v1/models

# SFT-1 · gerar (retomável; NUNCA dois escritores no mesmo arquivo)
uv run python scripts/gen_sft.py --max-examples 20 --dry-run              # só mostra trechos
uv run python scripts/gen_sft.py --max-examples 300 --out data/sft/pilot.jsonl
uv run python scripts/gen_sft.py --max-examples 30000 --out data/sft/sft-v1.jsonl --resume
wc -l data/sft/sft-v1.jsonl                                               # confira aqui, NÃO no tqdm

# SFT-2 · validar + dedup + splits (+ juiz de recusas, ~2 min)
uv run python scripts/validate_sft.py --judge-recusas
cat data/sft/final/stats.json

# SFT-3 · treinar (GPU)
uv run python scripts/train_sft.py --tiny --max-steps 20 --out /tmp/sft-tiny --device cpu
uv run python scripts/train_sft.py                                        # run real
uv run python scripts/train_sft.py --resume

# SFT-4 · avaliar
uv run python scripts/eval_sft.py --n 200 --judge
cat runs/sft-120m/final/eval_sft.json

# amostras do SFT (resolve o tokenizer de 16.387 sozinho)
uv run python scripts/sample.py --model runs/sft-120m/final --no-bench
```

## Comparação de professores (opcional, custa centavos)

```bash
export OPENROUTER_API_KEY=...
uv run python scripts/pilot_teacher_compare.py --n 10 \
  --models qwen/qwen3.7-flash deepseek/deepseek-v4.1-flash google/gemma-4-26b-a4b-it
```

## Tabela de argumentos que mais se usam

| Script | Arg | Efeito |
|---|---|---|
| `process_data.py` | `--min-chars 300` | tamanho mínimo do doc |
| | `--val-pct 2` | % de docs para val (por hash) |
| `train_tokenizer.py` | `--vocab 16384` | tamanho do vocabulário |
| | `--sample-chars 1e9` | tamanho da amostra de treino |
| `prepare_tokens.py` | `--train-tokens` | meta de tokens |
| | `--shard-tokens 1_000_000` | tokens por `.bin` |
| `train.py` | `--batch-size/--accum-steps` | batch global |
| | `--max-steps` | default: 1 época |
| | `--compile` | ~2× e menos VRAM |
| | `--resume` | retoma de `ckpt.pt` |
| `sample.py` | `--repetition-penalty 1.15` | mata o loop do greedy |
| | `--bench-contexts 512,1024` | contextos do bench |
| `gen_sft.py` | `--ancorado-frac 0.5` | mistura ancorado/livre |
| | `--recusa-frac 0.05` | fração de recusas |
| | `--tries 3` | tentativas por trecho |
| | `--workers 8` | chamadas simultâneas (= slots) |
| `validate_sft.py` | `--judge-recusas` | segunda opinião nas recusas |
| | `--val-frac/--heldout-frac 0.05` | tamanho dos splits |
| `train_sft.py` | `--epochs 3` | épocas |
| | `--max-len 1024` | tokens por exemplo |
| `eval_sft.py` | `--judge` | veredito do professor |
| | `--n 200` | quantos exemplos |

## Onde ficam os artefatos

| Caminho | Conteúdo |
|---|---|
| `data/2024-10/pt/*.jsonl.zst` | dump cru (40 GB, apagável) |
| `data/raw/*.parquet` | texto filtrado + `stats.json` |
| `tokenizer/` | `tokenizer.json` (16.384) + `vocab_stats.json` |
| `data/tokens/` | `train-*.bin`, `val-*.bin`, `meta.json` |
| `data/sft/` | JSONL do gerador + `final/{train,val,heldout}` |
| `runs/pt-120m/` | pré-treino (ckpt, best, final, log, eval) |
| `runs/pt-120m-BACKUP-2026-10-30/` | **backup read-only — não apagar** |
| `runs/sft-120m/` | SFT (ckpt, best, final + tokenizer 16.387) |

## Ver também

- [[Gotchas]] — o que dá errado e como perceber
- [[Suíte de testes]] — a rede de segurança
- [[Pingo]] — voltar ao índice
