# Plano — Pingo: LLM do zero em pt-BR

## Visão

Construir uma LLM decoder-only do zero (arquitetura própria em PyTorch), treinada em português,
usando Hugging Face Transformers apenas como *scaffolding* (`PreTrainedModel`/`PretrainedConfig`
para save/load/Hub) e `uv` para todo o gerenciamento do projeto.
Fase 1 = completor de texto (pré-treino). Mid-training (SFT) e RLHF ficam para depois.

## Decisões já tomadas

| Item | Decisão |
|---|---|
| Idioma | Português (pt-BR) |
| Arquitetura | **Própria**, decoder-only: MHA + **RoPE**, pré-norm, sem classes prontas (GPT2/Llama) |
| Infra de treino | RTX PRO 4500 Blackwell 32 GB (dev/testes na M4 local via CPU/MPS) |
| Porte inicial | **Pingo-120M** (ciclo curto, treinos interrompíveis); escalar p/ 350M+ depois |
| Gerenciador | `uv` (pyproject + `uv run`) |
| KV cache | Desde o primeiro dia, no módulo de atenção + loop de geração próprio |

## Arquitetura (Pingo v0)

Decoder-only, estilo moderno e simples:
- Token embedding + **tied embeddings** (sem embedding de posição — RoPE na atenção)
- **Pré-norm com RMSNorm** em cada bloco
- **Multi-head attention** causal (head_dim 64), scores via `F.scaled_dot_product_attention`
- **RoPE** (base θ=10000) em Q e K
- MLP: **SwiGLU** (hidden ≈ 8/3·d_model)
- Sem viés nas lineares (padrão atual); init `normal(0, 0.02)` com escala `1/√(2L)` nas residuais
- Contexto: **2048** posições (suficiente para texto; RoPE permite extensão futura)
- **KV cache**: classe `PingoCache` com tensors pré-alocados `[B, H, max_seq, D]` por camada,
  escrita incremental; `generate()` próprio aproveitando o cache

Configs-alvo (vocab 16k, head_dim 64, ctx 2048):

| Config | Camadas | d_model | Heads | SwiGLU hidden | Params ≈ | Tokens | Noite única? (RTX 4500) |
|---|---|---|---|---|---|---|---|
| **Pingo-120M (default, Fase 1)** | 12 | 896 | 14 | 2304 | ~125M | ~2.5B | ✔ ~9–12 h |
| Pingo-100M (alternativa p/ debug) | 12 | 768 | 12 | 2048 | ~100M | ~2B | ✔ ~5 h |
| Pingo-350M (Fase 2, escalar depois) | 24 | 1024 | 16 | 2816 | ~335M | ~6B | ✘ ~3 dias |

**Restrição do usuário:** sem run contínuo de 3 dias → Fase 1 roda em **janelas de ~1 noite**,
com checkpoints a cada ~15 min e `--resume` obrigatório e bem testado (treino pode ser fatiado
em várias noites, retomando otimizador + scheduler + RNG).

## Dataset de pré-treino (pt)

**Principal:** `oscar-corpus/community-oscar` — substitui o OSCAR-2301, que foi **suspenso com
revisão manual** (não libera mais). O Community OSCAR tem gate **auto** (aceitamos os termos
uma vez e o acesso é imediato), mesmo esquema de anotação do OSCAR 23.01 e 45 dumps do
Common Crawl (2014–2024).

- Estrutura no repo: `data/<dump>/pt/pt_part_NNN.jsonl.zst` (~630 MB por shard, JSONL com
  `content` + metadata). Shards independentes → download paralelizável e retomável.
- **Baixar só os dumps mais recentes (2–3)**, parando quando atingirmos o orçamento
  (~40–50 GB baixados, sobrando ~2.5B tokens após filtros). Não precisamos do corpus todo.
- **Sem filtro pt-BR vs pt-PT** (decisão: o pt-BR já domina ~90% do conteúdo web em
  português; não vamos complicar).
- Fallbacks conhecidos se precisar: `eduagarcia/cc_news_pt_v2` (sem gate, notícias),
  `statmt/cc100` `pt` (sem gate), Wikipedia pt (complemento de qualidade).

Plano de consumo: script faz download dos shards `.jsonl.zst` → extrai campo `content` →
filtro de qualidade local (comprimento de doc, ratio de linhas curtas, ratio de símbolos,
dedup exato por hash de doc) → parquet em `data/raw/`.

## Estrutura do projeto

```
pingo/
├── pyproject.toml            # uv; deps: torch, transformers, tokenizers, datasets, numpy, tqdm, pyarrow
├── data/                     # raw (parquet) + tokens (.bin uint16 + meta.json) — fora do git
├── tokenizer/                # tokenizer.json treinado por nós + vocab stats
├── pingo/                    # pacote python
│   ├── config.py             # PingoConfig(PretrainedConfig)
│   ├── model.py              # RMSNorm, RoPE, Attention(+KV cache), MLP, PingoLM(PreTrainedModel)
│   ├── cache.py              # PingoCache
│   ├── generate.py           # sampling: temperature, top-k, top-p, greedy
│   └── data.py               # DataLoader memmap sobre shards .bin
├── scripts/
│   ├── download_data.py      # OSCAR pt (+ wiki) → data/raw
│   ├── train_tokenizer.py    # BPE byte-level 16k
│   ├── prepare_tokens.py     # parquet → shards binários train/val
│   ├── train.py              # loop de pré-treino
│   └── sample.py             # geração + REPL interativo
└── tests/                    # shapes, contagem de params, equivalência do KV cache, overfit
```

## Passo a passo (cada passo entregável e verificável)

1. **Setup uv** — `uv init`, deps, estrutura, pytest. ✔ Critério: `uv run python -c "import torch; print(torch.cuda.is_available())"` ok nas duas máquinas.
2. **Dados** — aceitar termos do Community OSCAR (gate auto) 1x + `HF_TOKEN`; baixar dumps recentes `pt` → `data/raw/*.parquet`; segurar ~2% p/ validação. Meta da Fase 1: **~2.5–3B tokens brutos**. ✔ Critério: amostras inspecionadas, contagem de docs/tokens registrada.
3. **Tokenizer** — BPE byte-level **16k** treinado em amostra (~1 GB do corpus pt), tokens especiais `<|bos|> <|eos|> <|pad|>`. ✔ Critério: taxa de compressão (chars/token) registrada e comparada com GPT-2 no texto pt.
4. **Binarização** — tokenizer aplicado a tudo → shards `.bin` (uint16) de ~1M tokens + `val.bin`. ✔ Critério: loader memmap lê batches corretamente.
5. **Modelo** — implementação + testes unitários:
   - contagem de params bate com o alvo;
   - **teste de equivalência do KV cache**: `max|logits_full − logits_cached| < 1e-5` (fp32);
   - teste de overfit: batch pequeno → loss ≈ 0. ✔ Critério: tudo verde em CPU/MPS com config tiny.
6. **Pré-treino** — `train.py`: AdamW (wd 0.1, sem decay em norm), warmup+cosine, grad clip 1.0,
   autocast bf16, **batch global ~0.5M tokens** (grad accumulation; ~5k steps p/ 2.5B tokens),
   checkpoints a cada ~15 min (modelo + otimizador + scheduler + RNG + passo) e `--resume`,
   log JSONL (+ tensorboard opcional), seed fixa. Dry-run de ~30 min na M4 (config tiny) →
   run completo na RTX (pode ser fatiado em mais de uma noite). ✔ Critério: val loss/pplx
   caindo de forma estável; resume reproduz exatamente a mesma curva ao retomar.
7. **Geração e avaliação** — perplexidade no val; amostras qualitativas; `save_pretrained` +
   tokenizer → repo HF opcional. ✔ Critério: texto pt coerente em parágrafos; geração 10×+ mais rápida com KV cache vs sem (benchmark simples).

## Próximos passos (fora do escopo da Fase 1)

8. **Escalar o porte** — repetir o pipeline com **Pingo-350M** (já na tabela de configs) e mais tokens.
9. **Mid-training / SFT** — dataset de instruções pt (ex.: datasets pt-BR de instruções no HF).
10. **RLHF** — começar com **DPO** (dados de preferência pt-BR já existem, ex.: `Hub-Ai/ptbr-human-preferences`); PPO + reward model como opção avançada.
11. **Avaliação formal** — benchmarks pt (ex.: tarefas estilo NeurIPS-BR-LP, hellaswag-pt).

## Decisões fechadas ✔

- Porte Fase 1: **Pingo-120M** (12 camadas, d=896, 14 heads) — treina em ~6–10 h, escalável a 350M depois
- MLP: **SwiGLU** · Vocab: **16k** · Contexto: **2048** · Dataset: **Community OSCAR `pt`** (gate auto, sem filtro de variante)

## Últimos detalhes p/ fechar antes do Passo 1

- [ ] Preparar os dados (OSCAR + tokenizer + binários) **na M4** e levar só os `.bin` p/ a RTX? (recomendado: M4 baixa/tokeniza, RTX só treina)
- [ ] Onde ficam os checkpoints na RTX (disco local) e se queremos subir o modelo final ao HF Hub (repo privado?)
