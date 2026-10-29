# AGENTS.md — estado do projeto

Ao iniciar uma sessão, leia isto primeiro. O plano completo está em [`PLAN.md`](PLAN.md).

> **Regra de ouro: atualize este arquivo SEMPRE que fizer alguma coisa** — terminar um
> passo, meio passo, um bugfix, uma decisão nova ou um gotcha descoberto. Commit junto
> (mesmo commit ou commit separado `docs: estado`), para a próxima sessão (ou pessoa)
> retomar exatamente de onde parou. Mantenha a seção "Onde estamos" honesta: estado real
> dos arquivos/commands, não o estado desejado.

## Onde estamos

**Passo 6 de 7 — Pré-treino: PRÓXIMO.** (Passo 5 concluído em 2026-10-29)

Feito:
- ✅ Passo 1 (setup uv + estrutura): deps instaladas, pytest verde, CUDA ok na RTX PRO 4500.
- ✅ **Passo 2 (dados)**: 100/100 parts de `2024-10` baixadas (`data/2024-10/pt/`, 42,7 GB);
  `process_data.py` rodado completo → 19,3M docs → **10,7M aprovados (55,6%)** com
  1,9M dups exatos removidos; **59,7B chars ≈ ~15B tokens** (folga de 6× sobre a meta de
  2,5–3B → dump reserva `2023-40` NÃO será necessário). Parquet: 23 GB em `data/raw/`
  (colunas `id, content, url, dump, part, split`, ~2% `val`); stats em `data/raw/stats.json`.
  Amostras inspecionadas: pt real (ocasional pt-PT, aceitável por decisão do plano).
- ✅ **Passo 3 (tokenizer)**: `scripts/train_tokenizer.py` → BPE byte-level **vocab 16.384**
  (regex GPT-2 + `byte_fallback=True`), treinado em **1,0 GB** (179.619 docs, `split=train`,
  39 s, pico 3 GB RAM), especiais `<|bos|> <|eos|> <|pad|>` (ids 0–2). Salvo em `tokenizer/`
  (tokenizer.json + tokenizer_config.json). **Critério de aceite atingido: 4,09 chars/token
  no val (5,0M chars) vs 2,69 do GPT-2 → 34,3% menos tokens** (métricas em
  `tokenizer/vocab_stats.json`; top-20 tokens = function words pt, saudável).
  `tests/test_tokenizer.py` cobre especiais, roundtrip, byte-fallback e compressão pt.
- ✅ **Passo 4 (binarização)**: `scripts/prepare_tokens.py` + `pingo/data.py`. Tokens
  **train: 2,751B** (2.015.381 docs amostrados do `split=train`, seed 13, 12,8 min) em
  **2.752 shards** `train-NNNN.bin` de 1M tokens uint16 (docs separados por `<|eos|>` id=1) +
  **val: 292M** tokens (o `split=val` COMPLETO, 215.042 docs, 6 shards `val-NNNN.bin`).
  `meta.json` com contagens; loader memmap `pingo/data.py` (`load_tokens`/`get_batch`,
  alvo = entrada deslocada de 1). `tests/test_data.py` (5 testes, incl. integração com os
  shards reais) verde. Decisão cumprida: dos ~15B tokens do corpus, amostramos a meta 2,5–3B.
- ✅ **Passo 5 (modelo)**: `pingo/{config,model,cache,generate}.py` + `tests/test_model.py`
  (16 testes; suíte completa **26 verdes** em CPU, config tiny). Implementado: RMSNorm
  pré-norm (estatística em fp32), RoPE θ=10k clássico em Q/K (`inv_freq` on-the-fly, sem
  buffer), MHA causal via `F.scaled_dot_product_attention`, SwiGLU, lineares sem viés,
  init `normal(0,0.02)` + escala `1/√(2L)` em `o_proj`/`down_proj` (flag `_residual_proj`),
  tied embeddings, `PingoLM(PreTrainedModel)` com `_tied_weights_keys` (HF v5).
  `PingoCache`: buffers pré-alocados `[B,H,max_seq,D]` por camada alocados no 1º update,
  escrita em `cur_len` + `commit()` no fim do forward; pós-prefill só step de 1 token
  (ValueError claro). `generate()`: greedy/temperature/top-k/top-p, batch=1, para no ctx.
  **Critérios atingidos:** params exatos **127.555.456** (== alvo ~127,6M); equivalência
  KV cache `< 1e-5` fp32 (prefill-idêntico e prefill-parcial+steps token-a-token); overfit
  8×24 → loss < 0.05; roundtrip `save_pretrained`/`from_pretrained` bit-exato nos pesos.
  Sanidade no Pingo-120M real em CPU fp32: forward B2×1024 ok (1,8 s, loss 8,14 ≈ ln V);
  benchmark GPU de geração fica p/ Passo 7 (GPU tomada pelo Strata hoje).

A fazer no **Passo 6 (pré-treino)** — `scripts/train.py`:
1. AdamW wd 0.1 **sem decay em RMSNorm/embedding**; warmup linear + cosine p/ lr final;
   grad clip 1.0; autocast **bf16**; seed fixa.
2. Batch global **~0.5M tokens** (micro-batch B×2048 na RTX + grad accumulation) →
   ~5.000–5.500 steps p/ 2,75B tokens. Medir tok/s e ajustar B.
3. Checkpoints a cada ~15 min: modelo + otimizador + scheduler + RNG (torch/py/np) +
   passo global; `--resume` reproduzindo a curva (testar: 2 runs fatiados == 1 contínuo).
4. Log JSONL (loss, val loss/pplx periódico, lr, tok/s; tensorboard opcional).
5. Dry-run ~30 min (config tiny ou poucos steps) → run completo na RTX (janela de noite,
   retomável; **parar o Strata antes**, ele toma os 32 GB de VRAM).
6. Marcar `[x] 6.` no roadmap; commit `step 6: ...`.

## Próximos passos (do PLAN.md)

| Passo | O que | Arquivos | Critério |
|---|---|---|---|
| **6** | Pré-treino: AdamW wd 0.1, warmup+cosine, clip 1.0, bf16, ~0.5M tok/batch global, ckpt 15 min + `--resume`, seed fixa | `scripts/train.py` | val loss estável; resume reproduz a curva |
| **7** | Geração + perplexidade val + `save_pretrained` | `scripts/sample.py` | texto pt coerente; 10× mais rápido c/ KV cache |

Config default: **Pingo-120M** (12 camadas, d_model 896, 14 heads, SwiGLU hidden 2304, ctx 2048).

## Convenções

- **Tudo via `uv`**: `uv run ...`, `uv add ...`. Não usar pip/venv direto.
- Commits: mensagem `step N: ...` (ver `git log`); **push só após revisão do usuário**.
- `data/`, `tokenizer/`, `*.parquet`, `*.bin` são gitignored; `uv.lock` é versionado.
- Máquinas: dev/testes na M4 local (CPU/MPS, config tiny); treino na RTX PRO 4500 32 GB.
  **Os dados (`data/`, ~69 GB: raw + parquet + `tokens/`) e o `tokenizer/` estão nesta
  máquina RTX** (ambos gitignored → na M4 os testes de integração pulam). Nela o login HF
  já está feito (`marcelogsouza`) e o gate do Community OSCAR já foi aceito — download não
  precisa de setup.
- **Antes de usar a GPU (treino, bench, sanity em CUDA): AVISAR o usuário primeiro** —
  ela roda o servidor de inferência Strata (~31 GB dos 32 GB de VRAM) e é ele que precisa
  ser desligado. Nunca tomar a GPU por conta própria; conferir com `nvidia-smi`.
- Testes antes de fechar qualquer passo: `uv run pytest`.

## Gotchas já descobertos

- O idioma `pt` só existe nos dumps **`2024-10`** e **`2023-40`** do Community OSCAR; os
  dumps mais recentes (2024-18+) só publicam `pt_meta`. Não insistir neles.
- Dataset é *gated* (gate auto): precisa `uv run hf auth login` + aceitar os termos na
  página do dataset 1x. Sem token, o CDN responde 401.
- Paths do repo HF já começam com `data/` → cuidado com caminho duplicado ao compor.
- `Path("x.jsonl.zst").stem` = `x.jsonl` (só um sufixo); usar `name.removesuffix(".jsonl.zst")`.
- HF hub 2.x CLI: `uv run hf auth whoami` (não é `hf whoami`).
- `pt_part_N` não tem tamanho uniforme: muitas parts têm ~10 MB (não ~427 MB); validar por
  tamanho real, nunca assumir média. Taxa de dup exacto no corpus: ~10%.
- Disco: `data/` ocupa ~69 GB (40 `.jsonl.zst` + 23 parquet + 6,1 tokens). Os parquets e os
  `.bin` estão conferidos (Passos 2–4 rodados e testados) → **apagar os `.jsonl.zst` já é
  seguro** (tudo está tokenizado em `data/tokens/`; e são re-baixáveis a qualquer momento).
- `tokenizers` 0.23: `BpeTrainer` **não aceita** `workers`/`chunk_size` (paralelismo é interno;
  kwarg desconhecido só avisa e é ignorado); `initial_alphabet` é só no trainer, não no
  construtor de `models.BPE`; contagem via `get_vocab_size()` (não `.vocab_size`).
- RAM nesta máquina é apertada p/ pipelines de dados: o servidor de inferência Strata segura
  ~52 GB dos 62 GB. Pipelines (tokenizer, binarização) devem processar **por row group**
  (nunca carregar parquet inteiro); o run do tokenizer picou 3 GB assim.
- Byte-level BPE: `id_to_token()` mostra bytes como `Ġ`/`Ã©` etc. (mapeamento byte→unicode do
  GPT-2) — é cosmético; `decode()` faz roundtrip exato (coberto em `tests/test_tokenizer.py`).
- O `split=val` herdou o ruído web do OSCAR (páginas de casino/spam SEO tipo `mostbet`
  passaram nos filtros de 2024-10) → perplexidade de val sai alta em termos absolutos; serve
  como régua estável p/ comparar runs. Se incomodar, filtrar val antes do Passo 6.
- `prepare_tokens.py`: tokenização em lote com `n_process=8` (HF) ≈ 215M tokens/min
  (por-doc = 4 MB/s, evitar). O orçamento de tokens estoura em ~1 batch de flush
  (`--flush-chars` 10M ≈ 2,4M tokens) — irrelevante p/ 2,75B.
- **transformers v5 + modelo próprio:** (a) tied weights exigem declarar
  `_tied_weights_keys = {"lm_head.weight": "embed.weight"}` na classe, senão
  `save_pretrained` erro "shared tensors not properly defined"; (b) `from_pretrained`
  constrói o modelo sob meta-device e **materializa buffers non-persistent uninitialized**
  (o `inv_freq` do RoPE voltava lixo → logits errados com pesos bit-exatos) — por isso
  `RoPE.inv_freq()` é calculado sob demanda, sem buffer; (c) `post_init()` chama
  `_init_weights` em CADA módulo — o init custom (0.02 + escala residual via flag
  `_residual_proj`) vive ali, não no `__init__`.
- `torch.Tensor.ones_()` não existe — usar `fill_(1.0)` (pegadinha ao escrever `_init_weights`).
- `PingoCache`: todas as camadas de um forward escrevem em `cur_len` (o avanço é único,
  `commit()` no fim do forward — se cada camada incrementasse, a camada 2 escreveria
  deslocado); pós-prefill só step de 1 token (t>1 com cache não-vazio precisaria de mask).
  Cache é inferência (`no_grad`): a cópia p/ buffer pré-alocado não propaga gradiente.
- O Strata segura ~31,4 GB dos 32 GB de **VRAM** também (não só a RAM) → qualquer sanity
  GPU (forward 120M, bench de geração) precisa dele parado. Checar `nvidia-smi` antes.
- Speedup do KV cache em ctx curto (32–64 tokens, CPU) é só ~1,8×; o ganho de 10×+ do
  critério do Passo 7 vem de ctx longo (geração longa) na GPU.
