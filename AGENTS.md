# AGENTS.md — estado do projeto

Ao iniciar uma sessão, leia isto primeiro. O plano completo está em [`PLAN.md`](PLAN.md).

> **Regra de ouro: atualize este arquivo SEMPRE que fizer alguma coisa** — terminar um
> passo, meio passo, um bugfix, uma decisão nova ou um gotcha descoberto. Commit junto
> (mesmo commit ou commit separado `docs: estado`), para a próxima sessão (ou pessoa)
> retomar exatamente de onde parou. Mantenha a seção "Onde estamos" honesta: estado real
> dos arquivos/commands, não o estado desejado.

## Onde estamos

**Passo 3 de 7 — Tokenizer: PRÓXIMO.** (Passo 2 concluído em 2026-10-29)

Feito:
- ✅ Passo 1 (setup uv + estrutura): deps instaladas, pytest verde, CUDA ok na RTX PRO 4500.
- ✅ **Passo 2 (dados)**: 100/100 parts de `2024-10` baixadas (`data/2024-10/pt/`, 42,7 GB);
  `process_data.py` rodado completo → 19,3M docs → **10,7M aprovados (55,6%)** com
  1,9M dups exatos removidos; **59,7B chars ≈ ~15B tokens** (folga de 6× sobre a meta de
  2,5–3B → dump reserva `2023-40` NÃO será necessário). Parquet: 23 GB em `data/raw/`
  (colunas `id, content, url, dump, part, split`, ~2% `val`); stats em `data/raw/stats.json`.
  Amostras inspecionadas: pt real (ocasional pt-PT, aceitável por decisão do plano).

A fazer no **Passo 3 (tokenizer)**:
1. Criar `scripts/train_tokenizer.py`: BPE **byte-level, vocab 16k** (lib `tokenizers`),
   treinado em amostra de ~1 GB dos parquets (`split=train`!), especiais
   `<|bos|> <|eos|> <|pad|>`. Salvar em `tokenizer/` (gitignored).
2. **Critério de aceite**: taxa chars/token registrada e comparada com GPT-2 em texto pt
   (guardar o número em `tokenizer/vocab_stats.json` ou similar + no AGENTS.md).
3. Marcar `[x] 3.` no roadmap do README; commit `step 3: tokenizer ...`.

Decisão pendente do Passo 4 que já podemos tomar: com ~15B tokens disponíveis e meta de
2,5–3B, o `prepare_tokens.py` deve **amostrar** o corpus (ex.: subset de docs até atingir
N tokens; manter o `val.bin` completo dos docs `split=val` para medir perplexidade estável).

## Próximos passos (do PLAN.md)

| Passo | O que | Arquivos | Critério |
|---|---|---|---|
| **3** | Tokenizer BPE byte-level **16k** (~1 GB de amostra), especiais `<\|bos\|> <\|eos\|> <\|pad\|>` | `scripts/train_tokenizer.py` | taxa chars/token registrada vs GPT-2 em pt |
| **4** | Binarização: parquet → shards `.bin` uint16 (~1M tok) + `val.bin` | `scripts/prepare_tokens.py` | loader memmap (`pingo/data.py`) lê batches |
| **5** | Modelo do zero: RMSNorm, RoPE (θ=10k), MHA causal (`F.scaled_dot_product_attention`), SwiGLU, `PingoCache` KV, `PingoLM(PreTrainedModel)` | `pingo/{config,model,cache,generate}.py` | params ≈ alvo (120M: 12L/896d/14h); KV cache equiv `<1e-5` fp32; overfit loss≈0 |
| **6** | Pré-treino: AdamW wd 0.1, warmup+cosine, clip 1.0, bf16, ~0.5M tok/batch global, ckpt 15 min + `--resume`, seed fixa | `scripts/train.py` | val loss estável; resume reproduz a curva |
| **7** | Geração + perplexidade val + `save_pretrained` | `scripts/sample.py` | texto pt coerente; 10× mais rápido c/ KV cache |

Config default: **Pingo-120M** (12 camadas, d_model 896, 14 heads, SwiGLU hidden 2304, ctx 2048).

## Convenções

- **Tudo via `uv`**: `uv run ...`, `uv add ...`. Não usar pip/venv direto.
- Commits: mensagem `step N: ...` (ver `git log`); **push só após revisão do usuário**.
- `data/`, `tokenizer/`, `*.parquet`, `*.bin` são gitignored; `uv.lock` é versionado.
- Máquinas: dev/testes na M4 local (CPU/MPS, config tiny); treino na RTX PRO 4500 32 GB.
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
- Disco: `data/` ocupa ~63 GB (40 raw + 23 parquet). Para o Passo 4 pode-se apagar os
  `.jsonl.zst` depois dos parquets conferidos (re-baixável a qualquer momento).
