# AGENTS.md — estado do projeto

Ao iniciar uma sessão, leia isto primeiro. O plano completo está em [`PLAN.md`](PLAN.md).

> **Regra de ouro: atualize este arquivo SEMPRE que fizer alguma coisa** — terminar um
> passo, meio passo, um bugfix, uma decisão nova ou um gotcha descoberto. Commit junto
> (mesmo commit ou commit separado `docs: estado`), para a próxima sessão (ou pessoa)
> retomar exatamente de onde parou. Mantenha a seção "Onde estamos" honesta: estado real
> dos arquivos/commands, não o estado desejado.

## Onde estamos

**Passo 2 de 7 — Dados: quase lá.** (download 100% concluído; filtro/parquet rodando)

Feito:
- ✅ Passo 1 (setup uv + estrutura): deps instaladas, pytest verde, CUDA ok na RTX PRO 4500.
- ✅ Scripts do Passo 2: `scripts/download_data.py` (download retomável via HTTP Range,
  4 workers, orçamento de GB) e `scripts/process_data.py` (content → filtros de qualidade →
  `data/raw/*.parquet`, ~2% marcado `val`, stats em `data/raw/stats.json`).
- ✅ Proveniência dos dados documentada no README (repo/dump/arquivos) + wrappers
  `scripts/download_data.sh` / `.bat`.
- ✅ **Download completo**: 100/100 parts de `2024-10` em `data/2024-10/pt/` (42,7 GB,
  sizes validados contra o repo). Verificar: `ls data/2024-10/pt/*.jsonl.zst | wc -l` = 100.

Pendente (nesta ordem):
1. **Aguardar `process_data.py` completo** (rodando em background, log em `data/process.log`;
   idempotente — retomar é só rodar de novo). Parquet final: 100 arquivos
   `data/raw/2024-10-pt_part_N.parquet`.
2. **Critério de aceite do Passo 2** (do PLAN.md): conferir `data/raw/stats.json`
   (docs/tokens registrados — smoke do part_1: ~64% de aproveitamento) + inspecionar amostras.
   Meta: ~2,5–3B tokens brutos. ⚠️ Projeção inicial com só o dump 2024-10 é **1,5–2,7B
   tokens** → se ficar abaixo, baixar o dump reserva **`2023-40`** (68 parts, ~41,8 GB):
   `uv run python scripts/download_data.py --dumps 2023-40 --max-gb 45`.
3. Marcar `[x] 2.` no roadmap do README e commitar `step 2: dados ...`.
   **Nunca dar push sem o usuário revisar o diff antes.**

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
