# AGENTS.md — estado do projeto

Ao iniciar uma sessão, leia isto primeiro. O plano completo está em [`PLAN.md`](PLAN.md).

> **Regra de ouro: atualize este arquivo SEMPRE que fizer alguma coisa** — terminar um
> passo, meio passo, um bugfix, uma decisão nova ou um gotcha descoberto. Commit junto
> (mesmo commit ou commit separado `docs: estado`), para a próxima sessão (ou pessoa)
> retomar exatamente de onde parou. Mantenha a seção "Onde estamos" honesta: estado real
> dos arquivos/commands, não o estado desejado.

## Onde estamos

**Passo 6 de 7 — Pré-treino: EM ANDAMENTO** (script + testes prontos e verdes em CPU;
faltam timing na GPU e o run completo — a GPU será liberada: Strata derrubado e Opencode
migrado para modelo na nuvem).

### ▶️ Se você é uma NOVA sessão (Opencode na nuvem, GPU agora livre) — siga nesta ordem

1. **Sanity sem GPU** (não precisa do Strata): `uv run pytest -q` → **30 verdes** esperados.
2. **Confira a GPU livre**: `nvidia-smi` (com o Strata derrubado deve mostrar ~0 MiB usados;
   se vir ~31 GB ocupados, o Strata ainda está de pé — AVISE o usuário antes de qualquer
   coisa em CUDA, regra de ouro do projeto).
3. **Benchmark do micro-batch** (~2–3 min por config; compare `toks_s` no log de cada uma;
   monitore VRAM com `nvidia-smi` em outro terminal):
   ```bash
   uv run python scripts/train.py --batch-size 16 --accum-steps 16 --max-steps 40 \
       --eval-steps 999 --ckpt-min 999999 --out runs/bench-b16
   uv run python scripts/train.py --batch-size 32 --accum-steps 8 --max-steps 40 \
       --eval-steps 999 --ckpt-min 999999 --out runs/bench-b32
   ```
   Escolha o maior B que rodar folgado (deixar >4 GB livres) e mantenha o batch GLOBAL em
   512 sequences (B×accum=512, ou seja 524k tok/step — não mude isso sem recompensar lr).
4. **Smoke do 120M real** (~10–15 min):
   ```bash
   uv run python scripts/train.py --max-steps 150 --out runs/smoke-120m --log-steps 10 --eval-steps 75
   ```
   A loss deve sair de ~9,7 (ln 16384) caindo de forma limpa (sem NaN/spikes); `final/`
   deve aparecer em `runs/smoke-120m/`.
5. **Run completo** (~5.248 steps; lance ASSIM, com `nohup`, para sobreviver ao fim da
   sessão do agente):
   ```bash
   mkdir -p runs/pt-120m
   nohup uv run python scripts/train.py > runs/pt-120m/train.out 2>&1 &
   tail -n 20 runs/pt-120m/log.jsonl        # acompanhar depois (linhas train_loss/val_loss)
   ```
   Estimativa: plano previa ~60–80M tok/s na RTX PRO 4500 → 10–13 h; se não couber numa
   noite, deixe os ckpts de 15 min e na noite seguinte rode
   `uv run python scripts/train.py --resume` (mesmos args default; a curva continua
   idêntica — testado bit-exato). **Nunca rode o run completo sem `--resume` se
   `runs/pt-120m/ckpt.pt` existe** (recomeçaria do zero e sobrescreveria).
6. **Critérios para fechar o Passo 6** (da PLAN.md): val loss caindo de forma estável
   (esperado: ~9,7 no início; ordem de ~3,2–3,9 no fim com 2,75B tokens; ppl do val fica
   alta em absoluto — ruído do OSCAR, é régua relativa); checkpoint `--resume` já validado
   em teste unitário; se quiser provar na prática: Ctrl-C no meio, `--resume`, a curva
   continua sem descontinuidade.
7. **Fechar**: marcar `[x] 6.` no roadmap do README, atualizar AGENTS.md (estado real:
   steps rodados, tok/s medido, val loss final), commit `step 6: pré-treino ...`.
   Passo 7 (`scripts/sample.py`): perplexidade no val + geração (`pingo/generate.py` já
   existe e é testado) + benchmark 10× KV cache (agora possível com a GPU livre).

Vals de referência para não entrar em pânico: init = 9,71; NaN → cortar lr pela metade
(lr 1,5e-4) e re-começar do zero (run novo, sem ckpt); OOM → baixar B e subir accum na
mesma proporção. Disco: cada ckpt.pt ≈ 2,2 GB (modelo+otimizador), rotação mantém 2
(`ckpt.pt` + `ckpt-prev.pt`); `df -h` antes do run.

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
- 🔶 **Passo 6 (pré-treino)** — `scripts/train.py` pronto, **testes verdes em CPU**
  (`tests/test_train.py`, 4 testes; suíte **30 verdes**): lr warmup+cosine ok; AdamW com
  grupos de decay corretos (sem decay em RMSNorm 1-D + `embed.weight`); **teste central:
  run contínuo 6 steps == 3 steps + `--resume` até 6 com pesos BIT-EXATOS e mesma val
  loss** (RNGs python/numpy-Generator/torch salvos no ckpt). Dry-run CLI em CPU com os
  tokens REAIS (`--tiny --ctx 256`, 20 steps): loss inicial 9,71 ≈ ln 16384, val caindo,
  `final/` salvo em formato HF. Decisões de default: micro-batch B=16 × ctx 2048 ×
  accum 16 = **524k tok/step**; max_steps default = época do meta ≈ **5.248 steps**;
  lr 3e-4 → min 3e-5, warmup 400, betas (0.9, 0.95), clip 1.0; eval 20 batches fixos a
  cada 250 steps; ckpt a cada 15 min p/ `ckpt.pt` (atomic + rotação p/ `ckpt-prev.pt`);
  logs em `runs/pt-120m/log.jsonl`; KeyboardInterrupt salva ckpt; `--device` força
  cpu/cuda (testes travam em CPU de propósito). **A fazer:** (1) benchmark tok/s na GPU
  p/ calibrar B e estimar a noite; (2) run completo (1 noite, ou fatiado com --resume);
  (3) conferir val loss estável caindo; (4) `[x] 6.` no roadmap.

## Decisões de treino (Passo 6)

- **Sem run de 3 dias**: run cabe em janela de noite (~5,2k steps); se estourar, fatiar
  com `--resume` (curva idêntica, testada).
- Amostragem dos batches é **com reposição** (estilo nanoGPT sobre os 2,75B tokens);
  ~63% dos tokens vistos em média numa época de amostragem — aceitável p/ Fase 1.
- `--compile` existe mas fica OFF no 1º run (menos móveis p/ diagnosticar); testar depois.
- Val do OSCAR tem ruído (casino/spam) → ppl absoluta alta é esperado; régua p/ comparar.

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
  a RTX roda o servidor de inferência Strata (~31 GB dos 32 GB de VRAM; e até 2026-10-29
  era ELE que servia o próprio modelo do Opencode). Para liberar a GPU o usuário muda o
  Opencode para um modelo na nuvem (sessão nova) e derruba o Strata. Nunca tomar a GPU
  por conta própria; conferir com `nvidia-smi` (se mostra ~31 GB usados, Strata de pé).
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
  GPU (forward 120M, bench de geração, treino) precisa dele parado. Checar `nvidia-smi`
  antes de CUDA. (Desde 2026-10-29 o fluxo é: Opencode na nuvem + Strata derrubado p/
  treinar; ver "▶️ Se você é uma NOVA sessão" no topo.)
- Speedup do KV cache em ctx curto (32–64 tokens, CPU) é só ~1,8×; o ganho de 10×+ do
  critério do Passo 7 vem de ctx longo (geração longa) na GPU.
- `train.py` pega `device` automático (cuda se existir) → nesta máquina os testes do
  script **precisam de `--device cpu`** (a GPU está tomada e o teste de resume exige
  determinismo CPU). Estado retomável = `step` + `opt.state_dict` + RNGs (python,
  `np.random.Generator.bit_generator.state`, torch CPU/CUDA) — esquecer um deles e a
  "curva idêntica" quebra silenciosamente.
