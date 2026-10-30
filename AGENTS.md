# AGENTS.md — estado do projeto

Ao iniciar uma sessão, leia isto primeiro. O plano completo está em [`PLAN.md`](PLAN.md).

> **Regra de ouro: atualize este arquivo SEMPRE que fizer alguma coisa** — terminar um
> passo, meio passo, um bugfix, uma decisão nova ou um gotcha descoberto. Commit junto
> (mesmo commit ou commit separado `docs: estado`), para a próxima sessão (ou pessoa)
> retomar exatamente de onde parou. Mantenha a seção "Onde estamos" honesta: estado real
> dos arquivos/commands, não o estado desejado.

## Onde estamos

**Passo 7 CONCLUÍDO (2026-10-30) — o Pingo-120M da Fase 1 está pronto e avaliado.**
Próximo: **Passo 8** (escalar p/ ~350M). Modelo em `runs/pt-120m/final/` — ppl no val
**11,67** (3,54 bits/token, 409.600 tokens), geração de pt coerente, KV cache 10,6–11,8×
mais rápido em serving (B=16, ctx 1024–2016). Relatório completo em
`runs/pt-120m/eval.json` (`scripts/sample.py` reproduz).
Antes: Passo 6 fechado com val loss 2,507 / ppl 12,3 — 5.248 steps, 2,75B tokens,
7,56 h, 100,9k tok/s, curva monótona, zero NaN.

### ▶️ Se você é uma NOVA sessão — siga nesta ordem

1. **Sanity sem GPU**: `uv run pytest -q` → **31 verdes** esperados.
2. **Confira a GPU livre** (`nvidia-smi`): se mostrar ~31 GB usados, o Strata está de pé —
   **AVISE o usuário antes de qualquer coisa em CUDA** (regra de ouro). GPU livre = ~0 MiB.
3. **Não re-treine**: o Passo 6 já está fechado; `runs/pt-120m/{ckpt.pt,ckpt-prev.pt,
   best.pt,final/}` são o resultado. Rodar `scripts/train.py` sem `--resume` aí
   **sobrescreveria** o run bom — só faça isso se o usuário pedir outro run (use `--out`
   novo). Receita que funcionou: `uv run python scripts/train.py --compile` (defaults =
   B16×accum16, ctx 2048, lr 3e-4→3e-5, warmup 400, 5.248 steps, eval a cada 250,
   ckpt a cada 15 min). Memória de cálculo: ~101k tok/s com compile (55k sem) ⇒ 2,75B
   tokens ≈ 7,5 h; VRAM 18,8 GiB com compile (28,5 sem) de 32,6 GiB.
4. **Se precisar parar/retomar**: `--resume` com os MESMOS args (inclusive `--max-steps`,
   que NÃO vai no ckpt e define o lr); `KeyboardInterrupt` salva ckpt.
5. **Avaliação**: `uv run python scripts/sample.py` (ppl + amostras + bench do KV cache;
   `--no-ppl` / `--no-bench` p/ encurtar, `--out` p/ JSON). 2–3 min na GPU.
6. **Passo 8** (próximo, fora da Fase 1): escalar p/ **Pingo-350M** repetindo o pipeline
   (os shards de 2,75B tokens já existem; ~350M×2,75B é undertrained — vale arrumar mais
   tokens/Passo 9 antes). Nada foi começado ainda.

Referência de leitura da curva (não entrar em pânico): init = 9,71; o val desceu
5,02@250 → 3,17@1000 → 2,69@2750 → 2,51@5000 (bem melhor que os 3,2–3,9 que o plano
estimava). NaN → cortar lr pela metade (1,5e-4) e recomeçar do zero. OOM → baixar B e
subir accum na mesma proporção. Disco: `ckpt.pt` ≈ 1,5 GB (+ otimizador no payload);
`runs/pt-120m` inteiro = 4,8 GB (com `best.pt` do vigia).

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
  cpu/cuda (testes travam em CPU de propósito).
- ✅ **Benchmark + smoke na GPU (2026-10-29, Strata derrubado)** — números REAIS, medidos:
  `runs/smoke-120m` (B16×accum16, 44 steps): loss 9,85→8,12, val 8,63→8,07 (ppl 3186),
  ckpt+`final/`+`--resume` validados na prática (tokens contínuos, curva sem quebra);
  VRAM pico **28.531 MiB** de 32.623 (~4 GB livres) → B=32 NÃO cabe.
  Throughput: **~55k tok/s sem compile** e **~102k tok/s com `--compile`** (quase 2×,
  medido em `runs/bench-b16-compile`); VRAM com compile cai: B16 **18.8 GiB**, B12 **14.5 GiB**.
  → run completo = **~14 h sem compile / ~7,5 h com compile** em `runs/pt-120m`.
  Bugfix encontrado nesse teste: com `--compile` o `named_parameters()` vem como
  `_orig_mod.*`, então (a) o `embed.weight` caía no grupo COM weight decay e (b) o ckpt
  salvava chaves prefixadas (resume só com compile). Agora `build_optimizer`/`save_ckpt`/
  `resume` fazem `getattr(model, "_orig_mod", model)`; coberto por
  `test_optimizer_com_pesos_de_compile` (**31 testes verdes**).
- ✅ **PASSO 6 FECHADO (run de 2026-10-29→30)**: `runs/pt-120m` — 5.248 steps, 2,75B
  tokens, **7,56 h** (100,9k tok/s médio, `--compile`), VRAM 18,8 GiB, GPU 84–86 °C sem
  throttle. Val: 5,02@250 → 3,17@1000 → 2,69@2750 → 2,51@5000 → **2,507 final (ppl 12,3)**;
  monótono, zero NaN. `best.pt` (val 2,5139 @ step 5000) salvo pelo vigia.
  Qualidade pt: geração coerente com amostragem (ver `[x] 6.` no README); no greedy ela
  entra em loop de repetição — típico de 120M, assunto do Passo 7 (penalidade/typical).
  Vigia usado: script externo (lê `log.jsonl`, salva `best.pt`, SIGINT se o val piorar 2×
  seguidas ou der NaN) — mora fora do repo; se quiser, promover a `scripts/`.
- ✅ **PASSO 7 FECHADO**: `scripts/sample.py` (ppl + amostras + bench do KV cache) e
  `--repetition-penalty` em `pingo/generate.py` (`apply_repetition_penalty`, semântica HF).
  Números no modelo final: **ppl 11,67 / 3,54 bits/token** (val, 409.600 tokens, ctx 2048,
  rng fixo); amostras de pt em parágrafos coerentes; KV cache **5,8×/10,6×/11,8×**
  (ctx 512/1024/2016, B=16) com **tokens idênticos** ao caminho naive. `tests/test_sample.py`
  (6 testes; suíte **37 verdes**). Relatório em `runs/pt-120m/eval.json`.

## Decisões de treino (Passo 6)

- **Sem run de 3 dias**: run cabe em janela de noite (~5,2k steps); se estourar, fatiar
  com `--resume` (curva idêntica, testada).
- Amostragem dos batches é **com reposição** (estilo nanoGPT sobre os 2,75B tokens);
  ~63% dos tokens vistos em média numa época de amostragem — aceitável p/ Fase 1.
- `--compile` **ficou ON no run real** (medido: 101k vs 55k tok/s e menos VRAM) — o 1º
  run saiu sem sustos; manter ON em runs futuros.
- Val do OSCAR tem ruído (casino/spam) → ppl absoluta alta é esperado; régua p/ comparar.

## Próximos passos (do PLAN.md)

| Passo | O que | Arquivos | Critério |
|---|---|---|---|
| ~~6~~ ✅ | Pré-treino: AdamW wd 0.1, warmup+cosine, clip 1.0, bf16, ~0.5M tok/batch global, ckpt 15 min + `--resume`, seed fixa | `scripts/train.py` | ✅ val 2,507 estável; resume bit-exato |
| ~~7~~ ✅ | Geração + perplexidade val + `save_pretrained` | `scripts/sample.py` | ✅ pt coerente; ppl 11,67; KV cache 10,6–11,8× (B16) |
| **8** | Escalar p/ **Pingo-350M** (mesma receita; precisa de mais tokens — 2,75B é pouco p/ 350M) | `pingo/config.py` | ppl no val melhor que o 120M na mesma régua |

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
- **KV cache: o ganho NÃO é em batch=1.** Medido na RTX (120M, 32 tokens novos):
  B=1 dá ~0,8–1,9× (o decode é limitado por latência de kernel, não por FLOPs — recalcular
  a sequência inteira sai quase de graça); **B=16 dá 5,8×@512, 10,6×@1024, 11,8×@2016**.
  Ou seja: o critério "10×+" do plano é sobre throughput de serving, não latência
  single-stream. Bench sempre compara os tokens gerados (`mesmos_tokens`) p/ não medir
  coisa errada.
- Bench com `ctx == max_position_embeddings` não gera nada (o cache para no limite e
  `generate_naive` sai na hora) → `scripts/sample.py` corta o ctx para `max_ctx - n_new`
  e marca `ctx_ajustado: true` no JSON.
- Greedy em 120M **entra em loop de repetição** ("…está a ser implementado." em loop);
  `generate(..., repetition_penalty=1.15)` resolve (default do `sample.py`).
- `ppl` no val depende de quais janelas caem no rng fixo: 20 batches × B16 → 12,3;
  25 batches × B8 → 11,67. Comparações entre runs são válidas, valores absolutos variam
  ~5% com o nº de tokens amostrados.
- **O lr NÃO é gravado no ckpt**: ele é recalculado por `lr_at(step, max_steps)` a cada
  step. Logo `--resume` tem que usar o **mesmo `--max-steps`** (e mesmo lr/warmup) do
  início, senão a curva de lr "salta" sem nenhum aviso (foi o que aconteceu num teste
  manual 40→44). No run real o default vem do `meta.json` e é constante → ok.
- `--compile` (torch.compile) funciona com o modelo e com `save_pretrained` (o `final/`
  sai sem prefixo, via `_orig_mod`), e é **mais rápido E mais leve** em VRAM aqui
  (B16: 18.8 vs 28.5 GiB — fusão de rmsnorm/swiglu reduz ativações); ganho típico 55k→102k
  tok/s. Formas fixas (B, ctx constantes) → sem recompiles. Pode (e deve) usar
  `--compile` no run longo; só mantenha a flag IGUAL entre run e `--resume` (agora o
  ckpt é agnóstico, mas ajuda a não confundir).
- `train.py` pega `device` automático (cuda se existir) → nesta máquina os testes do
  script **precisam de `--device cpu`** (a GPU está tomada e o teste de resume exige
  determinismo CPU). Estado retomável = `step` + `opt.state_dict` + RNGs (python,
  `np.random.Generator.bit_generator.state`, torch CPU/CUDA) — esquecer um deles e a
  "curva idêntica" quebra silenciosamente.
