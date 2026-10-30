# AGENTS.md — estado do projeto

Ao iniciar uma sessão, leia isto primeiro. O plano completo está em [`PLAN.md`](PLAN.md).

> **Regra de ouro: atualize este arquivo SEMPRE que fizer alguma coisa** — terminar um
> passo, meio passo, um bugfix, uma decisão nova ou um gotcha descoberto. Commit junto
> (mesmo commit ou commit separado `docs: estado`), para a próxima sessão (ou pessoa)
> retomar exatamente de onde parou. Mantenha a seção "Onde estamos" honesta: estado real
> dos arquivos/commands, não o estado desejado.

## Onde estamos

**Fase 1 CONCLUÍDA (Passos 1–7) — e o PLANO MUDOU: agora é SFT, não o Pingo-350M.**
**SFT-1 (gerador do dataset) CONCLUÍDO em 2026-10-30** — `scripts/gen_sft.py` + 21 testes;
pilotos auditados (300 pares, 2 versões) e o volume de **30k pares rodando** em
`data/sft/sft-v1.jsonl` (retomável com `--resume`).

Decisão do usuário em 2026-10-30: **o Passo 8 (escalar p/ 350M) está suspenso**. A próxima
etapa é **SFT** (instruction tuning) do Pingo-120M com um dataset de Q&A gerado
sinteticamente a partir do próprio corpus OSCAR, usando um **modelo professor local**.

### 🔒 O checkpoint do Pingo-120M tem BACKUP (não apague!)

`runs/pt-120m-BACKUP-2026-10-30/` = cópia **read-only** (`dr-xr-xr-x`) de `runs/pt-120m/`
inteiro (4,8 GB), com `SHA256SUMS.txt` que bate com o original (model.safetensors, ckpt.pt,
best.pt). Pedido explícito do usuário: **preservar com carinho o modelo que funciona**.
Nenhum script até agora escreve nesse diretório; `train_sft.py` deve gravar em `runs/sft-*`.

Estado do que existe hoje:
- Modelo pré-treinado em `runs/pt-120m/final/` (5.248 steps, 2,75B tokens, 7,56 h):
  val loss 2,507 / **ppl 11,67** (3,54 bits/token, 409.600 tokens), geração de pt coerente,
  KV cache 10,6–11,8× em serving (B=16). Avaliação reproduzível: `scripts/sample.py`;
  relatório em `runs/pt-120m/eval.json`.
- **Suíte: 58 testes verdes** (37 dos Passos 1–7 + 21 do gerador SFT), `main` no GitHub.
- **Professor no ar**: `llama.cpp` (build b11669) na **porta 8001**, modelo
  `unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL`, **8 slots**, ctx 4096/slot. (Quem sobe o servidor
  é o usuário; ver a seção SFT abaixo.)

### ▶️ Se você é uma NOVA sessão — siga nesta ordem

1. **Sanity sem GPU**: `uv run pytest -q` → **58 verdes** esperados.
2. **GPU** (`nvidia-smi`): se mostrar ~31 GB usados, o **Strata** está de pé — **AVISE o
   usuário antes de qualquer coisa em CUDA** (regra de ouro). O professor llama.cpp ocupa
   só ~3–4 GB e pode estar rodando junto.
3. **Confirme o professor**: `curl -s localhost:8001/v1/models` deve devolver um Qwen3.5-4B
   GGUF. Se não houver nada, **peça ao usuário para subir** — não suba servidor por conta
   própria (o script de serving mora em `~/llm_serving_scripts/`, fora do repo).
4. **Não re-treine o base**: `runs/pt-120m/{ckpt.pt,ckpt-prev.pt,best.pt,final/}` é o
   resultado da Fase 1. `scripts/train.py` sem `--resume` **sobrescreveria** — só rode em
   outro `--out`. Receita que funcionou: `--compile` + defaults (B16×accum16, ctx 2048,
   lr 3e-4→3e-5, warmup 400, 5.248 steps); ~101k tok/s ⇒ 7,5 h; 18,8 GiB de VRAM.
5. **Avaliação do base**: `uv run python scripts/sample.py` (ppl + amostras + bench do
   KV cache; `--no-ppl`/`--no-bench` p/ encurtar). 2–3 min na GPU.
6. **Estado do SFT**: o gerador (`scripts/gen_sft.py`) está pronto e testado; o volume de
   30k pares roda em background (~11 h) e é **retomável**:
   `uv run python scripts/gen_sft.py --max-examples 30000 --out data/sft/sft-v1.jsonl --resume`
   (não deixe dois escritores no mesmo arquivo). Confira o progresso pela 1ª linha do
   `data/sft/sft-v1.log` e pelo `wc -l data/sft/sft-v1.jsonl`.
   **Próxima tarefa: SFT-2** (`scripts/validate_sft.py` — dedup MinHash + split por documento).

## SFT — rumo novo (decidido em 2026-10-30)

**Objetivo**: ensinar o Pingo-120M a **responder perguntas sobre o mesmo domínio em que foi
pré-treinado** (OSCAR pt), criando o dataset de instruções **sinteticamente**: um modelo
professor local lê um trecho do OSCAR e escreve pares pergunta/resposta sobre ele.
O 350M fica para depois (e, quando vier, precisa de mais tokens: 2,75B é pouco p/ 350M).

### Pipeline acordado (5 etapas, nada implementado ainda)

| Etapa | O que | Arquivo proposto | Status |
|---|---|---|---|
| **SFT-1** | **Gerador**: amostra docs de `data/raw/*.parquet` (`split=train`, **nunca** `val`), filtro duro de qualidade, chunk ≤ ~500 tokens, chama o professor, grava JSONL com procedência | `scripts/gen_sft.py` | ✅ (30k rodando) |
| **SFT-2** | **Validador**: JSON ok, `evidencia` é substring do doc, resposta em pt, tamanho, dedup (MinHash) + split train/val/heldout **por documento** | `scripts/validate_sft.py` | ⬜ |
| **SFT-3** | **Treino**: `attention_mask` + padding no modelo, loss **só nos tokens da resposta**, packing com `<|end|>`, cosine curto, ckpt | `scripts/train_sft.py` (+ `pingo/model.py`) | ⬜ |
| **SFT-4** | **Avaliação**: geração nos prompts do heldout, taxa de evidência/fidelidade, checar que a ppl do val do pré-treino não desabou (esquecimento catastrófico) | `scripts/sample.py --model <sft>` | ⬜ |
| **SFT-5** | Opcional: misturar 10–20% de instruções gerais pt para o modelo não ficar "só respondedor de documentos" | — | ⬜ |

**Piloto primeiro**: gerar **100–300 exemplos**, auditar à mão ~50, e só então rodar volume
(20–50 mil já são suficientes para 120M; o gargalo é qualidade, não quantidade).

### Formato do exemplo (JSONL, uma linha por par)

O desenho original ficou em `prompt_version v1`; o formato **em uso** (v2, com modo ancorado)
está na seção "Formato real do JSONL" abaixo.

Taxonomia de `tipo`: `extracao` (50%), `explicacao` (30%), `resumo` (15%) — e ~5% de
**recusas** para ensinar honestidade.

Ganho de graça descoberto no piloto: o próprio professor **veta o trecho** (`status:
"invalido"`) quando ele não é texto legível em pt — bolsão/menu/spam. Virou o filtro de
qualidade mais eficaz do pipeline (~48 vetoes em ~330 trechos), sem custo extra.

### Como falar com o professor (llama.cpp :8001) — testado

- **Estilo OpenAI**: `POST /v1/chat/completions`. `pyarrow` já está no projeto; **`requests`
  NÃO está instalado** (use `urllib.request` ou `uv add requests`).
- **JSON garantido (essencial)**: `response_format` precisa ser o formato completo —
  `{"type": "json_schema", "json_schema": {"name": "qa", "schema": {...}, "strict": true}}`.
  `{"type":"json_object", "schema":...}` é **ignorado** → sai texto livre/quebrado.
- **Thinking DESLIGADO** (`--reasoning off` / `enable_thinking:false` no servidor): em
  modo instruct a resposta sai direta. Se vier com raciocínio, ele vai em
  `message.reasoning_content` (separado do `content`) e **queima o budget de tokens**.
  Checar `chat_template_kwargs: {"enable_thinking": false}` por request se precisar.
- Sampling do card (instruct): `temp 0.7 / top_p 0.8 / top_k 20 / min_p 0 / presence 1.5`.
- `usage.completion_tokens` é o custo real; ~8 slots concorrentes ⇒ ~580 tok/s agregados
  (medido pelo usuário com outro modelo; reconfirmar com o Qwen).
- **NÃO** rodar nada em CUDA pesado enquanto o professor estiver servindo — o servidor
  precisa da VRAM.

### Decisões TOMADAS pelo usuário (2026-10-30) — valem para o SFT

1. **Tokens de chat**: **(B) tokens especiais** — adicionar `<|user|>/<|assistant|>/<|end|>` ao
   tokenizer (`vocab_size` 16.384→16.387, `resize_token_embeddings`, init dos novos na média
   dos embeddings). Ainda **não implementado**: é tarefa do SFT-3. *(ATENÇÃO: hoje o
   `tokenizer/tokenizer.json` está com 16.384 — não regerar o tokenizer, só adicionar tokens.)*
2. **Volume**: **30 mil pares** (`data/sft/sft-v1.jsonl`).
3. **Recusas**: **sim, ~5%** (`--recusa-frac 0.05`).
4. **Formato do exemplo: MISTO 50/50** (`--ancorado-frac 0.5`) — metade **ancorada** (o trecho
   vai junto: uso tipo mini-RAG, e "Isso não consta no texto." faz sentido) e metade **sem
   contexto** (closed-book: perguntas autossuficientes, recusa = "Não tenho informações
   suficientes para responder isso."). O campo `ancorado` (bool) do JSONL diz qual é qual;
   `chunk_text` já é gravado em **todos** os exemplos, então a serialização do treino pode
   mudar sem regerar nada.
5. **Dataset**: fora do repo (gitignored, `data/sft/`), gerador + sementes versionados.
6. **Filtro de fonte**: duro — `min_chars 2000`, `min_paras 4`, dígitos/palavra ≤ 0.25,
   maiúsculas ≤ 30%, URLs ≤ 4, blocklist de URL/conteúdo (`--deny-url-regex`
   `/--deny-text-regex`: adulto, pirataria, cassino, "assistir/dublado/legendado") e teto por
   domínio (5% do volume).

### Formato real do JSONL (v2 — o que o gerador grava)

```json
{"doc_id","url","split":"train","dump","chunk_idx","chunk_chars","chunk_text",
 "teacher":"unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL","prompt_version":"v2","seed","attempt",
 "tipo":"extracao|explicacao|resumo|recusa","ancorado":true|false,
 "pergunta","resposta","evidencia"}
```

`evidencia` é **cópia literal** do `chunk_text` (vazia nas recusas) — SFT-2 revalida isso.

### Gotchas já descobertos com professores locais

- **LFM2.5-2.6B (tentativa anterior, descartada)**: o template **crava ` thinking`** no
  prompt de geração (sem switch) e ele gastou **1.661 tokens de raciocínio em inglês para
  ~180 de conteúdo** (~10×), com `content` vazio quando o budget acaba. Foi o motivo da
  troca para o Qwen3.5-4B (que tem toggle de thinking de verdade).
- OSCAR é web cru: **spam/casino/SEO** passa nos filtros do pré-treino. Para o SFT o filtro
  de fonte tem que ser mais duro (tamanho, dígitos/palavra, links, boilerplate, parágrafos).
- Amostrar **de `split=train`** sempre; `val` é a régua do pré-treino e não pode vazar.
- **Desempenho medido do professor (2026-10-30, 8 slots, Qwen3.5-4B UD-Q4_K_XL)**:
  ~**160–175 tok/s agregados** (completion) com 8 concorrentes; cada par custa **~240
  completion tokens** (pergunta+resposta+evidência) e o **prompt do trecho de 2.000 chars
  custa ~1.100 tokens** (prefill é ~⅓ do tempo de cada request). Resultado: **~40 pares/min**
  ⇒ **30k pares ≈ 11–12 h**. Prefill é caro: chunks menores (ou 2 pares por chamada)
  são as alavancas se precisar acelerar.
- **Detector de spam**: `chars/token` do NOSSO tokenizer **não serve** (texto "spun" de SEO
  comprime igual a texto bom: 3,93 vs mediana 4,13). O que funcionou: filtro duro +
  veto `invalido` do professor + **blocklist de URL** (spam adulto, pirataria e sites de
  "assistir filme dublado" passam por serem texto legível). Gancho: `--deny-url-regex`.
- **`is_pt` precisa aceitar palavra de 1 letra**: `[^\W\d_]{2,}` ignora "a", "o", "e", "é" e
  reprovava perguntas curtas legítimas. Agora conta palavras funcionais **+ palavras
  acentuadas** (2 sinais).
- **Modo misto (50/50) muda duas regras**: sem contexto, a pergunta não pode citar
  "texto/trecho/documento" (ninguém teria o trecho em mãos) e a recusa é outra frase.
- O campo `evidencia` (cópia literal) é a única verificação automática barata de que o
  professor não inventou — **não remova do schema**.

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
- ✅ **SFT-1 (gerador do dataset) — 2026-10-30**: `scripts/gen_sft.py` (~600 linhas) +
  `tests/test_gen_sft.py` (**21 testes**, professor fake, sem rede; suíte total **58 verdes**).
  Filtro de fonte duro + blocklist + teto por domínio; chunking em parágrafos (≤2.000 chars
  ≈ 500 tokens); 8 chamadas concorrentes; JSON estrito via `json_schema`; validação local
  (evidência **literal**, pt, tamanhos, dedup exata de pergunta) com até 3 tentativas por
  trecho; JSONL com `chunk_text` + procedência; `--resume` idempotente; stats em
  `<out>.stats.json`. **Piloto v1** (ancorado só): 300 pares em 7,4 min, 143 docs, 139
  domínios, 0 perguntas duplicadas — auditado à mão, achou spam adulto/pirataria (→ blocklist)
  e 28 perguntas citando "o texto" (→ motivou o modo misto). **Piloto v2** (50/50):
  163 ancorados / 137 sem contexto, 0 duplicatas, 0 vazamentos, `menciona_fonte` barrou 37.
  Restos: `data/sft/pilot-v{1,2}.jsonl` + `.stats.json`.

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
| ~~SFT-1~~ ✅ | **Gerador do dataset** (30k pares, misto ancorado/closed-book), 21 testes | `scripts/gen_sft.py` | ✅ 2 pilotos auditados; volume rodando |
| **SFT-2..5** | **NOVO RUMO (ver seção "SFT — rumo novo")**: validador → treino SFT → avaliação | `validate_sft.py`, `train_sft.py` | dataset auditado + modelo respondendo em formato |
| ~~8~~ ⏸️ | **SUSPENSO**: escalar p/ Pingo-350M (decisão do usuário 2026-10-30) | `pingo/config.py` | — |

Config default: **Pingo-120M** (12 camadas, d_model 896, 14 heads, SwiGLU hidden 2304, ctx 2048).

> ⚠️ O roadmap do `README.md` (itens 8–11) ainda está na **ordem antiga** (350M antes de
> SFT). A ordem vigente é a desta tabela: **SFT primeiro**.

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
