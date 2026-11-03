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
├── pyproject.toml            # uv; deps: torch, transformers, tokenizers, datasets, numpy, tqdm,
│                             #     pyarrow, zstandard, flask, safetensors
├── data/                     # raw (parquet) + tokens (.bin uint16 + meta.json) + sft (jsonl) — fora do git
├── tokenizer/                # tokenizer.json treinado por nós + vocab stats
├── pingo/                    # pacote python
│   ├── config.py             # PingoConfig(PretrainedConfig)
│   ├── model.py              # RMSNorm, RoPE, Attention(+KV cache), MLP, PingoLM(PreTrainedModel)
│   ├── cache.py              # PingoCache
│   ├── generate.py           # sampling (temp, top-k, top-p, greedy) + `generate_stream`
│   ├── data.py               # DataLoader memmap sobre shards .bin
│   ├── sft.py                # contrato do exemplo de SFT, tokens de chat, heurística de pt
│   └── didatic.py            # modo didático: tokenização categorizada + embeddings na CPU
├── web/lab.html              # a página do modo didático (`/lab`)
├── scripts/                  # download_data, train_tokenizer, prepare_tokens, train, sample,
│                             #   gen_sft, validate_sft, train_sft, eval_sft, webui
├── docs/                     # vault Obsidian: como cada coisa foi feita e por quê
└── tests/                    # shapes, params, KV cache, overfit, SFT, WebUI, modo didático
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
9. **Mid-training / SFT** — dataset de instruções pt (ex.: datasets pt-BR de instruções no HF). ✔ *feito em 2026-10-30/31: SFT-1..4 + 30k pares; o run que vale é `runs/sft-120m-ep1`.*
10. **RLHF** — começar com **DPO** (dados de preferência pt-BR já existem, ex.: `Hub-Ai/ptbr-human-preferences`); PPO + reward model como opção avançada.
11. **Avaliação formal** — benchmarks pt (ex.: tarefas estilo NeurIPS-BR-LP, hellaswag-pt).
12. **Modo didático (`/lab`)** — mostrar o modelo por dentro, etapa por etapa, na mesma WebUI. **Passos ① ✔, ② ✔, ③ ✔, ④ ✔ e ⑤ ✔ entregues** (① tokenização: chips coloridos com id, superfície byte-level, bytes, a ficha do token com os 896 números e os vizinhos por cosseno; ② embedding: a frase como matriz desenhada, histograma das normas da tabela inteira com extremos, e a "lida de volta" `v·W` lado a lado com o cosseno — o stepper virou abas). ✔ Critério dos passos ①–②: rodar **sem GPU** (lê `embed.weight` do `.safetensors` na CPU) e responder com o professor servindo na VRAM. Restante da esteira, cada uma com um forward que devolve os intermediários:
    - **② embedding** — ✔ 2026-11-01: o id → vetor (matriz), as normas de todos os vetores, e a prévia do `lm_head` tied (produto escalar ordena diferente do cosseno porque a nota inclui a norma).
    - **③ RoPE + atenção** — ✔ 2026-11-01 (requisito do Marcelo cumprido): `capture` no
      forward devolve `q_lin/q/k_lin/k` e os pesos `w` por camada (sem captura, forward bit a bit
      idêntico — teste); no `/lab` são três cards: **relógio do RoPE** (32 ponteiros do Q real,
      par `(xⱼ,xⱼ₊₃₂)`, slider de posição + ponto verde de conferência com o Q capturado),
      **radar do token** (1 cabeça por linha × n tokens, clique abre a sparkline da cabeça) e a
      **matriz n×n com a máscara causal visível**, linha do token contornada. Forward real no
      device do servidor sob `gen_lock`, teto de 64 tokens (0,66 s/11 tokens na CPU).
    - **④ RMSNorm + SwiGLU** — ✔ 2026-11-01: o resto do bloco, na **carona do forward do ③**
      (o `capture` ganhou `x0/n1p/n1/a/x1/n2p/n2/g/u/s/m/x2` + os γ das normas; sem captura o
      forward é bit a bit idêntico — teste). `/api/lab/bloco` + três cards: **régua do RMSNorm**
      (‖x‖ → RMS puro = √d para todos → com γ → Δ da camada), **γ desenhado** (faixa alternável
      attn/MLP com média/min/max/σ) e **SwiGLU** (gate, up e o produto na mesma escala + portas
      abertas e Δ vs resíduo). Trocar camada = 12 ms (captura em cache).
    - **⑤ lm head** — ✔ 2026-11-01: a **norm final** e os **logits reais** pegam carona no
      forward do ③ (o último dict do `capture` ganhou `nfinp/nfin`; o `Server` guarda
      `out.logits` [64×16.387] — nada é recomputado). Quatro cards: **última régua** por token
      (‖x₂‖ 28,8–57,7 → 29,93 para todos → 46–48 com γ), **γ final desenhado** (média 1,61 —
      o botão de volume do produto escalar), **16.387 logits** (strip do vocab + top-10 +
      posto e nota do sucessor real: `até` em 5º de 16.387) e **a mesma tabela lida duas
      vezes** (prévia ② × logit real ⑤). Tie provado: `logits == nfin @ tabelaᵀ` bit a bit
      (maxdiff 0,0) — teste na suíte. 16 ms por posição (captura em cache, a mesma do ③/④).
    - **⑥ amostragem** — ✔ 2026-11-01: a **roleta** — `menu_amostra` no `didatic` aplica a
      mesma conta do `sample_next` do chat (temperatura → top-k → top-p → `multinomial`, com a
      penalidade de repetição sobre a história toda) e devolve o menu com **graus** (360° × p),
      rank, logit e `p_bruto`; o cardápio (24 fatias + a cinza "outros", soma = 360°, teste) é
      clicável: **forçar** obedece até um token cortado pelo filtro (fatia 0°). Girar é
      reproduzível com semente; o card **desdobramento** continua a história com
      `generate_stream` (KV cache, prévia do ⑦) até `<|end|>` ou o teto. Rotas
      `/api/lab/amostra|desdobrar` — stateless, carona na captura do ③–⑤ (teto subiu para 160
      tokens, `MAX_AMOSTRA_TOKENS`; girar/sliders são cache-hit, 2–35 ms na CPU). **Bug pego e
      consertado aqui**: o top-p do `sample_next` acumulava **logits**, não probabilidades —
      com logit positivo cortava tudo menos o favorito (padrão `top_p=1` não era afetado; runs
      e relatório não mudam; teste do nucleus na suíte).
    - **⑦ loop** — ✔ 2026-11-01: o **caderno do `generate_stream`**. `roda_loop` no `didatic`
      roda o loop do chat de verdade (mesma semente/penalidade/`sample_next` — teste confere
      token a token) e anota por step: `entrou` (o escolhido anterior — **o token volta**),
      `no_forward` (prefill = a frase; decode = 1), `escrito`/`cache_antes`/`cache_depois`/
      `kv_lidos`/`sem_cache`. A **fita** (`PingoCache.snapshots()`; colunas antigas congeladas,
      teste no `test_model`) mostra ‖K‖/‖V‖ por camada e posição; a **economia** conta
      tokens-forward (29 vs 407 medidos) e o **`compara`** refaz o último step sem cache:
      38,7 ms vs 20,6 ms com logits idênticos (maxdiff 4,8×10⁻⁶). Rota `/api/lab/loop`
      (tetos 160 ids + 64 novos `MAX_LOOP_NOVOS`; `parada:contexto` quando o contexto manda) +
      aba no `/lab` com replay stateless (o navegador anima; scrubber/⏯), fita acendendo uma
      coluna por step e barras de preço. 0,65 s o request de 8+22 tokens na CPU. **A esteira
      ①–⑦ fechou — não há mais etapa pendente no lab.**
    - Critério geral: nada de GPU até a etapa ② (foi o caso de ① e ②); a partir da ③, forward com captura de intermediários e aviso ao usuário antes de subir qualquer coisa na VRAM. ✔ Os ③, ④, ⑤, ⑥ e ⑦ entregues cumprem o critério (`capture` + `gen_lock` + tetos de 64/160/64, **um único forward** para ③–⑤ e o primeiro menu do ⑥; o ⑦ roda o loop real sob o mesmo lock; com o professor servindo, `--device cpu`).

13. **Interface de treino (WebUI `/treino`)** — orquestrar a esteira inteira numa aba da WebUI existente (:8010): apontar para um dataset JSONL qualquer, QA, tokenização, pré-treino e SFT, com monitor de job + HW. Motivador imediato: re-treinar com o dump 2026 da Wikipédia pt (`dumps.wikimedia.org/ptwiki/latest/`: `pages-articles.xml.bz2` 2,79 GB, 1.187.058 artigos, estimativa ~1–1,5 B tokens limpos — menos da metade dos 2,75 B do run OSCAR; wiki-puro vs. mistura com replay fica para quando o corpus for decidido). Decisões fechadas na conversa de 2026-11-02: mesma app Flask (`/treino`), UI **orquestra os scripts existentes** (nada de segundo loop de treino), **um job por vez** (escritor único + VRAM; chat bloqueado e modelo descarregado durante job), **tokenizer vira artefato por experimento** (`tokenizer/<id>/`; `meta.json` dos tokens amarra tokens↔tokenizer; config deriva `vocab_size` dele — erro se divergir; o default 16.384 fica intocado). Passos:
    - **treino-1** — parada graciosa: `SIGTERM/SIGINT` em `train.py`/`train_sft.py` → termina o step, grava ckpt + `{"evento":"interrompido"}` no `log.jsonl`, `--resume` continua. ✔ Critério: teste `--tiny` + sinal → ckpt íntegro → resume reproduz a curva.
    - **treino-2** — `scripts/qa_dataset.py`: contrato JSONL (1 linha = 1 JSON com `content` string não-vazia) → relatório estruturado (linhas boas/ruins, distrib. de comprimento, duplicatas pelo hash blake2b do `process_data.py`, amostras, avisos style `process_data.py`). CLI + biblioteca para a UI. ✔ (2026-11-02)
    - **treino-3** — `pingo/jobs.py` (JobRunner): subprocess dos scripts, fila 1-por-vez, registro persistente `runs/jobs.json`, stdout em anel + `log.jsonl` parseado via SSE; mutex com o chat; monitor HW (`psutil` + `nvidia-smi --query-gpu`, ~2 s, anel) — CPU/RAM/disco/VRAM. ✔ (2026-11-02; desvio: o log chega ao navegador por **polling** de `/api/treino/log?depois=<cursor>`, não SSE — mesmo anel, contrato mais simples)
    - **treino-4** — página `/treino` com quatro abas: **Dataset** (JSONL + relatório do QA), **Tokenização** (dois caminhos: usar vocabulário existente — dropdown do registry com a ficha de cada tokenizer — ou treinar um do zero), **Treino** (formulários pré-treino/SFT — lr, warmup, épocas→`--max-steps` do `meta.json`, batch/accum, `--out` com nome único e proteção contra sobrescrever run existente; iniciar/retomar/parar), **Runs** (histórico com val loss/ppl/duração/comando + curva de loss ao vivo + HW + log). ✔ (2026-11-02; a curva é SVG desenhado no front a partir de `/api/treino/runs/<nome>`, e `pt-120m`/backup são barrados por nome — 403)
    - **treino-5** — tokenizers múltiplos de ponta a ponta: `train_tokenizer.py --vocab N --specials ...` → `tokenizer/<id>/` (especiais: bos/eos/pad obrigatórios, tokens de chat opcionais já embutidos; corpus-amostra + seed configuráveis); ao fim do treino do zero, ficha de QA do tokenizer novo (compressão no val vs. default); `prepare_tokens.py --tokenizer` grava o id no `meta.json`; `train.py`/`train_sft.py --tokenizer` derivam o vocab (erro se divergir do config). ✔ (2026-11-02; desvio: a ficha compara compressão **vs. GPT-2** no mesmo texto — `vocab_stats.json` com `our_chars_per_token`/`gain_vs_gpt2_pct`, tolerante a offline — em vez de vs. o default; `train.py` **deriva** o vocab do `meta.json` em vez de erro, o meta nunca perde)
    - **depois** — `scripts/ingest_wikipedia.py` (XML → `content`, limpeza estilo wikiextractor) cai na aba Dataset; re-treino wiki fica com decisão de corpus documentada.

14. **QA do OSCAR com LLM (o Juíz → dataset novo)** — segunda camada de qualidade sobre `data/raw/*.parquet` (10,7M docs), com o Qwen 3.6 35B A3B (servidor `llama_qwen3_6_35b_a3b_q4_k_m_mtp.sh`, sobe/desce = usuário). Desenho fechado em 2026-11-03 (conversa + ADR-17 + [[QA do dataset com LLM]]): **juíz professor, não varredor** — 21B tokens de entrada p/ julgar tudo = 2,5–9 meses de GPU (medido: prompt 165 tok/s no servidor atual), então o LLM rotula **amostra estratificada ~4,8k docs** (tercis de comprimento × passou/reprovou nas heurísticas macias) e um **classificador CPU-puro** (TF-IDF hash + features das heurísticas + logística) destilado escoreia os 10,7M em minutos. Contratos: rubrica v1 JSON `{pt,coerencia,conteudo,ruido 0-3, veredito, motivo}` a T=0.1, trunc 4k chars; cache idempotente `data/judge/oscar.jsonl` chaveado por `norm_key` (blake2b), lock de escritor único, job do JobRunner; **trava de identidade** — o script confere `/v1/models` e recusa rodar com o modelo errado (hoje :8001 serve o Flash Next, não o 35B); planilha de discordância juíz×heurística (~200 docs) antes do limiar final; saída `data/oscar-qa/{final,val}.jsonl` + `meta.json`-receita no contrato do /treino (`prepare_tokens --input jsonl`). Passos:
    - **qa-llm-0** — `scripts/scan_raspa.py`: pente-fino determinístico (regex das marcas medidas em 2026-11-03: boilerplate 34%, cookie 11%, URL nua, rabo truncado, JS, wrapper EN + heurísticas análogas) sobre os 10,7M — corta a raspa de web **de graça, antes de gastar juíz**; relatório por marca + tamanho do resto. Medido: ~45% dos aprovados têm ≥1 marca; os indetectáveis (marketing liso, página-índice) ficam para o juíz.
    - **qa-llm-1** — `scripts/judge_score.py`: cliente OpenAI-compat com trava de id + rubrica v1 + cache idempotente (norm_key, lock, resume) + `--amostra N` aleatório p/ smoke. Critério: rodar 2× não re-cobrança; interromper e retomar continua; modelo errado = erro claro, nada roda.
    - **qa-llm-2** — `scripts/judge_sample.py` (6 baldes, ~800/balde) + relatório de discordância (CSV ~200) e métricas por balde. Critério: você revisa a planilha e ajusta a rubrica (v2) com números na mão.
    - **qa-llm-3** — `scripts/gen_qa_dataset.py`: TF-IDF hash + features + logística (SGD, CPU, sem sklearn), 5-fold F1 vs. rótulos do juíz; escoreia o corpus; limiar escolhido na calibração; escreve `data/oscar-qa/` + meta. Critério: tokenizar o novo dataset no /treino e ver o val ppl do run comparar com o run OSCAR atual.
    - **depois** — replay ponderado (loader prioriza shards aprovados) só se o run com o corpus limpo justificar.

14b. **Re-treino na Wikipedia PT** (pivô de 2026-11-03: "vamos utilizar o dataset em português do Wikipedia pra treinar novamente e ver no que dá. Depois voltamos ao OSCAR"). Corpus: `TucanoBR/wikipedia-PT` (HF, campo `text`, cc-by-sa-3.0, 1.103.446 artigos) — não o dump XML que o Passo 13 imaginava. Números e contratos em [[Dataset Wikipedia PT]].
    - **wiki-1** — `scripts/ingest_wikipedia.py`: HF → `content` com as MESMAS réguas do OSCAR (`process_data.py`) + split por hash **na origem** (`pt.jsonl` train, `pt-val.jsonl` val). ✔ (2026-11-03; desvio: `--short-line-ratio` default **0,9** para wiki — a régua 0,6 do OSCAR retinha só 33% dos artigos, medido; `qa_dataset.py` ganhou a mesma flag)
    - **wiki-2** — QA + binarização no tokenizer default 16.384 (vocab igual ⇒ embeddings/`runs/pt-120m` compatíveis). ✔ 836.266 docs train / 17.093 val → **650.248.473 + 13.117.734 tokens** em `data/tokens/wiki-16k/`. **Bug pego e consertado:** o `prepare_tokens` re-fatiava por hash um JSONL já fatiado ⇒ val vazio (o sintoma apontava para o hash, não para o pipeline); agora detecta o irmão `-val.jsonl` e **aborta** se o val sair vazio — teste na suíte.
    - **wiki-3** — pré-treino na wiki pelo `/treino`: **(a)** do zero, 1 época = 1.240 steps (~1,8 h); **(b)** continuando o `runs/pt-120m` com `--resume` **copiado** para `runs/pt-120m-wiki` (nunca escrever no `pt-120m`) — com `max_steps` explícito (5.248 + 1.240 = 6.488). Critério: os dois disparados pela interface, com o Strata desligado antes (combinado com o usuário). Os dois payloads prontos estão em [[Dataset Wikipedia PT]].
    - **wiki-4** — comparar com números: ppl no val da wiki, a régua de sempre (`eval_sft.py`) e a amostra gerada a olho. Critério de aceite: dizer se **663M tokens limpos** batem os **2,75B tokens sujos** no português do Pingo — e aí decidir o híbrido com o OSCAR destilado do Passo 14.

## Decisões fechadas ✔

- Porte Fase 1: **Pingo-120M** (12 camadas, d=896, 14 heads) — treina em ~6–10 h, escalável a 350M depois
- MLP: **SwiGLU** · Vocab: **16k** · Contexto: **2048** · Dataset: **Community OSCAR `pt`** (gate auto, sem filtro de variante)

## Últimos detalhes p/ fechar antes do Passo 1

- [ ] Preparar os dados (OSCAR + tokenizer + binários) **na M4** e levar só os `.bin` p/ a RTX? (recomendado: M4 baixa/tokeniza, RTX só treina)
- [ ] Onde ficam os checkpoints na RTX (disco local) e se queremos subir o modelo final ao HF Hub (repo privado?)
