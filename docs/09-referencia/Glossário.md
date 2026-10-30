---
tags: [pingo, referencia, glossario]
atualizado: 2026-10-30
---

# Glossário

## Modelo e arquitetura

**Decoder-only** — arquitetura que só tem atenção causal (cada token vê só o passado). É a
família de GPT/Llama. [[Arquitetura do Pingo]]

**Pré-norm** — a normalização acontece *antes* de cada subcamada, e a saída dela é somada ao
residual: $x \mathrel{+}= \mathrm{Attn}(\mathrm{Norm}(x))$. [[RMSNorm]]

**RMSNorm** — normalização pela raiz da média dos quadrados, sem centrar na média.
[[RMSNorm]]

**RoPE** — *Rotary Position Embedding*: codifica posição **rotando** Q e K por um ângulo
proporcional à posição. [[RoPE]]

**SwiGLU** — MLP com portão: $\mathrm{down}(\mathrm{SiLU}(\mathrm{gate}(x)) \odot \mathrm{up}(x))$.
[[SwiGLU]]

**Tied embeddings** — `lm_head.weight is embed.weight`: uma única matriz serve para ler e
prever tokens. [[Inicialização e embeddings tied]]

**head_dim** — dimensão de cada cabeça de atenção (aqui 64). [[Atenção causal]]

**SDPA** — `F.scaled_dot_product_attention`, o kernel de atenção do PyTorch (escolhe
Flash/Mem-efficient automaticamente). [[Atenção causal]]

**KV cache** — guardar K e V já calculados para não reprocessar o passado a cada token novo.
[[KV cache]]

**prefill** — o primeiro forward de uma geração, que processa o prompt inteiro.
**decode** — cada token seguinte, de 1 em 1 com cache. [[KV cache]]

**ppl (perplexidade)** — $e^{\text{loss}}$. "Entre quantos tokens o modelo fica em dúvida, em
média". [[Perplexidade e KV cache]]

**bits/token** — $\text{loss}/\ln 2$. Métrica comparável entre tokenizers diferentes.

**esquecimento catastrófico** — o modelo perde a capacidade anterior (linguagem) ao aprender a
nova tarefa (formato de chat). Medido por ppl antes/depois. [[Avaliação SFT]]

## Dados e tokenização

**OSCAR** — corpus do Common Crawl filtrado por idioma. "Community OSCAR" é a versão
comunitária. [[Escolha do dataset]]

**shard (part)** — pedaço de um corpus no Hub (`pt_part_N.jsonl.zst`). [[Download do OSCAR]]

**BPE** — Byte-Pair Encoding: começa com símbolos e funde os pares mais frequentes.
[[Tokenizer BPE byte-level]]

**byte-level** — o BPE opera sobre bytes UTF-8, não sobre caracteres → nenhum `[UNK]`.
[[Tokenizer BPE byte-level]]

**byte fallback** — garantia de que qualquer byte sempre tem representação, mesmo que raro.

**chars/token** — taxa de compressão do tokenizer. [[Métricas de compressão]]

**shard `.bin`** — arquivo binário de tokens `uint16` (1M tokens). [[Binarização e shards]]

**memmap** — mapear arquivo em memória e ler como array, sem carregar tudo.
[[Binarização e shards]]

**epoch (época)** — uma passagem completa pelo conjunto de dados. Cuidado: aqui a amostragem
é **com reposição**, então "1 época" = 1× o número de tokens, não 1× os documentos únicos.

## Treino

**AdamW** — Adam com weight decay desacoplado. [[Loop de treino]]

**decay group** — grupo de parâmetros com/sem weight decay. Norm e embedding ficam **sem**.
[[Loop de treino]]

**warmup** — subida linear do lr nos primeiros steps. **cosine** — decaimento suave até o fim.
[[Loop de treino]]

**grad accumulation** — acumular gradientes de vários micro-batches para simular um batch maior.
[[Loop de treino]]

**grad clip** — limitar a norma global do gradiente (aqui 1.0) para não explodir.

**bf16** — bfloat16: 8 bits de expoente (mesma faixa do fp32) e 8 de mantissa (precisão menor).
Usado em autocast no treino.

**autocast** — fazer as operações em bf16 mantendo parâmetros master em fp32.

**`torch.compile`** — compila o grafo do modelo (fusão de ops). Aqui: ~2× mais rápido **e**
menos VRAM. [[Benchmarks na GPU]]

**checkpoint** — arquivo com pesos + estado do otimizador + RNGs + step.
[[Checkpoints e resume]]

**resume bit-exato** — retomar e obter pesos idênticos aos de um run contínuo.

**best.pt** — cópia do checkpoint com a melhor val loss até agora (só o SFT faz isso dentro do
script). Vigia do pré-treino era externo.

## SFT

**SFT** — *Supervised Fine-Tuning*: treinar em pares (instrução, resposta).
[[Por que SFT]]

**professor (teacher)** — modelo que **gera** o dataset. Aqui: Qwen3.5-4B local via llama.cpp.
[[O professor local]]

**aluno** — o Pingo-120M, que é treinado com o dataset do professor.

**ancorado** — exemplo em que o trecho vai junto no prompt (tipo mini-RAG).
**sem contexto / closed-book** — exemplo em que só a pergunta vai no prompt.
[[Formato do exemplo]]

**evidência** — cópia **literal** de 1–2 frases do trecho que sustentam a resposta. Única
verificação automática barata contra invenção. [[Formato do exemplo]]

**recusa** — exemplo cujo objetivo é ensinar "não sei". Duas frases, uma por modo.

**juiz** — segunda chamada ao professor para verificar o que não é verificável
automaticamente. [[Juiz de recusas]], [[Avaliação SFT]]

**apoio (support)** — fração das palavras de conteúdo da resposta gerada que aparecem no
trecho, comparando radicais de 5 letras. [[Avaliação SFT]]

**heldout** — 5% do dataset, **intocado** até a avaliação final.
[[Validação, dedup e splits]]

**MinHash / LSH** — assinatura que estima Jaccard e índice que evita comparar tudo com tudo.
[[Validação, dedup e splits]]

**shingle** — n-grama de palavras usado no dedup (aqui 3 palavras).

## Projeto e infra

**Strata** — o servidor de inferência que ocupa a GPU no dia a dia (~31 GB de VRAM). Não é
parte do Pingo, mas condiciona o cronograma. [[Infra e GPU]]

**vigia** — script externo que monitora `log.jsonl` e salva `best.pt` / manda SIGINT se o val
piora ou aparece NaN.

**handoff** — `AGENTS.md`: o estado vivo do projeto entre sessões.

**ADR** — *Architecture Decision Record*: decisão + contexto + alternativas.
[[Decisões (ADR)]]

## Ver também

- [[Pingo]] — voltar ao índice
- [[Comandos e cheatsheet]] — a prática, não a teoria
