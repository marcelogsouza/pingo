<p align="center">
  <img src="logo.png" alt="Pingo — LLM em português do Brasil" width="320"/>
</p>

# 🐦 Pingo

**Uma LLM do zero, em português.**

Pingo é um projeto-curso para construir um modelo de linguagem pequeno do zero — da coleta de
dados ao texto gerado — sem atalhos e com cada passo entendido de verdade. Nada de chamar
`GPT2LMHeadModel` pronto: a arquitetura é escrita por nós em PyTorch, e o Hugging Face
Transformers entra como *scaffolding* (`save_pretrained`, `from_pretrained`, Hub).

## O modelo

Decoder-only, arquitetura própria:

| | |
|---|---|
| Atenção | Multi-head causal com **RoPE** (head_dim 64) |
| MLP | **SwiGLU** |
| Normalização | RMSNorm (pré-norm) |
| Embeddings | tied (sem embedding de posição) |
| Contexto | 2048 tokens |
| Vocabulário | BPE byte-level 16k, **treinado por nós** |
| Geração | `generate()` próprio com **KV cache** desde o dia 1 |

Configs previstas:

| Config | Params | Tokens | Uso |
|---|---|---|---|
| **Pingo-120M** | ~125M | ~2.5B | Fase 1 — treino em ~1 noite (RTX PRO 4500) |
| Pingo-350M | ~335M | ~6B | Fase 2 — escalar depois |

## Os dados

Pré-treino em português com o **Common Crawl filtrado por idioma** do Hugging Face. Registro
exato de onde vêm os dados (para qualquer pessoa reproduzir do zero):

| | |
|---|---|
| Dataset | [`oscar-corpus/community-oscar`](https://huggingface.co/datasets/oscar-corpus/community-oscar) — *gated* com gate automático: basta aceitar os termos 1× na página com uma conta HF |
| Dump usado | **`2024-10`** — o dump mais recente que traz o idioma `pt` (os dumps mais novos só publicam os metadados de `pt`) |
| Arquivos | `data/2024-10/pt/pt_part_1..100.jsonl.zst` — 100 shards independentes, ~42,7 GB (JSONL com `content` + `warc_headers` + `metadata`) |
| Reserva | dump `2023-40` (68 parts, ~41,8 GB), se a tokenização não alcançar a meta de ~2,5B tokens |
| Local | `data/<dump>/pt/*.jsonl.zst` (gitignored) → filtrado em `data/raw/*.parquet`; contagens em `data/raw/stats.json` |

O pt-BR domina naturalmente o corpus; não tentamos separar variantes. Filtros de qualidade
locais (comprimento do doc, ratio de linhas curtas, ratio de símbolos, dedup exato) seguram
~2% dos documentos para validação.

**Resultado do processamento (dump 2024-10):** 19,3M docs → **10,7M aprovados (55,6%)**,
com 1,9M duplicados exatos removidos; **59,7B caracteres** (≈15B tokens estimados — folga de
6× sobre a meta de 2,5B da Fase 1, o dump reserva `2023-40` não será necessário). Parquet
filtrado: 23 GB em `data/raw/`. Números completos em `data/raw/stats.json`.

Para reproduzir numa máquina nova (download retomável, pode rodar em partes):

```bash
uv run hf auth login        # token HF; aceite o gate do dataset na página dele 1x
./scripts/download_data.sh  # Windows: scripts\download_data.bat
```

Validação segurada e binarização estilo nanoGPT (shards `.bin` com mmap) para o loop de
treino ficar simples e rápido.

## Tokenizer

BPE **byte-level** (vocab **16.384**, regex estilo GPT-2 + byte fallback), treinado em amostra
de 1 GB do `split=train` (`scripts/train_tokenizer.py`), especiais `<|bos|> <|eos|> <|pad|>`.
No val segurado (5M chars): **4,09 chars/token** vs 2,69 do GPT-2 → **34% menos tokens** em
pt. Artefatos em `tokenizer/` (fora do git), métricas em `tokenizer/vocab_stats.json`.

## Tokens (shards binários)

`scripts/prepare_tokens.py` aplica o tokenizer aos parquets → **2,75B tokens de treino**
(2,02M docs amostrados do `split=train`, seed 13) em 2.752 shards `train-NNNN.bin` de 1M
tokens (uint16, docs separados por `<|eos|>`) + **292M tokens de val** (o `split=val`
completo, 6 shards). `meta.json` registra contagens/proveniência; `pingo/data.py` lê batches
`[B, ctx]` via `np.memmap` (alvo = entrada deslocada de 1). ~6,1 GB em `data/tokens/`
(foras do git).

## Roadmap

- [x] **1.** Setup do projeto com `uv`
- [x] **2.** Download e limpeza dos dados (Community OSCAR `pt`)
- [x] **3.** Treinamento do tokenizer BPE (16k) — byte-level, **4,09 chars/token** no val (GPT-2: 2,69)
- [x] **4.** Tokenização → shards binários train/val (uint16, memmap)
- [x] **5.** Modelo (RoPE, MHA, SwiGLU, KV cache) + testes unitários — **127.555.456 params**, equivalência do cache `<1e-5`, overfit→0, roundtrip HF
- [x] **6.** Loop de pré-treino (bf16, cosine, checkpoints retomáveis) — **2,75B tokens em 7,56 h** (~101k tok/s com `torch.compile`); val loss **9,69 → 2,507** (**ppl 12,3**), curva monótona, sem NaN; `--resume` bit-exato
- [x] **7.** Geração + perplexidade de validação — **ppl 11,67** no val (3,54 bits/token); texto pt coerente em parágrafos com amostragem; KV cache **10,6× (ctx 1024) / 11,8× (ctx 2016)** mais rápido com batch 16 e mesmos tokens
- [ ] 8. Escalar para ~350M
- [x] **9.** SFT — dataset sintético de 30k pares (professor Qwen3.5-4B local), treino de instrução e avaliação: **formato 100%, F1 0,375, fidelidade 0,59**, ppl do pré-treino 1,91×
- [ ] 10. SFT-5 (instruções gerais/replay de texto cru) e RLHF (DPO primeiro)
- [ ] 11. Avaliação em benchmarks pt

O plano detalhado, com decisões e critérios de aceite por etapa, está em [`PLAN.md`](PLAN.md).

## Estrutura

```
pingo/
├── PLAN.md
├── pingo/        # pacote: config, modelo, cache, geração, dados, didática
├── scripts/      # download, tokenizer, tokens, treino, amostragem, webui
├── web/          # página do modo didático (/lab)
├── docs/         # vault Obsidian: como cada coisa foi feita e por quê
├── tests/        # shapes, equivalência do KV cache, overfit
├── data/         # (gitignored) parquet + shards .bin
└── tokenizer/    # (gitignored) BPE treinado
```

## Rodando

Requisitos: Python 3.12+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Os scripts de dados/tokenizer rodam em qualquer máquina; o treino usa uma GPU NVIDIA
(testado com uma RTX PRO 4500 Blackwell 32 GB).

### WebUI (chat no navegador)

```bash
uv run python scripts/webui.py            # http://127.0.0.1:8010 e http://<ip-da-lan>:8010
uv run python scripts/webui.py --host 127.0.0.1   # só esta máquina
```

O host padrão é `0.0.0.0`, então dá para abrir de outra máquina da rede local; **não há
autenticação** — mantenha na LAN.

Uma página de chat em Flask que serve o **SFT** (`runs/sft-120m-ep1/final`) e o **base**
(`runs/pt-120m/final`): streaming token a token, contexto opcional (mini-RAG) e os
hiperparâmetros de amostragem. O prompt é exatamente o do treino
(`<|user|>Pergunta: …<|assistant|>`), então o que você vê na tela é o modelo no seu melhor
formato. Detalhes em [`docs/07-sft/WebUI.md`](docs/07-sft/WebUI.md).

### Modo didático — `/lab`

A segunda página da mesma UI, para o **curso**: o modelo por dentro, etapa por etapa. Os
passos ①, ② e ③ estão prontos — **tokenização**, **embedding** e **RoPE + atenção**:

- o prompt vira **chips coloridos** (especial, início de palavra, continuação, pontuação,
  número, byte puro), cada um com o **id** da tabela embaixo;
- clicando, abre a **ficha do token**: o texto humano ⇄ a superfície byte-level (`ĠpreÃ§o`),
  os bytes em hex, a norma, a faixa de calor dos **896 números** do embedding inicial e os
  **vizinhos por cosseno** — `Ġinvestir` mora ao lado de `Ġapostar`, `Ġgastar`, `Ġcomprar`;
- busca no vocab (`4560`, `ção` ou `Ã§Ã£o`) e o medidor `caracteres → tokens · chars/token`;
- no passo ② a frase vira uma **matriz desenhada** (uma linha por token, norma à direita),
  a tabela inteira vira um **histograma de normas** — as três menores são exatamente os
  tokens de chat — e a tabela é **lida de volta**: as notas que um `lm_head` tied daria do
  vetor inicial, lado a lado com o cosseno (a ordem muda: nota inclui a norma).

- no passo ③ o modelo faz um **forward real** e a tela mostra os intermediários capturados:
  o **relógio do RoPE** (os 32 ponteiros de um Q verdadeiro, girando em velocidades que caem de
  1 volta a cada ~6 posições até ~47 mil, com um ponto verde que confere a rotação contra o Q
  capturado), o **radar de atenção** do token escolhido (uma linha por cabeça, clique abre a
  cabeça em detalhe) e a **matriz n×n com a máscara causal visível**.

Os passos ①–② leem o `model.safetensors` na CPU e funcionam com qualquer coisa ocupando a VRAM.
O ③ carrega o modelo no device do servidor — respeita o lock do chat (devolve aviso se ele está
falando) e limita a frase a 64 tokens; com a VRAM ocupada, `--device cpu` serve igual (0,66 s o
forward de uma frase curta; na GPU livre são 0,64 s já com o load do modelo). Os passos ④–⑦
(RMSNorm/SwiGLU, lm head, amostragem, loop) aparecem
como "em breve" — o `capture` do ③ já entrega boa parte do que eles vão desenhar.
Detalhes em [`docs/07-sft/Modo didático.md`](docs/07-sft/Modo didático.md).

## Aviso

Projeto educacional. O corpus vem do Common Crawl e pode conter conteúdo impreciso,
ruidoso ou sensível; o modelo resultante reflete isso.
