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

Pré-treino em português com o [Community OSCAR](https://huggingface.co/datasets/oscar-corpus/community-oscar)
(Common Crawl filtrado por idioma). O pt-BR domina naturalmente o corpus; não tentamos
separar variantes. Validação segurada, filtros de qualidade e binarização estilo nanoGPT
(shards `.bin` com mmap) para o loop de treino ficar simples e rápido.

## Roadmap

- [ ] **1.** Setup do projeto com `uv`
- [ ] **2.** Download e limpeza dos dados (Community OSCAR `pt`)
- [ ] **3.** Treinamento do tokenizer BPE (16k)
- [ ] **4.** Tokenização → shards binários train/val
- [ ] **5.** Modelo (RoPE, MHA, SwiGLU, KV cache) + testes unitários
- [ ] **6.** Loop de pré-treino (bf16, cosine, checkpoints retomáveis)
- [ ] **7.** Geração de texto + perplexidade de validação
- [ ] 8. Escalar para ~350M
- [ ] 9. Mid-training / SFT
- [ ] 10. RLHF (DPO primeiro)
- [ ] 11. Avaliação em benchmarks pt

O plano detalhado, com decisões e critérios de aceite por etapa, está em [`PLAN.md`](PLAN.md).

## Estrutura

```
pingo/
├── PLAN.md
├── pingo/        # pacote: config, modelo, cache, geração, dados
├── scripts/      # download, tokenizer, tokens, treino, amostragem
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

## Aviso

Projeto educacional. O corpus vem do Common Crawl e pode conter conteúdo impreciso,
ruidoso ou sensível; o modelo resultante reflete isso.
