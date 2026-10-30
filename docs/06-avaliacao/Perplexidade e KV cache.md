---
tags: [pingo, avaliacao, perplexidade, kv-cache]
etapa: 7
atualizado: 2026-10-30
---

# Perplexidade e KV cache (avaliação do base)

Script: `scripts/sample.py` → relatório em `runs/pt-120m/eval.json`.
Testes: `tests/test_sample.py` (6 testes).

O script faz **três** coisas: perplexidade no val, amostras qualitativas e benchmark do KV
cache. É a régua que fecha a Fase 1.

## 1 · Perplexidade no `split=val`

```bash
uv run python scripts/sample.py --out runs/pt-120m/eval.json
```

| Métrica | Valor |
|---|---|
| Loss média no val | **2,4571** |
| **ppl** $= e^{\text{loss}}$ | **11,67** |
| bits/token $= \text{loss}/\ln 2$ | **3,5449** |
| Tokens avaliados | 409.600 (25 batches × B8 × ctx 2048) |
| Semente | `20_260_101` (a mesma do treino) |

$$
\text{ppl} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i\mid x_{<i})\right)
= \exp(2{,}4571) = 11{,}67
$$

> [!warning] Há **duas** ppl do mesmo modelo, e as duas estão certas
> - **12,3** — a do `train.py` (20 batches × B16 × ctx 2048 = 655.360 tokens), logada no fim
>   do treino;
> - **11,67** — a do `sample.py` (25 batches × B8 = 409.600 tokens).
>
> O `ppl` depende de **quais janelas** caem no rng fixo: ~5% de variação entre amostragens.
> Comparações entre runs são válidas; valores absolutos têm essa incerteza.

### Por que ppl 11,67 é um número bom *neste* corpus

O `split=val` herdou o ruído web do OSCAR (páginas de cassino, SEO, texto espalhado). Num corpus
limpo (Wikipedia, notícias), um 120M com 2,75B tokens daria ppl bem menor. Aqui a régua serve
para **comparar runs**, não para "provar qualidade" — que é o papel das amostras e, depois, do
SFT.

Em **bits/token** (3,54) a comparação é mais justa com a literatura, porque não depende do
vocabulário: é literalmente quantos bits o modelo gasta para codificar cada token.

## 2 · Amostras qualitativas

```bash
uv run python scripts/sample.py --prompt "Era uma vez" --temperature 0.9 --top-p 0.95 \
    --no-bench --no-ppl
```

Duas amostras por prompt (greedy + amostragem), com `--repetition-penalty 1.15` por default.
O resultado está reproduzido em [[Run do Pingo-120M]]; o resumo:

| Modo | Comportamento |
|---|---|
| greedy | **loop de repetição** — típico de modelo pequeno; `repetition_penalty` resolve |
| amostragem | parágrafos coerentes, pontuação correta, nomes próprios inventados |

## 3 · Benchmark do KV cache

`bench_kv_cache()` compara, com os **mesmos tokens gerados** (greedy):

- `generate_naive`: reprocessa a sequência inteira a cada token → $O(ctx^2)$;
- `generate_cached`: prefill + 1 token/step → $O(ctx)$ por step.

| ctx | B | naive (s) | cache (s) | speedup | mesmos tokens |
|---|---|---|---|---|---|
| 512 | 1 | 0,128 | 0,152 | 0,84× | ✅ |
| 512 | 16 | 1,034 | 0,179 | **5,78×** | ✅ |
| 1024 | 1 | 0,137 | 0,138 | 0,99× | ✅ |
| 1024 | 16 | 2,504 | 0,236 | **10,61×** | ✅ |
| 2016* | 1 | 0,251 | 0,139 | 1,81× | ✅ |
| 2016* | 16 | 5,230 | 0,443 | **11,81×** | ✅ |

\* `ctx_ajustado: true` — o prompt foi cortado para `max_ctx - n_new` (2048 − 32 = 2016) porque
o cache não pode encher antes de gerar.

**Conclusão registrada no plano**: o critério "10×+" é sobre **throughput de serving** (B=16,
ctx ≥ 1024), não sobre latência single-stream. Em B=1 o decode é limitado por latência de
kernel e o cache pode até perder (0,84×). A discussão completa está em [[KV cache]].

> [!note] `mesmos_tokens` é a parte que importa
> Um "speedup" com tokens diferentes seria lixo. O benchmark guarda os 16 primeiros ids dos
> dois caminhos e compara. Aqui: idênticos em todas as linhas.

## Resolve o tokenizer do próprio modelo

```python
def resolve_tokenizer(model_dir, default_path):
    cand = Path(model_dir) / "tokenizer.json"
    return cand if cand.exists() else Path(default_path)
```

O modelo de SFT salva `tokenizer.json` (16.387 tokens) ao lado dos pesos. Sem essa função, o
`sample.py` avaliaria o SFT com o vocabulário de 16.384 e **decodificaria os tokens de chat
como bytes soltos**. Está coberto por `test_sample_resolve_tokenizer_do_modelo`.

## Ver também

- [[Run do Pingo-120M]] — o modelo avaliado
- [[KV cache]] — a implementação e a teoria
- [[Avaliação SFT]] — a avaliação seguinte (e o risco de esquecimento)
- [[Pingo]] — voltar ao índice
