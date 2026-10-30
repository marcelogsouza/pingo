---
tags: [pingo, arquitetura, cache, inferencia]
etapa: 5
atualizado: 2026-10-30
---

# KV cache

Código: `pingo/cache.py` (`PingoCache`) + o "terceiro modo" da atenção em `pingo/model.py`.
Testes: `test_cache_equivalencia_*`, `test_cache_buffers_pre_alocados`,
`test_cache_overflow_e_steps_multiplos`.

## O problema que ele resolve

Gerar 32 tokens sem cache custa, para cada token novo, **reprocessar o prompt inteiro**:

$$
\text{sem cache: } \sum_{t=1}^{n} O\big((T_0+t)^2 d\big) \approx O\big(n\,T^2 d\big)
\qquad
\text{com cache: } O\big(T^2 d\big) + \sum_{t=1}^n O\big(T d\big)
$$

A parte $QK^\top$ é a que dói: sem cache, o **passado é recalculado** a cada token, com
atenção quadrática sobre ele. Com cache, o passado está em memória e cada token novo custa
$O(T d)$.

## Como está implementado

```python
class PingoCache:
    _k, _v: list[Tensor | None]   # por camada, alocados no 1º update
    _cur_len, _pending
```

- Buffers **pré-alocados** de shape `[B, H, max_seq, D]` por camada, criados no primeiro
  `update()` (quando se sabe device, dtype e batch). Pré-alocar evita realocar/concatenar a
  cada token — o `torch.cat` por step seria o gargalo.
- Escrita na fatia `[:, :, cur_len : cur_len+t]`;
- leitura devolve a **visão** `[:, :, :cur_len+t]`, que é o que o SDPA vê.

### O protocolo `cur_len` / `commit()` (e o bug que ele evita)

```python
# dentro de cada Attention.forward
k, v = cache.update(layer_idx, k, v)     # escreve, NÃO avança

# dentro de PingoLM.forward, depois do loop de blocos
cache.commit()                            # avanço único
```

> [!bug] Se cada camada incrementasse o contador...
> ...a camada 2 escreveria **deslocado** (em `cur_len+1`), e o cache ficaria com um "buraco"
> correspondente ao token do prompt. O avanço é por *forward*, não por camada — daí o par
> `update` (escreve) / `commit` (avança) e o `_pending`.

### Restrições deliberadas

| Situação | Aceita? | Por quê |
|---|---|---|
| Prefill com cache vazio (`t == kv_len`) | ✅ `is_causal=True` | um forward causal normal |
| Decode de **1 token** (`t == 1`) | ✅ `is_causal=False` | a query vê todo o passado, sem máscara |
| Prefill parcial (`t > 1` com cache cheio) | ❌ `ValueError` explícito | exigiria máscara deslocada (chunked prefill); não é necessária |
| `cache` + `attention_mask` | ❌ `ValueError` | cache é inferência de batch 1, sem padding |
| `cur_len + t > max_seq` | ❌ `ValueError` "cache cheio" | o `generate` também para antes disso |
| batch diferente do alocado | ❌ `ValueError` | buffers são `[B,...]` fixos |

O cache **não propaga gradiente** (é inferência; copiar amostras para buffers pré-alocados já
está fora do grafo). Ele é usado sob `@torch.no_grad()` em `generate()` e no benchmark.

## Quanto de memória isso custa

Por sequência e por token:

$$
2\,(\text{K,V}) \times L \times H \times D \times 2\ \text{B}
= 2 \cdot 12 \cdot 14 \cdot 64 \cdot 2 = 43\,008\ \text{B} \approx 42\ \text{KiB/token}
$$

Para a configuração máxima (B=16, ctx 2048, bf16):

$$
16 \times 2048 \times 43\,008\ \text{B} = 1\,407\,379\,456\ \text{B} \approx 1{,}31\ \text{GiB}
$$

| ctx (B=16) | cache |
|---|---|
| 512 | ~336 MiB |
| 1024 | ~672 MiB |
| 2048 | ~1,31 GiB |

Isso é o "aluguel" que se paga em VRAM por gerar rápido — e o motivo pelo qual GQA existe
(14 heads × 64 dims é caro; com 2 KV-heads seria 7× menor).

## O ganho medido (e a lição sobre batch)

`runs/pt-120m/eval.json`, 120M, 32 tokens novos, greedy, mesmos tokens nas duas rotas:

| ctx | B | sem cache | com cache | speedup |
|---|---|---|---|---|
| 512 | 1 | 0,128 s (249 tok/s) | 0,152 s (211 tok/s) | **0,84×** ❌ |
| 1024 | 1 | 0,137 s (234 tok/s) | 0,138 s (232 tok/s) | **0,99×** |
| 2016 | 1 | 0,251 s (128 tok/s) | 0,139 s (231 tok/s) | 1,81× |
| 512 | 16 | 1,034 s (495 tok/s) | 0,179 s (2.864 tok/s) | **5,78×** |
| 1024 | 16 | 2,504 s (205 tok/s) | 0,236 s (2.166 tok/s) | **10,61×** |
| 2016 | 16 | 5,230 s (98 tok/s) | 0,443 s (1.157 tok/s) | **11,81×** |

> [!important] A lição: o KV cache não é sobre latência single-stream
> Em **B=1** o decode é limitado por **latência de kernel**, não por FLOPs: recalcular a
> sequência inteira é quase de graça, e o cache pode até **perder** (0,84× — a cópia para os
> buffers custa mais que o recálculo). O ganho aparece quando há paralelismo (B=16) ou contexto
> longo — que é o cenário de **serving**. O critério "10×+" do plano é sobre throughput de
> serving, e foi atingido a partir de ctx 1024.

> [!note] Por que os dois caminhos geram os mesmos tokens
> O benchmark usa greedy e compara `mesmos_tokens` (os 16 primeiros ids). Se a equivalência
> numérica quebrasse, o "speedup" seria ficção. Ela é garantida pelo teste de equivalência de
> logits ($\max|\Delta| < 10^{-5}$ em fp32) com prefill completo e com prefill + steps.

## Ver também

- [[Atenção causal]] — quem usa os buffers
- [[Geração (sampling)]] — o loop que tira proveito
- [[Perplexidade e KV cache]] — o benchmark no contexto do Passo 7
- [[Benchmarks na GPU]] — o resto dos números de GPU
- [[Pingo]] — voltar ao índice
