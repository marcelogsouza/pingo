---
tags: [pingo, tokenizer, metricas]
etapa: 3
atualizado: 2026-10-30
---

# Métricas de compressão

Fonte: `tokenizer/vocab_stats.json`, produzido no fim de `scripts/train_tokenizer.py`.
**A régua é o `split=val`** — 5.001.128 caracteres que nunca entraram no treino do tokenizer.

## O número que fecha o critério do plano

$$
\text{chars/token} = \frac{\#\text{caracteres}}{\#\text{tokens}}
$$

| Tokenizer | Tokens para os mesmos 5.001.128 chars | chars/token |
|---|---|---|
| **Pingo BPE 16.384 (pt)** | **1.222.454** | **4,091** |
| GPT-2 (50.257, treinado em inglês) | 1.861.191 | 2,687 |

$$
\text{ganho} = 1 - \frac{1\,222\,454}{1\,861\,191} = 0{,}3432 \;\Rightarrow\; \textbf{34,3\% menos tokens}
$$

> [!tip] O que isso significa na prática
> Cada token é uma unidade de computação. 34% menos tokens para o mesmo texto = **34% menos
> passes de treino**, 34% menos disco, 34% mais contexto efetivo (os mesmos 2.048 tokens de
> contexto cobrem ~8.400 caracteres em vez de ~5.500). O critério do `PLAN.md` ("taxa de
> compressão registrada e comparada com GPT-2") foi atingido com folga.

## Por que GPT-2 é uma régua justa (e por que perde tanto)

O GPT-2 tem um vocabulário **3× maior** e foi treinado em inglês. Mesmo assim perde em
português, por dois motivos:

1. **acentos**: `é`, `ã`, `õ`, `ç` são bytes multigota no UTF-8. Sem merges específicos, cada
   `ã` vira 2 tokens (`Ã` + `£` no mapa byte→unicode). O vocabulário do GPT-2 tem alguns
   desses merges por acaso (via inglês), mas pouquíssimos;
2. **morfologia**: `casa`, `casas`, `casinha`, `casamento` — pt tem sufixação produtiva; um
   BPE treinado no idioma aprende `Ġcasa` e reusa o radical.

## Os 20 tokens mais frequentes no val

| # | token | contagem | # | token | contagem |
|---|---|---|---|---|---|
| 1 | `,` | 44.039 | 11 | `Ġda` | 9.109 |
| 2 | `Ġde` | 40.652 | 12 | `Ġ(` | 8.761 |
| 3 | `Ċ` | 39.382 | 13 | `-` | 7.613 |
| 4 | `.` | 31.389 | 14 | `Ġcom` | 7.536 |
| 5 | `Ġe` | 21.752 | 15 | `)` | 7.060 |
| 6 | `Ġa` | 18.343 | 16 | `Ġum` | 6.166 |
| 7 | `Ġo` | 14.935 | 17 | `Ġno` | 6.148 |
| 8 | `Ġque` | 13.260 | 18 | `ĠÃ©` (= ` ã©`) | 6.140 |
| 9 | `Ġdo` | 12.020 | 19 | `:` | 5.807 |
| 10 | `Ġpara` | 10.719 | — | — | — |

O diagnóstico é bem claro: a lista é **palavras funcionais + pontuação**, exatamente o perfil
de um tokenizer bem treinado em texto natural. Se aparecessem HTML (`<div`, `&nbsp;`) no topo,
seria sinal de que o filtro deixou passar boilerplate.

## Leitura de `Ġ` e `Ã©`

`id_to_token()` mostra a representação **byte→unicode** do GPT-2:

- `Ġ` (U+0120) é o **espaço**;
- `Ċ` (U+010A) é a **quebra de linha**;
- `Ã©` é o par de bytes que, decodificado como UTF-8, é **`é`**.

É cosmético: `decode()` devolve o texto correto, byte a byte. Confundir isso com "tokenizer
quebrado" é o erro clássico — está documentado em [[Gotchas]] e coberto pelo teste de roundtrip.

## A conta de sanidade que fecha o corpus

Se o corpus tem 59,7B caracteres e a compressão mede 4,09 chars/token, o potencial é

$$
\frac{59\,737\,711\,111}{4{,}091} \approx 14{,}6 \times 10^{9}\ \text{tokens}
$$

o que bate com a estimativa grosseira de ~15B (que usava 4 chars/token na
[[Filtro de qualidade]]) e explica por que **sobra folga de ~5×** sobre a meta de 2,75B
([[Binarização e shards]]).

## Cuidados ao comparar números

> [!warning] chars/token não é constante
> Depende do domínio: notícia/ensaio comprime bem; código, tabela de preço e lista de nomes
> comprimem mal. O número de 4,091 é a média do **val** (web-pt filtrada), e é por isso que
> serve como régua: é sempre o mesmo texto, medido do mesmo jeito.

> [!warning] O val do OSCAR tem ruído
> Páginas de cassino/SEO passam no filtro do pré-treino ([[Escolha do dataset]]). Isso não
> afeta a comparação com o GPT-2 (mesmo texto dos dois lados grep), mas explica variações de
> ppl absoluta mais adiante ([[Perplexidade e KV cache]]).

## Ver também

- [[Tokenizer BPE byte-level]] — como foi produzido
- [[Binarização e shards]] — o next step, usando essa taxa
- [[Pingo]] — voltar ao índice
