---
tags: [pingo, tokenizer, bpe]
etapa: 3
atualizado: 2026-10-30
---

# Tokenizer BPE byte-level

Script: `scripts/train_tokenizer.py` → artefatos em `tokenizer/` (`tokenizer.json`,
`tokenizer_config.json`, `vocab_stats.json`).

> [!abstract] Resultado
> BPE **byte-level**, vocabulário **16.384**, treinado em **1,0 GB** (179.619 docs do
> `split=train`, 39 s, pico 3 GB de RAM). Especiais: `<|bos|>` (0), `<|eos|>` (1),
> `<|pad|>` (2). Compressão no val: **4,091 chars/token** — ver [[Métricas de compressão]].

## Por que BPE byte-level (e não word-piece, não word-level)

O modelo precisa de uma função $\text{texto} \to \mathbb{Z}^V$ **sem `[UNK]`** e **invertível**.
Existem três famílias clássicas:

| Esquema | Problema em português |
|---|---|
| word-level | vocabulário explode; qualquer neologismo/erro de digitação vira `[UNK]` |
| word-piece (subword com `##`) | resolve, mas exige normalização e ainda tem `[UNK]` |
| **BPE byte-level** | **nenhum `[UNK]`**: os 256 bytes cobrem *qualquer* sequência UTF-8; roundtrip exato |

No byte-level, o texto é primeiro codificado em bytes UTF-8 e cada byte é mapeado para um
caractere "visível" (o mapa `bytes_to_unicode` do GPT-2, que é o motivo de você ver `Ġ` no lugar
de espaço e `Ã©` no lugar de `é`). O BPE roda *sobre esses bytes*.

$$
\text{texto} \xrightarrow{\text{UTF-8}} \text{bytes} \xrightarrow{\text{bytes\_to\_unicode}}
\text{símbolos} \xrightarrow{\text{merges BPE}} \text{tokens} \xrightarrow{\text{decode}} \text{texto}
$$

E o roundtrip é **exato**, sem perda — é o que `tests/test_tokenizer.py::test_roundtrip_e_byte_fallback`
garante.

## O algoritmo (o que "BPE" faz de verdade)

BPE começa com o vocabulário = todos os símbolos (256 bytes) e repete, até atingir o tamanho
alvo:

1. conte todos os **pares adjacentes** de símbolos no corpus;
2. escolha o par mais frequente $(a,b)$;
3. crie um token novo que representa $ab$ e substitua todas as ocorrências.

Formalmente, se o corpus tem $n$ símbolos, cada merge reduz o comprimento da sequência em
$c(a,b)$ (a contagem do par), então o custo total de compressão é

$$
\text{comprimento final} = n - \sum_{\text{merges}} c(\text{par escolhido})
$$

e o algoritmo ganancioso maximiza essa redução passo a passo. É por isso que os primeiros
merges são coisas como `Ġd`+`e` → `Ġde` (as palavras funcionais do português): eles aparecem
milhões de vezes.

> [!note] Por que os merges fazem sentido em pt
> Os 20 tokens mais frequentes no val são exatamente as palavras funcionais e a pontuação —
> `,`, `Ġde`, `Ċ` (newline), `.`, `Ġe`, `Ġa`, `Ġo`, `Ġque`, `Ġdo`, `Ġpara`, `Ġem`, `Ġda`, …
> Um vocabulário saturado de português é isso: um punhado de sílabas/palavras muito comuns e
> o resto bytes. Tabela completa em [[Métricas de compressão]].

## Configuração exata (e por que cada peça está lá)

```python
tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                   trim_offsets=False, use_regex=True)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=16_384,
                              special_tokens=["<|bos|>", "<|eos|>", "<|pad|>"],
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                              min_frequency=2,
                              limit_alphabet=vocab)
```

| Peça | Escolha | Razão |
|---|---|---|
| `models.BPE` | `unk_token=None`, `byte_fallback=True` | sem `[UNK]`; qualquer byte sempre tem representação |
| `pre_tokenizer` | `ByteLevel(use_regex=True)` | aplica a **regex estilo GPT-2** antes do BPE (ver abaixo) |
| `add_prefix_space=False` | não injeta espaço no começo | evita `Ġ` artificial na primeira palavra do prompt |
| `trim_offsets=False` | offsets não importam | não usamos `offset_mapping` |
| `initial_alphabet` | os 256 bytes | garante que **todo** byte existe antes de qualquer merge |
| `min_frequency=2` | merge só com ≥2 ocorrências | impede tokenizar *hapax legomena* (vocab poluído) |
| `limit_alphabet=vocab` | trava o alfabeto inicial | proteção: 256 bytes nunca estouram 16k |

A regex do pré-tokenizador (`use_regex=True`) é a do GPT-2:

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Ela separa contrações, palavras, números, pontuação e espaços — **antes** de qualquer merge.
Isso impede que o BPE aprenda tokens atravessando a fronteira de palavras ("casa de" virando um
token só), o que pioraria a generalização.

## A amostra de treino: 1 GB, `split=train`, por row group

`build_sample()` percorre os parquets em **ordem embaralhada** (`seed=13`) e escreve arquivos
de texto de ~128 MB (`sample_000.txt`, …). Filtra `split == "train"` **no row group**, sem
carregar o parquet inteiro — a mesma disciplina de RAM do [[Filtro de qualidade]].

Resultado real:

```json
{"sample_docs": 179619, "sample_chars": 1000001992, "sample_files": 8}
```

> [!important] Nunca amostrar o `val` para treinar o tokenizer
> O `val` tem de ser "texto que o modelo nunca viu" na régua de perplexidade. Se os merges
> fossem aprendidos nele, a ppl ficaria otimista. Por isso `iter_docs(..., split="train")` é
> obrigatório em `build_sample`, e o `val` só é usado na **avaliação** de chars/token.

## Artefatos gerados

| Arquivo | Conteúdo |
|---|---|
| `tokenizer.json` | o tokenizer completo (vocab + merges + pre-tokenizer + decoder) |
| `tokenizer_config.json` | `bos/eos/pad`, `model_max_length: 2048`, `tokenizer_class` |
| `vocab_stats.json` | métricas de compressão + top-20 tokens ([[Métricas de compressão]]) |

Salvos via `PreTrainedTokenizerFast(...).save_pretrained(tokenizer/)` — o mesmo formato que
`AutoTokenizer.from_pretrained` e `Tokenizer.from_file` entendem.

> [!warning] `tokenizer/` é gitignored — e 16.384 é uma decisão congelada
> O vocabulário **não** pode ser re-gerado sem invalidar todos os `.bin` do Passo 4. O SFT
> adiciona 3 tokens **em memória** (16.387) e salva uma cópia junto do run
> ([[Treino SFT]]); `tokenizer/tokenizer.json` continua com 16.384 para sempre.

## Ver também

- [[Métricas de compressão]] — o resultado medido
- [[Binarização e shards]] — onde o tokenizer é aplicado
- [[Gotchas]] — `BpeTrainer` não aceita `workers`/`chunk_size`
- [[Pingo]] — voltar ao índice
