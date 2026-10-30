---
tags: [pingo, sft, validacao, dedup, minhash, splits]
etapa: sft-2
atualizado: 2026-10-30
---

# Validação, dedup e splits (SFT-2)

Script: `scripts/validate_sft.py` + regras compartilhadas em `pingo/sft.py`.
Testes: `tests/test_validate_sft.py` (16 testes).

Entra `data/sft/sft-v1.jsonl` → saem `data/sft/final/{train,val,heldout}.jsonl` +
`stats.json` + `rejects.jsonl`.

> [!abstract] Desempenho
> **~1.300 exemplos/s** (30k em ~25 s). Em 3.386 linhas reais: **21 barradas** — 18 trechos
> espelhados (páginas de template e políticas de privacidade copiadas) e 3 perguntas quase
> idênticas.

## Por que revalidar o que o gerador já validou

> *"O dataset é artefato de longo prazo e o validador não deve confiar em quem escreveu o
> JSONL."*

O gerador pode ter rodado com outro `--tries`, o arquivo pode ter sido concatenado à mão, ou o
`--resume` pode ter deixado uma linha truncada. Além disso, **três checagens só fazem sentido
no conjunto inteiro** (e não linha a linha):

1. duplicata exata `(doc_id, chunk_idx)`;
2. quase-duplicata de pergunta (o gerador só faz dedup **exata**, em memória);
3. quase-duplicata de **trecho** (site espelho republicando o mesmo texto).

## Etapa 1 — revalidação (regras em `pingo/sft.py`)

`pingo/sft.py::example_reject_reason()` repete de propósito as checagens do gerador, num único
lugar compartilhado (o gerador importa de lá — nenhuma regra duplicada):

campos obrigatórios → `split == "train"` → `doc_id` não vazio → 200 ≤ `chunk_text` ≤ 6000 →
`tipo` válido → pergunta termina em `?` e tem 15–300 chars → resposta 10–900 →
`is_pt` → (modo livre) não citar a fonte → recusa no formato do modo e **sem** evidência →
**evidência literal** dentro do trecho → resposta não é cópia da pergunta.

Também tolera **linha truncada** na leitura (`read_jsonl`), porque o gerador pode estar
escrevendo no arquivo enquanto o validador roda.

## Etapa 2 — dedup quase-exata com MinHash + LSH

### O que é um shingle

Quebramos cada texto em **shingles de 3 palavras**, normalizados (minúsculas, só alfanumérico +
acentos) e **hasheados**:

$$
S(x) = \{\,h(w_i w_{i+1} w_{i+2}) \mid i = 1..n-2\,\}
$$

Jaccard entre dois conjuntos:

$$
J(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

### MinHash: estimar $J$ com 64 números

Para cada uma das $k$ funções de hash (permutações), a assinatura guarda o **mínimo** dos hashes
do conjunto:

$$
\mathrm{sig}_j(A) = \min_{s\in A} h_j(s)
$$

Propriedade central: $\Pr[\mathrm{sig}_j(A) = \mathrm{sig}_j(B)] = J(A,B)$. Logo, a fração de
coordenadas iguais é um estimador não-enviesado de Jaccard.

Aqui as permutações vêm de **hashing universal** sobre o primo de Mersenne $p = 2^{61}-1$:

$$
h_j(s) = (a_j\, s + b_j) \bmod p, \qquad a_j,b_j \sim U[0,p)
$$

Reprojetar 64 permutações de verdade seria caríssimo; essa família dá 64 hashes independentes
com duas multiplicações cada.

### LSH: não comparar todos com todos

As assinaturas são cortadas em **bandas** de $r$ linhas; se **uma** banda coincide, os dois
exemplos viram **candidatos** (e aí o Jaccard exato confirma). A probabilidade de virar
candidato em função da similaridade é

$$
P(s) = 1 - \big(1 - s^{\,r}\big)^{b}
$$

| Dedup | $k$ (perms) | bandas $b$ | linhas $r$ | limiar |
|---|---|---|---|---|
| **pergunta** | 64 | 16 | 4 | $J \ge 0{,}80$ |
| **trecho** | 32 | 8 | 4 | $J \ge 0{,}85$ |

$$
P_{q}(0{,}80) = 1-(1-0{,}8^{4})^{16} \approx 0{,}9998
\qquad
P_{c}(0{,}85) = 1-(1-0{,}85^{4})^{8} \approx 0{,}9973
$$

Ou seja: quase certeza de virar candidato acima do limiar — e, abaixo dele, muitos candidatos
são gerados (ex.: $P_q(0{,}5)\approx 0{,}64$) e **descartados pelo Jaccard exato**. O LSH só
reduz o número de comparações caras.

> [!bug] O bug do "espelho com 2 palavras trocadas" (Jaccard 0,38 em vez de 0,99)
> Para limitar custo em trechos longos (~500 palavras → ~500 shingles), é preciso **amostrar**
> shingles. A primeira tentativa amostrou **por posição** (`words[i::step]`, passo proporcional
> ao tamanho). Isso **desalinha** os dois textos: dois documentos quase idênticos passam a ter
> conjuntos de shingles diferentes, e um espelho com 2 palavras trocadas mediu **Jaccard 0,38**
> (deveria ser ~0,99).
>
> A correção foi amostrar **por hash**: `h % step == 0`. Como o hash de cada shingle não depende
> da posição, os **mesmos** shingles são selecionados nos dois textos — a similaridade medida
> volta a refletir a real. (Limites usados: 120 shingles para pergunta, 64 para trecho.)

> [!bug] Outras duas armadilhas do LSH
> - **O bucket tem de ser lista**, não um índice único: colisão de banda não pode se perder.
> - **Só se indexa o exemplo ACEITO.** Indexar antes do veredito deixa índices órfãos em
>   `signatures[i]` (o exemplo rejeitado some, mas o índice dele fica apontando para o nada e
>   contamina comparações futuras). Coberto por `test_lsh_nao_indexa_rejeitados_no_meio`.

## Etapa 3 — juiz de recusas (opcional, mas recomendado)

Ver [[Juiz de recusas]]. Roda **depois** do dedup e antes do split.

## Etapa 4 — split **por documento**, sem vazamento

```python
def split_of(doc_id, seed, val_frac=0.05, heldout_frac=0.05):
    h = int.from_bytes(blake2b(f"{seed}:{doc_id}".encode(), digest_size=8).digest(), "big")
    x = h / 2**64
    return "heldout" if x < 0.05 else ("val" if x < 0.10 else "train")
```

- **Por documento**: todos os exemplos de um `doc_id` caem no mesmo split. Se um artigo gerou
  3 chunks, os 3 ficam juntos — não há como o modelo ver a resposta no treino e ser avaliado no
  mesmo texto.
- **Determinístico**: mesmo `doc_id` + mesma `seed` → mesmo split, sempre (run reproduzível).
- **Checagem de vazamento**: depois de escrever, o script calcula as interseções
  `train∩val`, `train∩heldout`, `val∩heldout` e **aborta sem gravar** se qualquer uma for
  não-vazia.

| Split | Fração | Uso |
|---|---|---|
| `train` | 90% | treino |
| `val` | 5% | curva durante o treino (eval a cada 200 steps) |
| `heldout` | 5% | **intocado** até a avaliação final (`eval_sft.py`) |

A ordem dentro de cada arquivo é **embaralhada** com `random.Random(seed)` — para não formar
blocos por domínio (o gerador processa em round-robin, então a ordem crua é correlacionada com
a fonte).

## O que sai

```
data/sft/final/
├── train.jsonl      (≈90%)
├── val.jsonl        (≈5%)
├── heldout.jsonl    (≈5%)
├── stats.json       (funil, motivos, splits, parâmetros)
└── rejects.jsonl    (até 500 exemplos, com o motivo)
```

`stats.json` traz `entrada`, `linhas_lidas`, `aceitos`, `rejeitados`, `motivos`,
`quase_duplicatas`, `juiz_de_recusas`, `splits` (exemplos/docs/`exemplos_por_doc`),
`vazamento_docs` (tem de ser tudo 0) e `tipos`/`modos`.

## Ver também

- [[Formato do exemplo]] — o que é validado
- [[Juiz de recusas]] — a única classe com segunda opinião
- [[Treino SFT]] — quem consome `train`/`val`
- [[Avaliação SFT]] — quem consome `heldout`
- [[Pingo]] — voltar ao índice
