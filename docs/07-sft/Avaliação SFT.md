---
tags: [pingo, sft, avaliacao, fidelidade]
etapa: sft-4
atualizado: 2026-10-30
---

# Avaliação SFT (SFT-4)

Script: `scripts/eval_sft.py` (+ `--judge`). Testes: `tests/test_eval_sft.py` (**12 testes**).
Gera nas perguntas do **`heldout`** (nunca visto no treino) e escreve
`runs/sft-120m/final/eval_sft.json`.

> [!abstract] As três perguntas, na ordem em que importam
> 1. **Ele responde no formato?** → `taxa_terminou` + F1 contra a referência.
> 2. **Ele é fiel?** → `apoio` lexical + veredito do professor (`--judge`).
> 3. **Ele esqueceu o que sabia?** → ppl no val do pré-treino **antes × depois**.

## 1 · Formato

| Métrica | Definição |
|---|---|
| `terminou` | a geração contém `<|end|>` (aprendeu o fim de turno) |
| `taxa_terminou` | fração dos exemplos com `terminou` |
| `f1` | F1 de palavras contra a `resposta` de referência |

$$
\text{F1} = \frac{2\,P\,R}{P+R},
\qquad P = \frac{|\text{multiconj}(gerado) \cap \text{multiconj}(ref)|}{|\text{gerado}|}
$$

É a métrica estilo SQuAD, calculada sobre **multiconjuntos** de palavras (minúsculas, só letras):

```python
pred, ref = Counter(words(gerado)), Counter(words(referencia))
comuns = sum((pred & ref).values())
```

> [!note] F1 é triagem, não veredito
> A resposta de referência é *uma* resposta possível. Um modelo que responde certo com outras
> palavras tira F1 baixo. Serve para detectar "não respondeu nada" e para comparar dois
> checkpoints — não para dizer "acertou".

## 2 · Fidelidade lexical (`apoio`)

$$
\mathrm{apoio}(g, t) =
\frac{\big|\{\,\mathrm{stem}(w) : w \in \text{conteúdo}(g),\ \mathrm{stem}(w) \in \mathrm{stems}(t)\,\}\big|}
{\big|\{\,\mathrm{stem}(w) : w \in \text{conteúdo}(g)\,\}\big|}
$$

onde $\mathrm{stem}(w)$ = **5 primeiras letras** e $\text{conteúdo}$ = palavras que não estão na
lista de stopwords em português (`PT_STOP`, ~5 dezenas de palavras funcionais).

É reportado **separado** por modo: `apoio_ancorado` e `apoio_sem_contexto` — no modo livre o
modelo não tem o trecho no prompt, então o número é uma medida de "ele lembra do domínio".

> [!warning] Métrica lexical não é prova de fidelidade
> O `_stem` de 5 letras foi um **bugfix real**: sem ele, `investe` × `investimentos` já derrubava
> o apoio. Mas o limite oposto existe: **paráfrase legítima** ("planeja" × "plano") também derruba
> o número, e **invenção gramatical** pode passar por acidente. Por isso:
> use `apoio` como **triagem**; o número que vale para "não inventou" é o `--judge`.

## 3 · Veredito do professor (`--judge`)

Uma chamada por exemplo (em 8 threads) com **temperature 0.0**, JSON estrito:

```json
{"apoio": "sim" | "parcial" | "nao", "motivo": "..."}
```

O prompt pede explicitamente: `sim` = fatos conferem com o TRECHO; `parcial` = parte confere;
`nao` = contradiz ou inclui fatos que não estão nele. O relatório traz a **distribuição**
(`{"sim": 0.82, "parcial": 0.11, "nao": 0.07}`), não só uma média.

## 4 · Recusas

```python
def recusa_ok(gerado, ancorado):
    alvo = RECUSA_ANCORADO if ancorado else RECUSA_LIVRE
    return normalize_ws(gerado).lower().startswith(alvo[:24])
```

Duas frases diferentes, uma por modo — e a métrica sabe qual esperar
([[Formato do exemplo]]).

## 5 · Esquecimento catastrófico

$$
\text{fator} = \frac{\text{ppl}_{\text{SFT}}}{\text{ppl}_{\text{base}}}
\quad\text{no mesmo } \texttt{val} \text{ do pré-treino}
$$

O script roda a mesma régua (`--ppl-iters 10`, `--ppl-batch 8`, `--ppl-ctx 1024`, rng
`20_260_101`) em dois modelos: o base e o SFT.

> [!danger] `fator > 1,3` = investigar
> É o sinal clássico de **esquecimento catastrófico**: o modelo aprendeu o formato de chat e
> perdeu a modelagem de linguagem. Sintomas típicos: ppl sobe muito, geração vira "frases de
> manual" e o F1 no heldout parece bom porque as respostas de referência também são curtas.
> Correções possíveis: lr menor, menos épocas, ou **misturar** um pouco de texto cru no treino do
> SFT (é o que o SFT-5 faria de outro jeito).

## Como rodar

```bash
uv run python scripts/eval_sft.py                       # SFT vs base, sem juiz
uv run python scripts/eval_sft.py --n 200 --judge       # + veredito do professor
uv run python scripts/eval_sft.py --model runs/pt-120m/final --no-ppl   # avaliar o base
```

| Arg | Default | Nota |
|---|---|---|
| `--n` | 200 | quantos exemplos do heldout gerar (0 = todos) |
| `--max-new-tokens` | 96 | |
| `--max-len` | 1024 | trunca o prompt (trecho + pergunta) |
| `--temperature` | **0.0** | greedy: reprodutível |
| `--repetition-penalty` | 1.15 | evita loop ([[Geração (sampling)]]) |
| `--judge` | off | liga o professor |
| `--show` | 5 | quantas gerações imprime no terminal |

## O relatório (`eval_sft.json`)

```
model, heldout, n, device, decodificacao, vocab
metricas: { n, taxa_terminou, f1, apoio_respondiveis,
            apoio_ancorado, apoio_sem_contexto, recusa_ok,
            por_tipo: {extracao|explicacao|resumo|recusa: {n, terminou, f1, apoio, recusa_ok}} }
amostras: [{pergunta, referencia, gerado, tipo, ancorado, f1}]
juiz:     {"sim"|"parcial"|"nao": fração}, juiz_n
ppl_pre_treino: {sft: {...}, base: {...}}, esquecimento: {ppl_antes, ppl_depois, fator}
```

> [!tip] Leia `por_tipo`, não só a média
> Um modelo pode acertar extração e falhar resumo; e `recusa_ok` só existe para 5% dos exemplos.
> A média geral esconde exatamente o que se quer ver.

## Detalhe que já mordeu: limpar a geração

```python
def limpar_geracao(texto):
    texto = texto.replace(END_TOKEN, " ").replace("<|user|>", " ")...
```

`generate(stop_token=END)` **inclui** o `<|end|>` na saída. Duas medidas diferentes:

- `terminou` → medido no texto **cru** (contém o token?);
- `f1`/`apoio`/`recusa_ok` → medidos no texto **limpo** (sem os tokens de chat).

Coberto por `test_score_example_marca_fim_de_turno` e
`test_limpar_geracao_tira_tokens_de_chat`.

## Ver também

- [[Treino SFT]] — o modelo avaliado
- [[Juiz de recusas]] — o juiz irmão, no dataset
- [[Perplexidade e KV cache]] — a régua do pré-treino
- [[Pingo]] — voltar ao índice
