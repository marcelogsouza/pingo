---
tags: [pingo, sft, recusas, juiz]
etapa: sft-2
atualizado: 2026-10-30
---

# Juiz de recusas

```bash
uv run python scripts/validate_sft.py --judge-recusas
```

Uma chamada **extra** ao mesmo professor, só para as ~5% de recusas. É o único caso em que o
dataset precisa de segunda opinião para existir.

## O problema

A recusa é a única classe **sem verificação automática**:

| Tipo | Verificação automática |
|---|---|
| `extracao`, `explicacao`, `resumo` | ✅ `evidencia` tem de ser cópia literal do trecho |
| **`recusa`** | ❌ formato passa, evidência vazia passa… e a pergunta pode ser **respondível pelo trecho** |

O professor, ao inventar uma pergunta "cuja resposta não está no trecho", às vezes erra e
pergunta algo que o trecho **responde** — e escreve `"Isso não consta no texto."` de qualquer
forma.

> [!danger] O que isso ensina ao modelo
> Uma recusa errada ensina o aluno a **recusar o que está no texto**. Se ~1 em cada 10 recusas
> está errada, e as recusas são 5% do dataset, ~0,5% dos exemplos ensinam o comportamento
> oposto ao desejado — e é justamente o comportamento que a gente quer instalar.

## O caso real que motivou

Achado no dataset real, na auditoria:

> **Pergunta:** "Qual é o valor total investido?"
> **Resposta gerada:** "Isso não consta no texto."
> **Trecho:** "…já investiu R$ 21.056.242,25…"

A resposta **estava** no trecho. Formato perfeito, evidência vazia, e o par ensinaria o modelo a
errar.

## Como o juiz funciona

```python
"É uma segunda chamada, com temperature 0.0, ao MESMO professor:"
TRECHO: <chunk_text>
PERGUNTA: <pergunta>
→ {"responde": "sim" | "parcial" | "nao"}
```

| Veredito | Ação |
|---|---|
| `sim` | ❌ **descarta** o exemplo (`recusa_respondivel`) |
| `parcial` | ✅ mantém (é plausível recusar o que só está parcialmente) |
| `nao` | ✅ mantém |
| erro de rede/timeout (`None`) | ✅ **mantém** (não se pune o dataset por falha de infra) |

- `temperature: 0.0` e `max_tokens: 64` → resposta curta e determinística.
- JSON garantido por `json_schema` + `strict` ([[O professor local]]).
- 8 threads (`--judge-workers`), timeout 180 s.

## Custo

Uma chamada por recusa: ~1.500 chamadas para 30.000 pares (5%). A ~0,5 s cada com 8 slots, dá
**~2 minutos**. Comparado às ~11 h de geração, é ruído — e o benefício é remover a única fonte
sistemática de "mentira" no dataset.

## Onde os números aparecem

Em `data/sft/final/stats.json`:

```json
"juiz_de_recusas": {
  "ligado": true,
  "checadas": 1493,
  "descartadas_respondiveis": 63,
  "erros": 0
}
```

E cada descarte aparece em `motivos` como `recusa_respondivel`.

> [!tip] Valide o juiz com um controle
> Os testes (`test_filtrar_recusas_descarta_as_respondiveis`,
> `test_judge_ligado_derruba_recusa_respondivel`) usam um professor fake. Na prática, a
> verificação é: pegue uma recusa **sabidamente** respondível (como o caso acima) e confirme que
> o juiz a marca — e pegue uma recusa sabidamente **correta** e confirme que passa.

## Ver também

- [[Validação, dedup e splits]] — onde o juiz se encaixa
- [[Avaliação SFT]] — o `--judge` do `eval_sft.py` (papel parecido, objetivo diferente)
- [[Geração do dataset]] — quem produz as recusas
- [[Pingo]] — voltar ao índice
