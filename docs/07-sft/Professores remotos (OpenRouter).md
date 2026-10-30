---
tags: [pingo, sft, professor, openrouter, comparacao]
etapa: sft
atualizado: 2026-10-30
---

# Professores remotos (OpenRouter)

Script: `scripts/pilot_teacher_compare.py`. Saídas: `data/sft/or-{pilot,pilot40,final}/`.
**Custo total do experimento: US$ 0,07.**

> [!abstract] Conclusão
> Nenhum professor remoto foi adotado. O melhor em qualidade de Q&A (`deepseek-v4.1-flash`) é
> **pior em vetar spam** — aceita página de cassino e de SEO. Trocar o professor local
> injetaria lixo no dataset, que é justamente o que o veto do Qwen protegia.

## O método (para ser comparável)

- **Mesmos 60 trechos** para os 3 modelos, amostrados do corpus com seed própria (não toca no
  run principal nem no llama.cpp).
- **Mesmo prompt** do gerador (`build_body` importado de `gen_sft.py`, via `importlib`) e
  **mesmo validador**.
- 32 requisições em voo (janela deslizante).
- `response_format` = `json_schema` + `strict` em todos; `reasoning: {"enabled": false}`.
- 1 linha por **chamada** no JSONL (aceita ou não) → dá para auditar cada rejeição.

> [!note] `chat_template_kwargs` é do llama.cpp
> No OpenRouter esse campo tem de sair do body; o equivalente é `reasoning: {"enabled": false}`.
> Com isso, `reasoning_tokens = 0` nos três — nenhum deles gastou budget pensando.

## Resultados

| modelo | aceitos/chunk | chamadas/chunk | tok/par (in/out) | ~22k pares | vazão @32 |
|---|---|---|---|---|---|
| `qwen/qwen3.7-flash` | 75% | 1,68 | 1.625 / 247 | ~2,0 h · **US$ 1,79** | 3,07 par/s |
| `deepseek/deepseek-v4.1-flash` | **97%** | **1,08** | 971 / 189 | ~2,6 h · US$ 3,06 | 2,37 par/s |
| `google/gemma-4-26b-a4b-it` | 77% | 1,48 | 1.562 / 284 | ~2,9 h · US$ 5,00 | 2,14 par/s |

Medido nos 60 trechos (`or-final/*.stats.json`):

| modelo | aceitos | chamadas | evidência não literal | fonte inválida (veto) | `menciona_fonte` |
|---|---|---|---|---|---|
| deepseek-v4.1-flash | **58** | 65 | 3 | **3** | 1 |
| gemma-4-26b-a4b-it | 46 | 89 | **33** | 10 | 0 |
| qwen3.7-flash | 45 | 101 | 17 | **34** | 5 |

## Os quatro achados

1. **Quanto melhor a prosa, pior o veto de spam.** O deepseek tem a melhor qualidade de Q&A
   (resposta de 178 chars, zero metalinguagem no modo livre) e vetou só **3** trechos contra
   **34** do qwen. Ele aceitou página de cassino ("Qual é o provedor do slot Lucky Little
   Gods?") e página de SEO. É uma "máquina de sempre responder": sem uma instrução de veto mais
   dura (ou um passe extra de spam), trocá-lo **injeta casino/SEO no dataset**.
2. **Gemma parafraseia em vez de citar**: 33 de 89 tentativas caíram em
   `evidencia_nao_literal` — **⅓ dos tokens desperdiçados**. O campo `evidencia` é o que expõe
   isso; sem ele, o dataset pareceria ótimo.
3. **Qwen veta demais** (34 vetos): perde notícia legítima e texto de lei — mas também é o que
   mantém spam fora. Também vazou "texto/trecho" no modo livre em 5 casos.
4. **O speedup real é modesto (2–3×), não ordens de grandeza.** ~2–3 h em vez de ~11 h — e o
   gargalo passa a ser latência/rate limit do provedor, não computação.

## Por que a comparação foi feita (e por que não mudou nada)

O objetivo era responder: *"vale pagar US$ 4–7 para gerar os 30k com um modelo melhor, em vez
de ~11 h de GPU local?"* A resposta medida foi **não**, por um motivo que não estava no cálculo:
o **veto de qualidade** é a parte mais valiosa do gerador local, e o professor remoto "melhor"
é o que menos veta.

> [!tip] Se um dia for para valer
> O coerente é **regerar os 30k inteiros com UM professor** (US$ 4–7 pelo conjunto), não
> misturar com os 8,7k do local. Professor misturado = dois estilos, duas taxas de erro, e
> `teacher` diferente por linha complicando a análise.

> [!warning] Nenhuma credencial ficou no repositório
> A chave foi apagada de `~/.openrouter_key`; o script lê `OPENROUTER_API_KEY` do ambiente ou
> `--key-file`. Nunca commitar chave.

## Detalhe de implementação: importar um script como módulo

```python
_SPEC = importlib.util.spec_from_file_location("gen_sft", "scripts/gen_sft.py")
g = importlib.util.module_from_spec(_SPEC)
sys.modules["gen_sft"] = g          # ← obrigatório: @dataclass quebra sem isso
_SPEC.loader.exec_module(g)
```

Sem registrar em `sys.modules`, o `@dataclass` de `gen_sft.py` estoura
`AttributeError: 'NoneType' object has no attribute '__dict__'` — a mesma pegadinha dos testes
([[Gotchas]]).

## Ver também

- [[O professor local]] — o escolhido
- [[Geração do dataset]] — o pipeline comparado
- [[Decisões (ADR)]] — ADR-14
- [[Pingo]] — voltar ao índice
