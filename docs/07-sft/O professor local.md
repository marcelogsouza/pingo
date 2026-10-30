---
tags: [pingo, sft, professor, llama.cpp]
etapa: sft
atualizado: 2026-10-30
---

# O professor local

O dataset de SFT é **gerado** por um modelo maior rodando na mesma máquina, via `llama.cpp`.

| Item | Valor |
|---|---|
| Servidor | `llama.cpp` build **b11669** |
| Modelo | `unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL` (quantizado Q4_K_XL) |
| Endpoint | `http://localhost:8001/v1/chat/completions` (estilo OpenAI) |
| Slots | **8** concorrentes, ctx **4096 por slot** |
| VRAM | ~5 GB (deixa ~27 GB livres — mas ocupa ~85% da GPU **enquanto gera**) |
| Quem sobe | **o usuário** (o script de serving mora em `~/llm_serving_scripts/`, fora do repo) |

> [!warning] Não suba o servidor por conta própria
> E, principalmente: **não rode nada pesado em CUDA enquanto o professor serve** — ele precisa
> da GPU. Ver [[Infra e GPU]].

## O contrato HTTP (o que o gerador manda)

```python
body = {
  "model": "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL",
  "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
  "max_tokens": 400, "seed": seed,
  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user",   "content": "MODO: ...\nTIPO DE EXEMPLO: ...\n\nTRECHO:\n..."}],
  "response_format": {"type": "json_schema",
                      "json_schema": {"name": "qa", "schema": QA_SCHEMA, "strict": True}},
  "chat_template_kwargs": {"enable_thinking": False},
}
```

Os parâmetros de amostragem são os do *model card* em modo instruct.

## Três pegadinhas que custaram tempo (e que valem para qualquer llama.cpp)

> [!bug] 1 · `{"type": "json_object", "schema": {...}}` é **ignorado**
> O llama.cpp só garante JSON se receber o formato **completo**:
> `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}, "strict": true}}`.
> Com o formato "curto", a resposta sai em texto livre e o parser quebra — silenciosamente,
> como `json_invalido`.

> [!bug] 2 · Thinking ligado queima o budget inteiro
> Se o modelo entrar em modo raciocínio, o conteúdo vai em `message.reasoning_content`
> (separado de `content`) e consome `max_tokens`. Com `max_tokens=400`, o `content` volta
> **vazio**. Solução: `chat_template_kwargs: {"enable_thinking": false}` no servidor **e** por
> request.

> [!bug] 3 · O caso LFM2.5-2.6B (o professor anterior, descartado)
> O template dele **cravava** ` thinking` no prompt de geração, sem switch. Resultado medido:
> **1.661 tokens de raciocínio em inglês para ~180 de conteúdo** (~10×) — e `content` vazio
> quando o budget acabava. Foi o motivo da troca para o Qwen3.5-4B, que tem toggles de verdade.

## Desempenho medido (8 slots, Qwen3.5-4B UD-Q4_K_XL)

| Grandeza | Valor |
|---|---|
| Completion agregado | **160–175 tok/s** |
| Prompt tokens por chamada (trecho ~2.000 chars) | **~1.100** |
| Completion tokens por par | **~240** |
| Vazão de pares | **~40 pares/min** |
| Custo de 30k pares | **≈ 11–12 h** |

> [!note] Prefill é ~⅓ do tempo de cada request
> O trecho de 2.000 caracteres custa ~1.100 tokens de *prompt* (prefill). As alavancas, se
> precisar acelerar: **chunks menores** (menos prefill), ou **2 pares por chamada** (amortiza o
> prefill no dobro de completion). Aumentar `max_tokens` não ajuda: o gargalo é o prefill.

## Por que o professor é chamado de "professor"

Porque o que sai dele **não** é usado como texto: é usado como **rótulo** para treinar o aluno.
O pipeline descarta o texto bruto e guarda só o JSON (pergunta, resposta, evidência, tipo), com
procedência completa (`teacher`, `prompt_version`, `seed`, `attempt`).

Essa separação é o que permite, depois, **medir** a qualidade do professor ([[Juiz de recusas]],
[[Professores remotos (OpenRouter)]]) e até trocá-lo sem reescrever os scripts.

## O "veto" — o efeito colateral mais valioso

O schema inclui `status: "ok" | "invalido"`. Se o trecho não for texto legível em português
(menu, lista de links, texto embaralhado, propaganda, spam), o professor **recusa** e o trecho
é descartado.

Medido no piloto: **~48 vetos em ~330 trechos** (≈ 15%). É o filtro de qualidade mais eficaz do
pipeline — e custa zero, porque a chamada ia acontecer de qualquer jeito.

## Ver também

- [[Geração do dataset]] — como o professor é usado
- [[Professores remotos (OpenRouter)]] — a comparação que **não** o substituiu
- [[Infra e GPU]] — convivência com a GPU
- [[Gotchas]] — a lista completa
- [[Pingo]] — voltar ao índice
