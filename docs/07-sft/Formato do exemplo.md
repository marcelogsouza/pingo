---
tags: [pingo, sft, formato, jsonl]
etapa: sft-1
atualizado: 2026-10-30
---

# Formato do exemplo

Uma linha de JSONL por par. Tudo que o treino precisa está na linha — inclusive o texto do
trecho, para que a serialização do treino possa mudar **sem regerar o dataset**.

## Os campos (v2)

| Campo | Tipo | Quem usa | Observação |
|---|---|---|---|
| `doc_id` | str | validador, splits | id do parquet = blake2b do conteúdo ([[Filtro de qualidade]]) |
| `url` | str | teto por domínio, auditoria | pode ser vazia |
| `split` | `"train"` | validador | **tem de ser `train`**: o val do pré-treino não pode vazar |
| `dump` | str | procedência | `2024-10` |
| `chunk_idx` | int | dedup exata `(doc_id, chunk_idx)` | 0..2 |
| `chunk_chars` | int | sanity | = `len(chunk_text)` |
| `chunk_text` | str | treino (modo ancorado), verificação | o trecho literal |
| `teacher` | str | procedência | `unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL` |
| `prompt_version` | `"v2"` | procedência | muda quando o prompt muda |
| `seed` | int | reprodutibilidade | semente enviada ao professor |
| `attempt` | int | auditoria | 0..2 |
| `tipo` | enum | taxonomia | `extracao`/`explicacao`/`resumo`/`recusa` |
| `ancorado` | bool | **serialização** | `true` = leva o trecho no prompt |
| `pergunta` | str | treino | termina em `?` |
| `resposta` | str | treino + referência de avaliação | 10–900 chars |
| `evidencia` | str | verificação | **cópia literal** do trecho; vazia nas recusas |

## Exemplo real (ancorado)

```json
{
  "doc_id": "9c1b7a3f...", "url": "https://exemplo.com.br/noticia/123",
  "split": "train", "dump": "2024-10", "chunk_idx": 0, "chunk_chars": 1987,
  "chunk_text": "O prefeito anunciou nesta terça-feira que o investimento total ...",
  "teacher": "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL", "prompt_version": "v2",
  "seed": 1883524912, "attempt": 0, "tipo": "extracao", "ancorado": true,
  "pergunta": "Qual é o valor total que será investido?",
  "resposta": "O investimento total será de R$ 21.056.242,25.",
  "evidencia": "O investimento total será de R$ 21.056.242,25, segundo a prefeitura."
}
```

## Exemplo real (sem contexto)

```json
{
  "tipo": "explicacao", "ancorado": false,
  "pergunta": "Por que o prazo do concurso foi estendido?",
  "resposta": "O prazo foi estendido por causa da baixa procura de inscrições.",
  "evidencia": "A decisão foi tomada após a baixa adesão ..."
}
```

> [!note] `chunk_text` e `evidencia` continuam gravados **mesmo no modo livre**
> O treino pode ignorá-los (e ignora: o prompt é só `Pergunta: ...`), mas a **avaliação** precisa
> do trecho como fonte de verdade (`--judge`, `apoio`) — ver [[Avaliação SFT]].

## Como o exemplo é serializado para o modelo

`pingo/sft.py::render()`:

```python
# ancorado
prompt   = "<|user|>Contexto: {chunk_text}\n\nPergunta: {pergunta}<|assistant|>"
resposta = "{resposta}<|end|>"

# livre
prompt   = "<|user|>Pergunta: {pergunta}<|assistant|>"
resposta = "{resposta}<|end|>"
```

Visualmente (ancorado):

```
<|user|>Contexto: O prefeito anunciou nesta terça-feira que o investimento total ...
Pergunta: Qual é o valor total que será investido?
<|assistant|>O investimento total será de R$ 21.056.242,25.<|end|>
^^^^^^^^^^^^                                                                       ^^^^^^^^
     tokens do prompt → -100 na loss                            tokens da resposta → loss
```

Três detalhes de projeto:

1. **O prompt termina em `<|assistant|>`** — é o último token previsto pelo prompt, e é dele que
   o modelo começa a prever a resposta ([[Treino SFT]]).
2. **A resposta termina em `<|end|>`** — o "fim de turno" que o modelo aprende a **emitir**. A
   avaliação mede a taxa de `terminou` exatamente nesse token.
3. **Recusa não leva contexto extra**: a resposta é uma frase fixa por modo, então o modelo
   aprende duas posturas distintas (`Isso não consta no texto.` quando há contexto;
   `Não tenho informações suficientes...` quando não há).

## Por que `evidencia` é obrigatória

É a **única verificação automática barata** de que o professor não inventou. Não é perfeita
(uma frase verdadeira pode não responder à pergunta — daí o juiz de recusas e o `--judge`), mas:

- pega paráfrase disfarçada (comparação é literal, tolerando só whitespace);
- pega resposta cujo "apoio" não está no trecho;
- **não custa nada**: a chamada ao professor já ia acontecer.

> [!warning] Nunca remova a `evidencia` do schema
> Sem ela, o dataset passa a depender exclusivamente da boa-fé do professor. É a linha de
> defesa mais barata que existe.

## A região perigosa: "respondível" vs "não respondível"

| Situação | `evidencia` | Verificação |
|---|---|---|
| extração/explicação/resumo | deve existir e ser literal | automática ✅ |
| recusa | deve ser **vazia** | juiz do professor ([[Juiz de recusas]]) |

Uma recusa com evidência preenchida é rejeitada na hora (`recusa_com_evidencia`). Uma recusa
cuja pergunta **o trecho responde** só o juiz pega — e é por isso que ele é obrigatório.

## Ver também

- [[Geração do dataset]] — quem produz a linha
- [[Validação, dedup e splits]] — quem revalida a linha
- [[Treino SFT]] — quem consome a linha
- [[Pingo]] — voltar ao índice
