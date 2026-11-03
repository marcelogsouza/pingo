---
tags: [pingo, dados, qa, llm, juiz]
etapa: 14
atualizado: 2026-11-03
status: desenho fechado, implementação pendente
---

# QA do dataset com LLM (o Juíz)

Segunda camada de qualidade para o **OSCAR filtrado** (`data/raw/*.parquet`, 10,7M docs): além
das réguas determinísticas do `process_data.py` (comprimento, linhas curtas, símbolos, dedup),
um **modelo de linguagem julga a qualidade** — e o resultado vira um **dataset novo**
(`data/oscar-qa/`) no contrato JSONL do [[Interface de treino|/treino]].

> [!warning] A conta que decide o desenho
> 10,7M docs × ~5,8k chars ≈ **1,8k tokens/doc → ~21B tokens de entrada** para o juíz.
> Medido no servidor local (Flash Next, llama.cpp + MTP): prompt **165 tok/s**. Mesmo um 35B
> A3B otimista a 1.000–4.000 tok/s de prompt: **2,5–9 meses de GPU exclusiva**. Juíz em tudo é
> inviável; juíz em **amostra** (3–5k docs ≈ ~11M tokens ≈ meia noite de 35B) é trivial.
> Por isso o juíz é **professor, não varredor**: ele rotula amostra, um classificador barato
> escoreia o corpus inteiro.

## O juiz

- **Modelo**: Qwen 3.6 35B A3B (`~/llm_serving_scripts/llama_qwen3_6_35b_a3b_q4_k_m_mtp.sh`),
  servido em `:8001`. Quem sobe o servidor é **o usuário** (regra de sempre: nunca derrubar o
  Strata/professor por conta própria).
- **Trava de identidade** (nasceu de um quase-erro real): o script lê `GET /v1/models` e **recusa
  rodar** se o id servido não for o esperado — hoje o servidor responde `qwen3.8-flash-next-iq3_s`,
  e um `--modelo` desatento julgaria o corpus com o modelo errado para sempre.
- **Rubrica** (`rubrica: v1`, versionada — mudar a rubrica muda a chave do cache): JSON estrito,
  `temperature 0.1`, doc truncado a 4.000 chars com `truncado: true` no registro:

  ```json
  {"pt": 0-3, "coerencia": 0-3, "conteudo": 0-3, "ruido": 0-3,
   "veredito": "fica"|"joga", "motivo": "≤12 palavras"}
  ```

- **Cache idempotente**: `data/judge/oscar.jsonl`, uma linha por doc, chave = `norm_key`
  (o MESMO blake2b do `process_data.py`). Re-rodar não re-cobrança; parar/retomar a qualquer
  momento; **um escritor** (lock de pid no `.lock` — a regra do projeto).

## A corrente (os três scripts)

```
scripts/judge_sample.py   amostra estratificada p/ o juíz
scripts/judge_score.py    chama o juíz (OpenAI-compat), escreve o cache
scripts/gen_qa_dataset.py classificador + limiar → dataset novo
```

1. **Amostragem** (`judge_sample.py`): o espaço dos *candidatos* é o que passou nas réguas duras
   (JSON/`content`/dedup/comprimento). Estratos = tercis de comprimento × passou/reprovou nas
   heurísticas macias (linhas curtas, símbolos) → **6 baldes**, ~800 docs cada = **~4,8k docs**.
   As reprovadas pelas heurísticas macias importam: é ali que está o falso-negativo do filtro atual.
2. **Juíz** (`judge_score.py`): roda pela amostra, alimenta o cache. Job do [[Interface de treino|
   /treino]] (`JobRunner`, tipo `qa_llm`) — fila, log e "parar" valem. Roda sozinho, sem webui.
3. **Calibração**: relatório de **discordância juíz × heurística** (~200 exemplos, CSV) para o
   usuário ajustar a régua na mão; métricas: concordância por balde, quanto do corpus o juíz
   aprova que a heurística reprovava (e vice-versa).
4. **Destilar** (`gen_qa_dataset.py`): classeificador **CPU-puro** (TF-IDF de unigramas +
   bigramas hash, N features, + as features das heurísticas + comprimento; logística com SGD —
   nada de sklearn pesado) treinado nos ~4,8k rótulos do juíz, validação 5-fold com F1 e
   precisão-alvo. Escoreia os 10,7M em minutos (`--n-process`).
5. **Dataset novo**: `data/oscar-qa/final.jsonl` (+ `val.jsonl` pelo mesmo hash de split do
   `process_data.py`) com `meta.json`-receita: modelo do juíz, versão da rubrica, seed da
   amostra, limiar, contagem aprovada. Entra direto na aba **Tokenização** do /treino
   (`prepare_tokens --input jsonl --jsonl data/oscar-qa/final.jsonl`).

## Como o pré-treino usa (duas opções documentadas, decisão fica pro run)

- **Re-treino limpo**: tokeniza só o `oscar-qa` (≈ metade do corpus se o limiar cortar ~50% →
  menos tokens que o run OSCAR atual — a época/`train-tokens` ajusta).
- **Replay ponderado** (recomendado p/ SFT-5 e continuation): tokeniza o corpus inteiro mas o
  loader prioriza shards dos aprovados — fica para quando o classificador existir.

## Limites aceitos (honestidade primeiro)

- O classificador destilado **não vê o que o juíz viu** — ele imita; vale para *reranking e
  corte grosso*, não para garantir qualidade doc a doc.
- Truncar a 4k chars ignora o rabo de docs longos (mediana 5,8k chars — a maior parte cabe).
- Rubrica v1 pode perdoar texto "liso" (Wikipédia rasa) e punir estilo oral; a planilha de
  discordância existe exatamente para calibrar isso antes do veredito final.

## Ver também

- [[Decisões (ADR)]] — ADR-17 (juíz professor, não varredor)
- [[Interface de treino]] — o contrato JSONL de chegada e o JobRunner
- [[Filtro de qualidade]] — as heurísticas que o juíz vai julgar também
