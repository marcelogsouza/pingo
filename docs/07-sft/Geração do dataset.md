---
tags: [pingo, sft, dataset, gerador]
etapa: sft-1
atualizado: 2026-10-30
---

# Geração do dataset (SFT-1)

Script: `scripts/gen_sft.py` (~600 linhas) + `tests/test_gen_sft.py` (**21 testes**, professor
fake via `monkeypatch`, sem rede). Saída: `data/sft/sft-v1.jsonl` (meta **30.000** pares) +
`sft-v1.stats.json` + `sft-v1.log`.

> [!abstract] Resultado até agora
> ~**40 pares/min** com 8 slots (1,7–2,1 s/par); 30k pares ≈ **11–12 h**. Os pilotos (300
> pares cada) rodaram em **7,4 e 7,5 min** e foram auditados à mão.

## O pipeline por documento

```mermaid
flowchart TB
    A["parquet (split=train)<br/>round-robin de row groups"] --> B{"filtro de fonte<br/>(duro)"}
    B -->|"rejeita"| X["descartado"]
    B -->|"ok"| C["clean_paragraphs<br/>(boilerplate fora)"]
    C --> D["make_chunks<br/>≤2000 chars, sem cortar frase"]
    D --> E["até 3 chunks/doc → fila"]
    E --> F["8 threads → professor"]
    F --> G{"JSON válido?"}
    G -->|"não"| H["re-tenta (até 3x)"]
    G -->|"sim"| I{"status == invalido?"}
    I -->|"sim"| J["VETO: trecho é lixo"]
    I -->|"não"| K["validação local<br/>evidência literal, pt, tamanhos, dedup"]
    K -->|"falhou"| H
    K -->|"ok"| L{"domínio no teto?"}
    L -->|"sim"| M["descarta (dominio_no_teto)"]
    L -->|"não"| N["append no JSONL<br/>+ flush"]
    H --> F
```

## 1 · Filtro de fonte (duro, bem mais que o do pré-treino)

| Regra | Limiar | Motivo |
|---|---|---|
| tamanho do doc | $2000 \le n \le 60.000$ chars | texto curto não dá pergunta autossuficiente; doc gigante é coletânea |
| parágrafos úteis | **≥ 4** | menu/lista não tem prosa |
| dígitos por palavra | **≤ 0,25** | tabela de preço, planilha, listagem |
| maiúsculas | **≤ 30%** das letras | CAPS LOCK, título/post |
| links | **≤ 4** | página de links |
| **blocklist de URL** | porn, xxx, escort, casino, bet, tigrinho, slot, "assistir/dublado/legendado", megafilmes, warez… | spam que **parece** texto legível |
| **blocklist de texto** | "vídeos picantes", "conteúdo adulto explícito", "maiores de 18 anos" | idem |

Todos os limiares da tabela são flags (default entre parênteses): `--min-chars` (2000),
`--min-paras` (4), `--max-digits-per-word` (0,25), `--max-upper-ratio` (0,30), `--max-urls` (4),
`--deny-url-regex` / `--deny-text-regex` (as blocklists abaixo), `--domain-cap-frac` (0,05) e
`--domain-cap-min` (3). Os do chunking: `--chunk-chars` (2000), `--chunk-min-chars` (700),
`--max-per-doc` (3). Os da geração: `--workers` (8), `--tries` (3), `--recusa-frac` (0,05),
`--ancorado-frac` (0,5), `--max-tokens` (400).

```python
DENY_URL_DEFAULT = r"(?i)(porn|xxx|nsfw|hentai|escort|onlyfans|putaria|sexo\b|crack|keygen|" \
                   r"warez|torrent|nifentas|bet[0-9]{2}|casino|tigrinho|slot[s]?\b|" \
                   r"assistir|dublado|legendado|filmes?online|megafilmes|superflix|" \
                   r"series?online|baixar|gratisbaixar)"
```

> [!important] Por que a blocklist existe se o professor veta
> Porque o professor **não veta tudo**: spam adulto e pirataria passam por serem texto legível
> (achado do piloto v1). A blocklist barra **antes** da chamada, o que economiza tokens e
> impede que o assunto entre no dataset por acidente.

## 2 · Limpeza de parágrafos

`clean_paragraphs()` quebra por `\n`, colapsa whitespace e descarta linhas que:

- têm menos de **60 chars** (`min_para_chars`);
- casam com `BOILER_RE`: `facebook|twitter|whatsapp|instagram|...|compartilh|assine|publicidade|
  leia mais|voltar|início|menu|pesquisar|cookie|todos os direitos|política de privacidade|
  siga-nos|newsletter|©|copyright|<|>`;
- contêm **mais de 2 URLs** (linha que é lista de links).

## 3 · Chunking em parágrafos inteiros

```python
def make_chunks(paras, cfg):
    units = [p if len(p) <= 2000 else *_split_sentences(p) for p in paras]
    # junta parágrafos inteiros até 2000 chars; nunca corta frase
```

- Alvo: **≤ 2.000 chars** (≈ 500 tokens do nosso tokenizer) — cabe folgado em 4096 de contexto e
  deixa o prefill barato.
- Mínimo: **700 chars** (`chunk_min_chars`) — trecho curto demais não sustenta pergunta.
- Parágrafo maior que 2.000 chars é quebrado **em frases** (`split` por `[.!?]\s+`); uma frase
  gigante fica sozinha, inteira.
- Até **3 chunks por doc** (`max_per_doc`) — evita que um único artigo domine.

## 4 · O prompt v2 (com MODO)

```
MODO: ANCORADO: a pergunta será feita junto com o TRECHO, que o leitor terá em mãos — ela
pode se referir ao conteúdo ('segundo o texto', 'no caso descrito').
TIPO DE EXEMPLO: extracao
Tarefa: faça uma pergunta cuja resposta seja um fato explícito do trecho (nomes, números,
datas, lugares).

TRECHO:
<chunk>
```

O `MODO` foi a correção mais importante do piloto v1:

- **v1 (só ancorado)**: 28 de 300 perguntas citavam "o texto" — inúteis para um modelo que
  recebe só a pergunta;
- **v2 (50/50)**: no modo `SEM CONTEXTO` a pergunta é proibida de citar "texto/trecho/documento/
  artigo/fonte" (`FONTE_RE`, regex) e a recusa muda de frase.

| Modo | Pergunta | Recusa |
|---|---|---|
| **ancorado** (`true`) | pode dizer "segundo o texto" | `"Isso não consta no texto."` |
| **livre** (`false`) | tem de ser autossuficiente | `"Não tenho informações suficientes para responder isso."` |

## 5 · Schema e JSON garantido

```json
{"status": "ok|invalido", "tipo": "...", "pergunta": "...", "resposta": "...", "evidencia": "..."}
```

Todo campo obrigatório, `additionalProperties: false`, `strict: true`. Sem isso o professor
devolve texto livre e o gerador perde a chamada.

## 6 · Validação local (antes de aceitar)

`validate()` devolve `(motivo, pergunta, resposta, evidencia)`. Motivos possíveis:

| Motivo | Regra |
|---|---|
| `json_invalido` | resposta não é JSON |
| `fonte_invalida` | professor vetou o trecho (`status: invalido`) |
| `tipo_diferente` | devolveu outro `tipo` que o pedido |
| `pergunta_ruim` | não termina em `?` ou fora de 15–300 chars |
| `resposta_ruim` | fora de 10–900 chars |
| `nao_pt` | heurística `is_pt` (stopwords + acentos, ≥ 2 sinais) falhou |
| `menciona_fonte` | modo livre citando "texto/trecho/…" |
| `resposta_copia_pergunta` | resposta contida na pergunta |
| `recusa_ruim` | recusa não começa pela frase do modo |
| `recusa_com_evidencia` | recusa com `evidencia` preenchida (não devia) |
| `evidencia_vazia` | evidência < 20 chars |
| **`evidencia_nao_literal`** | evidência **não está** dentro do `chunk_text` |
| `pergunta_duplicada` | já vista (normalizada) — dedup exato em memória |
| `dominio_no_teto` | domínio atingiu o teto (5% de `--max-examples`, mínimo 3) |

Até **3 tentativas** por trecho (`--tries`), sorteando tipo e modo novos a cada tentativa.

> [!note] `evidencia_nao_literal` é o motivo mais frequente
> ~98 de 325 trechos no piloto v2. É o preço de exigir **cópia caractere por caractere**: o
> modelo às vezes parafraseia e perde a chamada. Esse rigor é o que permite afirmar que o par
> tem base no trecho.

## 7 · Concorrência e escrita

- `ThreadPoolExecutor(8)` com fila limitada a `workers × 3` futures — nunca carrega o corpus
  na memória e mantém os 8 slots sempre cheios.
- Escrita **incremental** com `flush()` por trecho concluído (+ `tqdm` com `rej`, `falhas`, `tok`).
- `--resume` lê `(doc_id, chunk_idx)` já gravados e **pula** esses trechos; o teto por domínio é
  reconstruído relendo o arquivo, para não estourar o teto ao retomar.
- `KeyboardInterrupt` para limpo: o que foi aceito já está no disco.

> [!bug] `--resume` engana o contador do `tqdm` — e o teto é **por processo**
> A barra começa do zero: conta só os pares **deste** processo. Em 2026-10-30 14h33 ela marcava
> `5804/30000` com **8.959** linhas no arquivo (3.155 da fase anterior + 5.804 desta) e o ETA
> aparecia ~1 h maior que o real. **Sempre confira com `wc -l`.**
>
> Consequência de projeto: como `--max-examples` limita só os pares **aceitos pelo processo
> atual**, retomar um arquivo que já tem *N* linhas e deixar a barra fechar produz `N + 30000`.
> No run de 30/10 isso ia gerar ~33.154 pares; o processo foi interrompido (`SIGINT` → bloco
> `finally` grava o `stats.json`) quando o `wc -l` cruzou 30.000 — sobrou 1 linha a mais, que
> foi cortada à mão. Para retomar de verdade contando o que já existe, o certo é
> `--max-examples $((30000 - $(wc -l < arquivo)))`.

## 8 · O que o `stats.json` registra

`accepted`, `rejected_chunks`, `chunks`, `reasons` (contagem por motivo), `tipos`, `modos`,
`completion_tokens`, `prompt_tokens`, `elapsed_s`, `tok_s_agregado`, `examples_min`, e a
**config inteira** (para reproduzir a geração exata).

### Os dois pilotos (auditados à mão)

| | piloto v1 (só ancorado) | piloto v2 (50/50) |
|---|---|---|
| aceitos | 300 | 300 |
| trechos tentados | 339 | 325 |
| rejeitados | 39 | 25 |
| `fonte_invalida` (veto) | 51 | 48 |
| `evidencia_nao_literal` | 104 | 98 |
| `menciona_fonte` | **57** | **37** |
| duração | 459,5 s | 448,0 s |
| pares/min | 39,2 | 40,2 |
| modos | 300 ancorados | 163 ancorados / 137 livres |

O que a auditoria manual encontrou: **spam adulto e de pirataria** (→ blocklist), **28 perguntas
citando "o texto"** (→ modo misto) e respostas que copiavam a pergunta.

### O run dos 30.000 (fim do dia 2)

Duas fases: **3.154** pares na primeira e **26.846** numa retomada de 11 h 51 min no professor
local Qwen3.5-4B (8 slots), até o arquivo fechar em **30.000**.

| | retomada final (30/10 11h52 → 23h43) |
|---|---|
| pares aceitos | 26.846 (3.154 + 26.846 = **30.000**, 78 MB) |
| trechos processados | 30.263 |
| trechos rejeitados | 3.417 |
| tokens do professor | 6,98 M de resposta / 35,3 M de prompt |
| duração | 42.650 s · **37,8 pares/min** · 163,7 tok/s agregados |

Motivos de rejeição acumulados na retomada (o contador é por motivo, um trecho pode somar
vários): `evidencia_nao_literal` 10.758 · `fonte_invalida` 5.810 · `menciona_fonte` 4.255 ·
`nao_pt` 318 · `pergunta_ruim` 280 · `pergunta_duplicada` 264 · `json_invalido` 233 ·
`evidencia_vazia` 48 · erros de rede 3.

Distribuição final por tipo: extração **18.388** · explicação **7.401** · recusa **2.279** ·
resumo **1.932**; por modo: ancorado **16.006** / livre **13.994**. Perguntas duplicadas: 14
(0,05%). Respostas: p50 102 chars, p90 258.

O teto por domínio (`domain_cap_frac` 5% → 1.500 pares) **nunca entrou em ação**: o domínio
mais produtivo rendeu 69 pares. Nenhuma linha truncada — o `flush()` por trecho garante que a
interrupção (`SIGINT`) não deixa lixo, e o `stats.json` é gravado no `finally`.

## Ver também

- [[Formato do exemplo]] — o JSONL que sai daqui
- [[Validação, dedup e splits]] — o que acontece com ele
- [[O professor local]] — o outro lado do HTTP
- [[Pingo]] — voltar ao índice
