---
tags: [pingo, webui, treino, jobs]
etapa: 13
atualizado: 2026-11-02
---

# Interface de treino (`/treino`)

A WebUI (:8010) ganhou uma segunda ferramenta: em `http://127.0.0.1:8010/treino` você aponta
para um **dataset JSONL qualquer**, roda a QA, tokeniza (com vocabulário existente ou um novo
treinado do zero), dispara **pré-treino ou SFT** e assiste — log ao vivo, curva de loss e
monitor de CPU/RAM/disco/VRAM. Motivador: re-treinar com o dump 2026 da Wikipédia pt sem
sair do terminal… mas sem ter que lembrar os argv de cor.

> [!important] A UI não treina nada — ela **orquestra**
> Nenhum segundo loop de treino foi escrito. Cada job é um `subprocess` dos scripts de sempre
> (`train.py`, `train_sft.py`, `prepare_tokens.py`, `train_tokenizer.py`), supervisionado pelo
> `JobRunner` (`pingo/jobs.py`). O que a CLI faz, a interface faz igual — mesma esteira, mesmas
> regras do AGENTS.md.

## As quatro abas

| Aba | O que faz | Backend |
|---|---|---|
| ① dataset (QA) | escolhe um `.jsonl` de `data/**` (ou digita o caminho), roda `qa_dataset.analisar` | `GET /api/treino/datasets`, `GET /api/treino/qa?path=…` |
| ② tokenização | lista o registry `tokenizer/*` (o default vem marcado **congelado**); treina vocab novo (`--id/--vocab/--specials/--jsonl`) ou binariza um JSONL aprovado com um vocab existente | `POST /api/treino/tokenizador`, `POST /api/treino/tokenizacao` |
| ③ treino | formulários pré-treino/SFT com os padrões dos runs recomendados; **épocas → `--max-steps`** calculado sobre o `train_tokens` do `meta.json`; iniciar / retomar (`--resume`) / parar (SIGTERM) | `POST /api/treino/pretreino`, `POST /api/treino/sft`, `POST /api/treino/parar` |
| ④ runs & monitor | histórico de `runs/*` com steps/**progresso**/val/final/ckpt; clique → **curva de loss SVG** do `log.jsonl`; hardware na barra de cima sempre | `GET /api/treino/runs`, `GET /api/treino/runs/<nome>`, `GET /api/treino/hw` |

O front (`web/treino.html`, vanilla JS, zero dependências) faz **polling de 2 s** em
`status`/`hw`/`log?depois=<cursor>` — não SSE (decisão do treino-4: o anel do runner + cursor
resolvem sem o peso de um stream por aba aberta).

## O JobRunner (`pingo/jobs.py`)

- **fila de um-por-vez**: `iniciar()` levanta `JobOcupado` se já há job; as rotas devolvem
  **429**. É a regra "um escritor por saída" do AGENTS.md virada código.
- **Estados**: `rodando → parando (SIGTERM pedido) → interrompido | concluido | falhou`
  (rc 130 ou estado `parando` ⇒ `interrompido`).
- **registro persistente** em `runs/jobs.json` (últimos 200, escrita atômica). Sobrevive ao
  restart da webui.
- **caça ao órfão no boot**: se o servidor morreu com um filho vivo, o runner SIGTERM o pid
  registrado e fecha o registro (`motivo: "webui reiniciou — órfão encerrado"`). Nenhum treino
  fantasma segurando a GPU depois de um Ctrl+C no servidor.
- **saída** em anel de 400 linhas + `log_records()` lendo o `log.jsonl` do `--out` (tolerante a
  linha escrita pela metade entre flushes).
- **mutex com o chat**: `Server._jobs` é injetado no `create_app`; com job rodando,
  `Server.carregar()` **recusa** carregar modelo (a VRAM é uma só), e toda rota de disparo
  descarrega o modelo do chat antes de começar (`_desocupa_gpu`).

## Progresso do run (quanto falta)

Desde 2026-11-03 a aba ③ mostra **quanto falta** de um run, não só a curva de loss:

```
pré-treino wiki-120m                                     4,0%
▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
step 50 / 1.240 · tokens 26,2M / 650,1M · dataset 650,2M (1,00×) · 100,9k tok/s
decorrido 4m 48s · restam 1h 43m
```

A conta é sempre a mesma: `% = step / max_steps`, `tokens = tokens_seen / (max_steps ×
tokens_por_step)`, `ETA = (max_steps − step) ÷ taxa`, e a **taxa é medida no próprio log**
(dois registros de treino consecutivos: steps ÷ Δ`t`, onde `t` é o relógio do processo) — não
é média desde o início, então uma queda de throughput aparece no ETA em segundos. Runs com
`--resume` não enganam a conta: o step inicial é o do checkpoint, os tokens vistos vêm do log
e `max_steps` é absoluto.

De onde vem o **plano** (`max_steps`, `tokens_por_step`), em ordem de preferência:

1. o `argv` do job (a UI sabe o comando que disparou);
2. o registro **`evento=inicio`** que o `train.py` grava no `log.jsonl` na largada —
   `{evento, max_steps, tokens_por_step, tokens_dir, train_tokens}`. É o que faz um run
   disparado **fora** da UI (nohup, terminal) também responder "quanto falta";
3. nada disso → a barra mostra o andamento sem `%` nem ETA (runs antigos, ex. `pt-120m`).

O "tamanho do dataset" é o `train_tokens` do `meta.json` do `--tokens-dir`: a barra diz
quantas **épocas** o plano cobre (1,00× = o corpus inteiro uma vez). Detalhe importante: se o
run é `max_steps` acima de 1 época, o `%` é do **plano**, não do dataset — os dois números
aparecem lado a lado justamente para não confundir.

Detalhe de implementação que vale saber: a página calcula o progresso **sozinha** quando a API
ainda não manda o campo `progresso` (fallback no JS) — foi assim que a barra entrou em
produção **sem reiniciar** a webui no meio de um treino (o boot do runner caça e mata órfãos,
então reiniciar = matar o run).

## O que as validações barram (antes de gastar processo)

- nomes de artefato (`runs/`, `data/tokens/`, `tokenizer/`) casam `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`;
  caminhos fora do repo são rejeitados (o servidor escuta na LAN **sem autenticação**);
- `runs/pt-120m` e o backup são **protegidos por nome → 403** — a interface nunca escreve neles;
- `--out` existente sem `retomar` → **409** ("marque retomar"); `retomar` sem `ckpt.pt` → 400;
- `tokens_dir` sem `meta.json`, tokenizer sem `tokenizer.json`, base sem `config.json`,
  `data_dir` sem `train.jsonl` → 400;
- fila ocupada → **429 verificado antes de qualquer outra validação**;
- device `cpu` ⇒ o filho roda com `CUDA_VISIBLE_DEVICES=""` (com o professor na VRAM, até o
  health check do `opt.step()` tenta `cuInit` e OOM-flaka).

## Tokenizer por experimento (treino-5, o invariante novo)

`tokenizer/<id>/` é artefato de experimento; `tokenizer/` (16.384) fica intocado. A corrente:

```
dataset.jsonl ──train_tokenizer --id wiki-16k --vocab 16384 --specials '<|bos|>,…,<|end|>' --jsonl …──▶ tokenizer/wiki-16k/
              ──prepare_tokens --input jsonl --tokenizer tokenizer/wiki-16k --out data/tokens/wiki──▶ meta.json {tokenizer_id, vocab_size}
              ──train.py: vocab_size DERIVADO do meta.json (config nunca vence o meta)
              ──train_sft.py --tokenizer tokenizer/wiki-16k (chat já embutido ⇒ resize 0)
```

- O split val/train do JSONL é o **mesmo hash blake2b** do `process_data.py` — doc nunca troca de lado.
- Os tokens de chat (`<|user|> <|assistant|> <|end|>`) podem nascer **embutidos** no vocabulário
  (`--specials`), e então o SFT não precisa do resize de embedding.
- A ficha do vocab novo (`vocab_stats.json`) traz compressão `our_chars_per_token` e
  `gain_vs_gpt2_pct` no texto de val — tolerante a offline (sem cache do GPT-2, os campos saem `null`).

## Parada graciosa (treino-1, o contrato do "parar")

SIGTERM só **pede pausa**: a flag `INTERROMPER` é checada no topo do loop; o step corrente fecha,
o ckpt é salvo, `{"evento":"interrompido"}` vai ao `log.jsonl`, e o processo sai com **rc 130**.
Um run interrompido **nunca** ganha `final/`. O botão "retomar" é literalmente `--resume`, que os
testes já provam ser bit-exato.

## Ver também

- [[WebUI]] — o chat e o `/lab` na mesma app
- [[Suíte de testes]] — `test_treino_api.py`, `test_jobs.py`, `test_treino_pipeline.py`
- [[Gotchas]] — o flake do torch com GPU alheia cheia
- [[Run do SFT-120M]] — os padrões que os formulários trazem
