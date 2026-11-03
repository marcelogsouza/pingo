# AGENTS.md — estado vivo do projeto

Handoff entre sessões, em ~80 linhas. Responde só a três perguntas: **onde estamos, o que
fazer agora, o que não quebrar**. O plano está em [`PLAN.md`](PLAN.md), a explicação de **como
cada coisa foi feita e por quê** está no vault [`docs/Pingo.md`](docs/Pingo.md) e a história
está no `git log`.

> **Regra de ouro, em duas partes.**
> 1. **Ao fechar um passo** (ou uma etapa dele): atualize a seção "Onde estamos" aqui **e
>    migre para o `docs/`** o que virou conhecimento — narrativa, números, gotchas, decisões —
>    numa nota estruturada, mantendo o que já existe lá. Depois commite (`docs: estado` ou no
>    mesmo commit do passo).
> 2. **Este arquivo não acumula história.** O que já está no `docs/` e não é necessário para
>    executar a etapa atual **sai daqui**. `docs/` é o arquivo crescente; `AGENTS.md` é a fatia
>    viva. Se os dois divergirem num número, o `AGENTS.md` ganha e o vault é corrigido.

## Onde estamos

**Fase 1 concluída (Passos 1–7)**: Pingo-120M pré-treinado em `runs/pt-120m/final/` — 5.248
steps, 2,75B tokens, 7,56 h, val **2,507 / ppl 11,67**, geração de pt coerente, KV cache
5,8–11,8× (B=16). **SFT concluído**: `runs/sft-120m-ep1/final/` (1 época, lr 5e-5) — formato
100%, F1 0,375, fidelidade 0,59, ppl do pré-treino **1,91×**. WebUI em Flask para conversar
com ele e um **modo didático em `/lab`** — o curso de sete passos (① tokenização → ⑦ loop +
KV cache) está **completo** (2026-11-01): conta, números e contrato das rotas moram em
`docs/07-sft/Modo didático.md`. **Interface de treino em `/treino` concluída** (Passo 13,
treino-1…5, 2026-11-02): QA de dataset, tokenização (vocab por experimento `tokenizer/<id>/`),
pré-treino/SFT com fila de 1 job, parar (SIGTERM→ckpt) e retomar, curva de loss e HW — o
contrato mora em `docs/07-sft/Interface de treino.md`. Suíte: **232 testes verdes**.

**Agora: acabou o caminho feliz.** O pipeline SFT rodou; o que sobra é **melhorar** — e o eixo
atual é **corpus** (wiki limpa × OSCAR sujo). Comandos por etapa em
`docs/09-referencia/Comandos e cheatsheet.md`; o modelo do projeto é o SFT de 1 época
(`runs/sft-120m-ep1`, lr 5e-5) — **não** o `runs/sft-120m` de 3 épocas, que esquece o português
(`docs/07-sft/Run do SFT-120M.md`). WebUI: `./scripts/webui.sh` → :8010 (LAN, sem auth).

- **Corpus wiki pronto (2026-11-03)**: `data/wiki/pt.jsonl` (836.266 docs) + `pt-val.jsonl`
  (17.093) e os tokens em `data/tokens/wiki-16k/` — **650.248.473 train / 13.117.734 val**,
  vocab 16.384 congelado (embeddings e `runs/pt-120m` compatíveis). 1 época = **1.240 steps
  ≈ 1,8 h** de GPU (com `--compile`; sem a flag cai para ~56k tok/s). Régua
  `--short-line-ratio 0.9` (0.6 do OSCAR mataria 67% da wiki); números em
  `docs/01-dados/Dataset Wikipedia PT.md`.
- **Baseline cruzado (medido 2026-11-03, CPU, mesmo harness)**: o Pingo-120M do OSCAR tira ppl
  **13,75** no val OSCAR e **24,72** no val wiki — estilo enciclopédico é fora de distribuição
  para ele. É o número-alvo dos runs da wiki.

### ⏳ Runs da wiki em andamento (retomar daqui)

| run | o que é | estado em 2026-11-03 15:38 |
|---|---|---|
| **`runs/wiki-120m`** | do zero na wiki, 1.240 steps | ✅ **concluído** — val loss **2,878 / ppl 17,78** no val da wiki |
| `runs/pt-120m-wiki` | `--resume` do ckpt do OSCAR na wiki, `max_steps` **6488** (5.248 + 1.240) | **rodando** (step 5.300, restam ~103 min, termina ~17:22) |

**O (a) já respondeu a pergunta da fase**: 650M tokens **limpos** de wiki dão ppl **17,78** no
val da wiki, contra **21,7** do modelo treinado em 2,75B tokens **sujos** do OSCAR medido no
mesmo harness (o `pt-120m` no step 5.250 do run (b) marca exatamente 3,0759 / 21,7). Ou seja:
no domínio da wiki, o corpus limpo e 4× menor ganha. O (b) dirá se o OSCAR ainda serve de
inicialização (o número a bater é 17,78).

- **Ao voltar**: `curl -s localhost:8010/api/treino/status` → a barra do `/treino` diz step/%/ETA
  **da sessão** (run retomado conta a partir do `--resume`, não do plano absoluto que mistura
  corpora). O (b) terminando, o passo é o **wiki-4** (abaixo).
  Se precisar disparar o (b) de novo, o comando é:
  `curl -s -X POST localhost:8010/api/treino/pretreino -H 'Content-Type: application/json' -d
  '{"out":"pt-120m-wiki","retomar":true,"tokens_dir":"data/tokens/wiki-16k","max_steps":6488,"compile":true}'`
  (o `ckpt.pt` do `pt-120m-wiki` é cópia do `pt-120m`; `runs/pt-120m` continua intocado).
- **Avaliar (wiki-4)**: ppl no val da wiki + `eval_sft.py` nos dois runs contra os baselines
  acima. O critério é dizer se **663M tokens limpos** batem os **2,75B sujos** no português —
  e então decidir o híbrido com o OSCAR destilado do Passo 14. `PLAN.md` 14b tem os passos.
- Próximos, em `PLAN.md`: **Passo 14 — QA do OSCAR com LLM** (juíz Qwen 3.6 35B rotula amostra,
  classificador CPU escoreia os 10,7M → `data/oscar-qa/`; desenho em
  `docs/01-dados/QA do dataset com LLM.md`; ~45% do OSCAR aprovado tem raspa de web detectável,
  então o passo começa pelo pente-fino qa-llm-0), SFT-5 (instruções gerais + texto cru; a
  fraqueza atual é `recusa_ok` ~0,5) e Passo 8 (**350M — suspenso**).

## ▶️ Se você é uma NOVA sessão — nesta ordem

1. `nvidia-smi` **e** `curl -s localhost:8010/api/treino/status`. Se houver job `rodando`, ele é
   o run da wiki (seção acima): **não reinicie a webui** (o boot mata o job de propósito) e não
   rode a suíte inteira enquanto ele treina (é CPU que o loader do treino precisa). Se ~31 GB
   ocupados por outro processo, o **Strata** está de pé; o professor ocupa ~5 GB.
   **Avise o usuário antes de qualquer CUDA** — não derrube nada por conta própria.
2. `uv run pytest -q` → **232 verdes** (CPU, ~85 s). Desde 2026-11-03 a suíte **não mata mais**
   treino vivo (a caça ao órfão só roda no processo da webui) — mas os jobs de subprocess ainda
   disputam CPU com o treino.
3. `./scripts/webui.sh` (ou `uv run python scripts/webui.py`; o script derruba a instância
   anterior e desliga com Ctrl+C) e abra `http://127.0.0.1:8010` (ou o IP da LAN que ele
   imprime): se o chat responder, o modelo e a GPU estão de pé. (A porta 8000 desta máquina é
   de outra aplicação; o host padrão é `0.0.0.0`, **sem autenticação**.)
4. `curl -s localhost:8001/v1/models` → só é preciso se você for usar `--judge` ou regerar
   dataset. Deve devolver o `Qwen3.5-4B-GGUF`; se não houver nada, **peça ao usuário para subir
   o professor** (o script mora em `~/llm_serving_scripts/`, fora do repo — não suba servidor
   você mesmo).

## 🚫 Não quebrar

- **`runs/pt-120m/` não se re-treina nem se apaga** — é o resultado da Fase 1;
  `scripts/train.py` **sem** `--resume` sobrescreve. Há backup `read-only` em
  `runs/pt-120m-BACKUP-2026-10-30/` (`dr-xr-xr-x`, `SHA256SUMS.txt` conferido): **nunca**
  escrever lá. O SFT grava em `runs/sft-*`.
- **`tokenizer/tokenizer.json` fica com 16.384.** Os 3 tokens de chat (→16.387) entram em
  memória no `train_sft.py`, que salva uma cópia em `runs/sft-*/tokenizer.json`. Regerar o
  tokenizer invalidaria todos os `data/tokens/*.bin`.
- **Nada de CUDA pesado com o professor servindo** (ele precisa da VRAM). Com o professor de
  pé, o webui roda em `--device cpu` — e nunca derrube o professor por conta própria.
- **A conta de amostragem é uma só e o loop é um só**: `sample_next` do chat (o top-p acumula
  **probabilidades**, não logits) e `generate_stream` — todo loop do projeto é ele, teste
  trava. Não recriar um segundo filtro nem um segundo loop.
- **Um escritor por saída.** Dois `gen_sft.py` no mesmo JSONL corrompem o arquivo; dois
  `train_sft.py` no mesmo `--out` idem. A WebUI também ocupa GPU: rode uma geração por vez.
- **JSONL já fatiado não se re-fatia.** `data/wiki/pt.jsonl` é o lado **train**; o val é o irmão
  `pt-val.jsonl`. O `prepare_tokens --input jsonl` detecta o irmão e **aborta** se o val sair
  vazio — não contorne a guarda nem passe `--val-pct` esperando que ele divida de novo.
- **Não reinicie a webui com job rodando** (o boot do `JobRunner` caça órfãos e encerra o job —
  é o preço de não ter treino fantasma na GPU) e **não dispare dois runs na GPU**: um escritor
  por saída, uma VRAM. Conserto de backend não entra em produção durante um run; mudança de
  **front** (`web/treino.html`) vale no reload da página.
- **A receita do run é com `--compile`** (56k → 102k tok/s, medido 2026-11-03): bs 16 × ctx 2048
  × accum 16 = 524.288 tokens/step, lr 3e-4 → 3e-5, warmup 400, wd 0.1.
- **Treino disparado é só pelo `JobRunner`** (`pingo/jobs.py` — o `/treino` e nada mais):
  fila de 1 job, `runs/jobs.json`, caça ao órfão no boot, e com job rodando o chat **não
  carrega modelo** (a VRAM é uma só). `runs/pt-120m` e o backup são protegidos por nome (403)
  na API; `tokenizer/` default continua intocado — vocab novo é `tokenizer/<id>/` e o
  `meta.json` dos tokens manda no `vocab_size` (o `train.py` deriva dele).

## Convenções

- Tudo via **`uv`** (`uv run …`, `uv add …`); `uv.lock` é versionado.
- Testes antes de fechar qualquer passo: `uv run pytest -q`.
- Gitignored: `data/`, `tokenizer/`, `runs/`, `*.bin`, `*.parquet`, `*.pt`. Os dados (~69 GB)
  ficam só nesta máquina (RTX) — na M4 os testes de integração pulam.
- Commits `step N: …` / `sft-N: …` / `docs: …`; **push só depois da revisão do usuário**.
- Config default **Pingo-120M**: 12 camadas, d_model 896, 14 heads, SwiGLU 2304, ctx 2048.
- RTX PRO 4500 (32 GB VRAM); RAM apertada (o Strata segura ~52 GB dos 62).

## Fontes da verdade (não duplicar aqui)

- **Plano e critérios de aceite** → [`PLAN.md`](PLAN.md). **Sequência histórica** →
  `git log --oneline`.
- **Como foi feito e por quê** → `docs/` (vault Obsidian, home `docs/Pingo.md`): projeto/ADRs,
  dados, tokenizer, tokens, arquitetura, pré-treino, avaliação, **SFT (12 notas)**, testes e
  `09-referencia/{Comandos,Gotchas,Glossário}`.
- Atalhos: `docs/09-referencia/Comandos e cheatsheet.md` (comandos por etapa),
  `docs/09-referencia/Gotchas.md` (todas as armadilhas), `docs/00-projeto/Decisões (ADR).md`
  (por que cada decisão foi tomada).
