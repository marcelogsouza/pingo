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
5,8–11,8× (B=16). Suíte: **103 testes verdes**.

**Agora: SFT.** O dataset está pronto e o pipeline inteiro escrito e testado; falta **rodar**
validação → treino → avaliação:

| Passo | Comando | Estado |
|---|---|---|
| gerar 30k pares | `uv run python scripts/gen_sft.py --max-examples 30000 --out data/sft/sft-v1.jsonl --resume` | ✅ **30.000** pares em `data/sft/sft-v1.jsonl` (78 MB) |
| validar | `uv run python scripts/validate_sft.py --judge-recusas` | ⬜ **próximo** — ~30 s + ~2 min de juiz → `data/sft/final/` |
| treinar | `uv run python scripts/train_sft.py` | ⬜ ~1–2 h GPU → `runs/sft-120m/` |
| avaliar | `uv run python scripts/eval_sft.py --n 200 --judge` | ⬜ depois do treino |

- Dataset (conferido em 30/10 23h45): 30.000 linhas, todas JSON válido, 14 perguntas
  duplicadas; 3.417 trechos rejeitados (12 motivos); extração 18.388 · explicação 7.401 ·
  recusa 2.279 · resumo 1.932; ancorado 16.006 / livre 13.994.
- O teto `--max-examples` é **por processo**: com `--resume` o arquivo passa do alvo (foi o
  que aconteceu — parei em 30.000 pelo `wc -l`, não pela barra).
- Depois do SFT-4: SFT-5 (opcional, instruções gerais) e Passo 8 (**350M — suspenso**), como
  em `PLAN.md`.

## ▶️ Se você é uma NOVA sessão — nesta ordem

1. `nvidia-smi`. Se ~31 GB ocupados, o **Strata** está de pé; o professor ocupa ~5 GB.
   **Avise o usuário antes de qualquer CUDA** — não derrube nada por conta própria.
2. `uv run pytest -q` → **103 verdes** (CPU, ~13 s).
3. `curl -s localhost:8001/v1/models` → deve devolver o `Qwen3.5-4B-GGUF`. Se não houver nada,
   **peça ao usuário para subir o professor** (o script mora em `~/llm_serving_scripts/`, fora
   do repo — não suba servidor você mesmo).
4. Continue pela tabela acima, do primeiro passo 🔄/⬜.

## 🚫 Não quebrar

- **`runs/pt-120m/` não se re-treina nem se apaga** — é o resultado da Fase 1;
  `scripts/train.py` **sem** `--resume` sobrescreve. Há backup `read-only` em
  `runs/pt-120m-BACKUP-2026-10-30/` (`dr-xr-xr-x`, `SHA256SUMS.txt` conferido): **nunca**
  escrever lá. O SFT grava em `runs/sft-*`.
- **`tokenizer/tokenizer.json` fica com 16.384.** Os 3 tokens de chat (→16.387) entram em
  memória no `train_sft.py`, que salva uma cópia em `runs/sft-*/tokenizer.json`. Regerar o
  tokenizer invalidaria todos os `data/tokens/*.bin`.
- **Nada de CUDA pesado com o professor servindo** (ele precisa da VRAM).
- **Um escritor por saída.** Dois `gen_sft.py` no mesmo JSONL corrompem o arquivo.

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
  dados, tokenizer, tokens, arquitetura, pré-treino, avaliação, **SFT (9 notas)**, testes e
  `09-referencia/{Comandos,Gotchas,Glossário}`.
- Atalhos: `docs/09-referencia/Comandos e cheatsheet.md` (comandos por etapa),
  `docs/09-referencia/Gotchas.md` (todas as armadilhas), `docs/00-projeto/Decisões (ADR).md`
  (por que cada decisão foi tomada).
