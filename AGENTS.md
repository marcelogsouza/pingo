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
com ele e um **modo didático em `/lab`** — **curso completo, ①–⑦**: ① tokenização, ② embeddings,
③ RoPE + atenção, ④ RMSNorm + SwiGLU, ⑤ lm head, **⑥ amostragem — a roleta** (fatias de 360° × p,
girar com semente ou escolher o token na mão, e o desdobramento com `generate_stream` até o
`<|end|>`) e **⑦ loop + KV cache em cena** — `roda_loop` toca o próprio `generate_stream` do
chat com caderno de notas (o token escolhido **volta** como entrada; fita do cache 12×n acendendo
**uma coluna por step**, preço medido: 29 vs 407 tokens-forward, o step sem cache a 38,7 ms contra
20,6 ms com logits idênticos, maxdiff 4,8×10⁻⁶). ⑥ e ③–⑤ dividem a captura, `pingo/generate.py`
é a conta de verdade — e o top-p do `sample_next` foi **consertado** no caminho (cumsum em
probabilidades; testes novos a travam). Suíte: **193 testes verdes**.

**Agora: acabou o caminho feliz.** O pipeline SFT inteiro rodou; o que sobra é **melhorar**:

| Passo | Comando | Estado |
|---|---|---|
| dataset | `data/sft/final/` (26.479/1.581/1.455) | ✅ validado, sem vazamento |
| SFT recomendado | `uv run python scripts/train_sft.py --epochs 1 --lr 5e-5 --min-lr 5e-6 --warmup-steps 40 --out runs/sft-120m-ep1` | ✅ 828 steps, 4 min 52 s |
| avaliar | `uv run python scripts/eval_sft.py --model runs/sft-120m-ep1/final --n 200` | ✅ 24 s (sem `--judge`) |
| WebUI | `./scripts/webui.sh` (Ctrl+C desliga) → `http://127.0.0.1:8010` ou `http://192.168.0.101:8010` (LAN, sem auth) | ✅ SFT + base, streaming |
| modo didático | mesma UI, rota **`/lab`** (lógica em `pingo/didatic.py`, página em `web/lab.html`) | ✅ **①–⑦ — curso fechado**: o `capture` serve atenção, bloco, lm head **e roleta** (um forward só); o ⑥ é stateless (chave `modelo+ids`, `girar` na mesma frase é cache-hit, 35 ms) e o ⑦ (`/api/lab/loop` → `roda_loop`) replay o loop real numa chamada — tetos 160 ids + 64 novos, 0,65 s o request de 8+22 na CPU |

- **Não use `runs/sft-120m/` (3 épocas, lr 1e-4) como o modelo do projeto**: F1 igual, mas
  esquece o português (fator 6,94× contra 1,91×). Ficou como contraexemplo em
  `docs/07-sft/Run do SFT-120M.md`.
- O SFT **sem juiz de recusas** (o professor já estava derrubado): `recusa_ok` 0,455 e 0,545
  nos dois runs — é a fraqueza mais clara e o alvo do SFT-5 (instruções gerais + texto cru).
- Próximos, em `PLAN.md`: SFT-5 (opcional, instruções gerais/replay de texto) e Passo 8
  (**350M — suspenso**). O modo didático **não tem etapa pendente** — a esteira ①–⑦ fechou
  em 2026-11-01 (se um ⑧ existir um dia, o andaime está em `docs/07-sft/Modo didático.md`).

## ▶️ Se você é uma NOVA sessão — nesta ordem

1. `nvidia-smi`. Se ~31 GB ocupados, o **Strata** está de pé; o professor ocupa ~5 GB.
   **Avise o usuário antes de qualquer CUDA** — não derrube nada por conta própria.
2. `uv run pytest -q` → **193 verdes** (CPU, ~14 s).
3. `./scripts/webui.sh` (ou `uv run python scripts/webui.py`; o script derruba a instância
   anterior e desliga com Ctrl+C) e abra `http://127.0.0.1:8010` (ou o IP da LAN que ele
   imprime): se o chat responder, o modelo e a GPU estão de pé; `/lab` responde mesmo com a
   GPU ocupada. (A porta 8000 desta máquina é de outra aplicação; o host padrão é `0.0.0.0`,
   **sem autenticação**.)
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
- **Nada de CUDA pesado com o professor servindo** (ele precisa da VRAM). O lab ①–② leem
  `embed.weight` do `model.safetensors` na CPU (`Server.tabela_de()`) e **nunca** chamam
  `Server.carregar()`. A exceção deliberada são os passos ③–⑦ (`/api/lab/attn|rope|bloco|
  lmhead|amostra` → `Server.captura_atencao()`, e `/api/lab/desdobrar|loop` → o
  `generate_stream` real): carregam o modelo no device do servidor sob `gen_lock` (429 se o
  chat gera) e tetos de tokens — **um único forward serve ③, ④, ⑤ e o primeiro menu do ⑥**
  (chave `modelo+ids`; a captura guarda também `nfinp/nfin` e os logits [n×16.387]). ③–⑤
  limitam a 64 tokens; ⑥–⑦ sobem o teto para 160 (`MAX_AMOSTRA_TOKENS`, frase + gerados) e
  são **stateless** — o cliente manda a história inteira; `girar`/`forcar`/mexer nos sliders
  na mesma frase é cache-hit (2–35 ms). O ⑦ é o único que **gera** no lab: `/api/lab/loop` →
  `roda_loop` (o loop do stream com caderno de notas; +64 novos `MAX_LOOP_NOVOS`, a fita lê o
  cache por `PingoCache.snapshots()` — **leitura**, não se escreve nela). Com o professor de
  pé, rode o servidor em `--device cpu` (0,66 s o forward de 11 tokens a frio, 0,07 s quente;
  desdobramento de ~30 tokens até o `<|end|>`, 0,64 s; o replay do ⑦ com 8+22 tokens, 0,65 s);
  nunca derrube o professor por conta própria.
- **A conta de amostragem é uma só** (`sample_next` no chat, `menu_amostra` no ⑥, e o ⑦ É o
  loop do chat — `roda_loop` espelha `generate_stream` linha a linha): o top-p acumula
  **probabilidades**, não logits — o cumsum em logits que existia antes cortava tudo com
  logit positivo (bug pego pela roleta do ⑥; `test_sample_next_top_p_e_nucleus_de_
  probabilidade` e os testes do ⑥ travam a semântica;
  `test_roda_loop_e_o_generate_stream_token_por_token` trava o loop do ⑦ contra o do chat).
  Não recriar um segundo filtro nem um segundo loop.
- **Um escritor por saída.** Dois `gen_sft.py` no mesmo JSONL corrompem o arquivo; dois
  `train_sft.py` no mesmo `--out` idem. A WebUI também ocupa GPU: rode uma geração por vez.

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
