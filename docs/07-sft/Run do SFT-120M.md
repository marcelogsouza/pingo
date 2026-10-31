---
tags: [pingo, sft, treino, resultado]
etapa: sft-3
atualizado: 2026-10-31
---

# Run do SFT-120M

O SFT rodou **três vezes** em 2026-10-30/31, e não foi de primeira: três épocas com
`lr 1e-4` entregam um modelo que segue instruções e **esquece o português**. O run que vale é
o de **1 época com `lr 5e-5`**. Números, curva e a decisão.

| | v1 · 3 épocas | v1 · `best.pt` (step 200) | **v2 · 1 época** |
|---|---|---|---|
| diretório | `runs/sft-120m/` | `runs/sft-120m/best/` | **`runs/sft-120m-ep1/`** |
| lr / warmup | 1e-4 / 50 | — | **5e-5 / 40** |
| steps | 2.484 (3 ép.) | 200 | **828 (1 ép.)** |
| duração na GPU | 13 min | — | **4 min 52 s** |
| val loss (SFT) | 2,1615 | 2,0218 | **1,7624** |
| `taxa_terminou` | 0,995 | 0,975 | **1,000** |
| `f1` (heldout, n=200) | 0,383 | 0,310 | **0,375** |
| `apoio` (fidelidade) | 0,592 | 0,519 | **0,594** |
| `recusa_ok` | 0,545 | 0,364 | 0,455 |
| ppl no val do pré-treino | 88,6 (**fator 6,94**) | 39,2 (3,07) | **24,3 (fator 1,91)** |

O v1 vence por 0,008 de F1 — ruído — e paga **3,6× mais esquecimento**. O v2 é o modelo que
a WebUI serve e o que está documentado como resultado.

## A curva que contou a história

Val loss do SFT, por step (mesmo `val`, 1.581 exemplos):

```
v1 (3 épocas, lr 1e-4)    200: 2,022   600: 2,053  1000: 2,052  1400: 2,075  1800: 2,098  2400: 2,164
v2 (1 época,  lr 5e-5)    100: 1,989   300: 1,814   500: 1,780   700: 1,775   828: 1,762
```

O v1 **mínimo no step 200** (0,24 de época!) e daí só piora — é overfit puro: train loss
desce a 0,40 (o modelo decora 26k respostas) enquanto o val sobe. O v2 desce a curva inteira
e termina no mínimo, com lr menor e uma passada só.

> [!danger] O que "esquecer o português" significa na prática
> ppl 88,6 no val do pré-treino contra 12,8 do base. Não é um número abstrato: na amostragem
> livre o v1 vira um gerador de "frases de manual" — responde no formato, mas perdeu o fio da
> língua. O v2 fica em 24,3: ainda aprendeu o formato, mas continua um modelo de linguagem
> razoável. `fator > 1,3` é o gatilho de investigação do [[Avaliação SFT]].

## O que o v2 acerta e erra (amostras reais, greedy)

Com contexto (mini-RAG), ele é **bom no que copia** e fraco no que calcula:

```
Contexto: …plano de investimentos de 12 bilhões de reais até 2028, foco em redes de distribuição…
P: Quanto a Cemig vai investir até 2028?
R: A Cemig vai investir R$ 1 bilhão até 2028.        ← errou o número (12 bi)
P: O que a empresa vai modernizar?
R: A Cemig vai modernizar as redes de distribuição no interior do estado.   ← certo
P: Resuma o texto:
R: A Cemig anunciou que vai lançar um plano de investimentos de 12 bilhões de reais até 2028.  ← certo
```

Sem contexto, ele **inventa com segurança** ("O presidente do Brasil é Luiz Carlos Trabuco")
em vez de recusar — `recusa_ok` 0,455 explica isso: o modelo aprendeu a frase de recusa, mas
decide mal quando usá-la. É a fraqueza mais clara do run e o alvo do SFT-5.

## Como reproduzir

```bash
# o run recomendado (4min52s numa RTX PRO 4500)
uv run python scripts/train_sft.py --epochs 1 --lr 5e-5 --min-lr 5e-6 \
    --warmup-steps 40 --eval-steps 100 --out runs/sft-120m-ep1

# avaliar (24 s, sem juiz: o professor já tinha sido derrubado)
uv run python scripts/eval_sft.py --model runs/sft-120m-ep1/final --n 200
```

O `best.pt` não é um diretório HF; para avaliá-lo houve um export à mão:

```python
import shutil, torch
from pingo.config import PingoConfig
from pingo.model import PingoLM
ck = torch.load("runs/sft-120m/best.pt", map_location="cpu")
cfg = PingoConfig.from_pretrained("runs/sft-120m/final")   # vocab 16.387
m = PingoLM(cfg); m.load_state_dict(ck["model"])
m.save_pretrained("runs/sft-120m/best")
shutil.copy("runs/sft-120m/final/tokenizer.json", "runs/sft-120m/best/tokenizer.json")
```

## Detalhes operacionais

- **Batch efetivo 32** (8 × 4 accum), `max-len 1024`, cosine com warmup, `bf16`, grad-clip 1,0.
  Nenhum ajuste de código foi preciso entre os runs: tudo é CLI.
- **Velocidade estável**: ~110 exemplos/s (~3.200 tokens de resposta/s) nos dois runs; o v2
  gasta 4 min 52 s porque vê 1/3 dos dados.
- **VRAM**: 8,6 GB de treino; a WebUI roda junto (2 GB) nos 32 GB.
- O SFT **não** usou `--compile` (ganho pequeno em 800–2.500 steps, risco de recompilar por
  shape variável).
- O dataset foi o `data/sft/final/` do validador: 26.479 treino / 1.581 val / 1.455 heldout,
  zero vazamento de documento ([[Validação, dedup e splits]]).

## Ver também

- [[Treino SFT]] — o que o script faz (tokens de chat, labels, máscara)
- [[Avaliação SFT]] — como cada métrica é calculada
- [[WebUI]] — onde ver o modelo conversando
- [[Geração do dataset]] — de onde vieram os 30.000 pares
- [[Pingo]] — voltar ao índice
