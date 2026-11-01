---
tags: [pingo, sft, ferramenta, curso, tokenizer]
etapa: 0
atualizado: 2026-11-01
---

# Modo didático (`/lab`) — passo 1: tokenização

O Pingo é um **projeto-curso** ([[Visão e filosofia]]): o valor está tanto no modelo quanto em
conseguir *mostrar* o que acontece dentro dele. O modo didático é a tela que faz isso — um
"laboratório" na mesma WebUI, com uma esteira de etapas no topo (tokenização → embeddings →
atenção → MLP → lm head → amostragem → loop) e, por enquanto, a **primeira etapa pronta**.

```
http://127.0.0.1:8010/lab        (ou o IP da LAN, ver [[WebUI]])
```

## O que a tela faz

| elemento | por quê |
|---|---|
| **chips coloridos** | cada pílula é um token; a cor é a *categoria* (especial, início de palavra, continuação, pontuação, número, byte puro, quebra de linha) |
| **id embaixo de cada chip** | o modelo não vê letras, vê números — ver o id colado no pedaço é o que fecha a ideia |
| **o invólucro do chat, sem rótulo** | sempre tokenizamos `<|user|>seu texto<|assistant|>`: os dois violeta no começo/fim são o que o SFT ensinou o modelo a reconhecer. No **base** não há invólucro (o vocab dele não tem `<|user|>`) |
| **medidor** | `N caracteres → M tokens · X chars/token · bytes · distintos` — o aluno vê o custo do texto em tempo real |
| **ficha do token** (clique) | o texto humano, a superfície byte-level, os bytes em hex, a norma, e a faixa de calor dos 896 números |
| **vizinhos por cosseno** | o argumento mais forte do passo 1: a semântia já está na geometria da tabela, antes da primeira camada |
| **busca no vocab** | digita `4560`, `ção` ou `Ã§Ã£o` e cai na mesma ficha |
| debounce de 250 ms | o aluno tokeniza enquanto digita; nada de botão "processar" |

## As quatro decisões que importam

**1. Superfície ≠ texto.** Num BPE byte-level o token é uma sequência de *bytes*: o vocab
guarda `ĠatÃ©`, o humano lê ` até`. As duas aparecem na tela, lado a lado, porque é ali que o
byte-level costuma fazer sentido. A implementação nunca recupera o texto com `decode([id])`
(isso devolve `U+FFFD` em byte incompleto): ela usa os **`offsets` do `encode()`, que são
posições de caractere** do texto original — conferido no acento e nos tokens de chat.

```
i   id  categoria    superfície   bytes                texto
5  406 início       'ĠO'         20 4f                 ' O'
6 2598 início       'ĠpreÃ§o'    20 70 72 65 c3 a7 6f  ' preço'
10 393 número       'Ġ1'         20 31                 ' 1'
13 161 bytes        'â'          e2                    '≈'   ← byte-fallback puro
```

**2. O laboratório não encosta na GPU.** A ficha do token precisa da tabela de embeddings, e
o modelo mora na VRAM disputada com o Strata. Então
`TabelaEmbeddings.do_checkpoint()` lê **só a chave `embed.weight`** do `model.safetensors`
na CPU — 58 MB, ~0,1 s — e indexa `id → token`. Consequências boas: o lab funciona com o
professor servindo, funciona na M4, não precisa do `gen_lock` (dá para abrir o lab enquanto
uma resposta é gerada no chat) e o `lm_head` ser a *mesma matriz*
([[Inicialização e embeddings tied]]) deixa o passo 5 do curso já anunciado no passo 1.

**3. O lab mostra o invólucro, não o prompt do treino.** `montar_prompt_lab()` devolve
`<|user|>texto<|assistant|>` — **sem o `Pergunta: `** que o dataset ensinou. É de propósito: o
rótulo é detalhe do dataset, e o que a tela quer mostrar é o par de tokens especiais cercando o
texto do aluno. Consequência honesta (e boa para a aula): sem o espaço, a primeira palavra cai
num id diferente — `Quanto` é 6.666, `ĠQuanto` é 8.579 — porque **o espaço faz parte do token**.
Para *gerar* continuamos usando `montar_prompt()` (com `Contexto:`/`Pergunta:`), e o teste que
trava o formato do prompt do chat continua lá. No modelo base o invólucro não existe:
`<|user|>` não está no vocab de 16.384 e viraria `<`, `|`, `user`… — a tela avisa disso.

**4. Checagem de vocab obrigatória.** O base tem 16.384 tokens e o SFT 16.387 (os 3 de chat
são `added_tokens`, [[Treino SFT]]). Tokenizar com um tokenizer e desenhar a ficha com a
tabela do outro daria o vetor do token errado — silenciosamente. `Server.tabela_de()` recusa
o par quando `embed.weight` e `tokenizer.get_vocab_size()` não batem.

## O contrato (o que o front-end recebe)

```
POST /api/lab/tokenize {texto, modelo}
  → {prompt, chat,                      // chat=False no base: sem invólucro de chat
     tokens:[{i, id, texto, superficie, bytes, offset, kind, acentuado, repetido}],
     resumo:{chars, bytes, n_tokens, chars_por_token, por_categoria, acentuados,
             especiais, repetidos, distintos, vocab, cortado_em}}
GET  /api/lab/embedding?modelo=sft&id=4560
  → {id, token, texto, bytes, dim, valores[896], norma, min, max, std, escala,
     norma_media, dim_mais_ativa, vizinhos:[{id, token, texto, cos, norma}]}
GET  /api/lab/lookup?modelo=sft&q=ção
```

Tetos: 4.000 caracteres no pedido e 512 chips desenhados (`cortado_em` avisa). A lógica toda
mora em `pingo/didatic.py` (puro, testado em CPU); a página é `web/lab.html`; as rotas em
`scripts/webui.py` são finas.

## O que os números reais do Pingo mostram nesta tela

Medidos em `runs/sft-120m-ep1/final` (tabela fp32, 16.387 × 896):

- **vizinhos de `Ġinvestir`**: `Ġapostar` 0,45 · `Ġgastar` 0,42 · `Ġcomprar` 0,41 ·
  `Ġcontratar` 0,37 · `Ġinvestimento` 0,36. Os de `Ġpresidente`: `ĠPresidente` 0,53 ·
  `Ġprefeito` 0,45 · `Ġdiretor` 0,44 · `Ġsecretário` 0,42 · `Ġministro` 0,41. Os de `ĠJoão`:
  `ĠPedro` 0,43 · `João` 0,42 · `ĠJosé` 0,40 · `ĠPaulo` 0,35 · `ĠTiago` 0,33. É o slide que
  fecha o passo 1 — e repare que `João` sem espaço é outro token, logo ao lado.
- **custo do acento**: `até` é 1 token; `É` sozinho virou `ĠÃī` (espaço + 2 bytes); `≈`
  virou **3 tokens de byte puro** (`â`,`ī`,`Ī`). A barra de avisos da tela conta isso sozinha
  (`bytes > chars` ⇒ "acento em UTF-8 não é um símbolo só").
- **os tokens de chat quase não se moveram**: norma 0,235 (`<|user|>`) contra média da tabela
  1,14 — e cosseno **0,97 com a média do vocab**. Ou seja: depois de 1 época de SFT eles
  ainda estão quase onde o `resize` os colocou ([[Treino SFT]]: init na média). Dá para mostrar
  o porquê da [[Avaliação SFT|recusa fraca]] sem dizer uma palavra de loss.
- `Ċ` (quebra de linha) tem `<|eos|>` como vizinho nº 1 (0,35) — outro bom achado para aula.

## Próximas etapas da esteira (critérios no `PLAN.md`, passo 12)

`② embedding (896)` · `③ RoPE + atenção` · `④ RMSNorm + SwiGLU` · `⑤ lm head — a mesma
tabela` · `⑥ amostragem` · `⑦ loop`. As etapas ③ em diante vão precisar de *forward com
captura de intermediários* — aí sim o laboratório vai tocar a GPU, e aí vale o aviso de
sempre ([[Infra e GPU]]).

## Ver também

- [[WebUI]] — o chat (Flask + SSE); o lab é a segunda página do mesmo servidor
- [[Tokenizer BPE byte-level]] — por que o vocab é de bytes e o que é byte fallback
- [[Inicialização e embeddings tied]] — a tabela que a ficha desenha é também o lm head
- [[Gotchas]] — as armadilhas desta feature (offsets, base sem `tokenizer.json`)
