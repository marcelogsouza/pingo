---
tags: [pingo, sft, ferramenta, curso, tokenizer]
etapa: 0
atualizado: 2026-11-01
---

# Modo didático (`/lab`) — passos ①–③: tokenização, embeddings, RoPE + atenção

O Pingo é um **projeto-curso** ([[Visão e filosofia]]): o valor está tanto no modelo quanto em
conseguir *mostrar* o que acontece dentro dele. O modo didático é a tela que faz isso — um
"laboratório" na mesma WebUI, com uma esteira de etapas no topo (tokenização → embeddings →
atenção → MLP → lm head → amostragem → loop) que **funciona como abas**: os passos ①, ② e ③
estão prontos, ④–⑦ aparecem marcados "em breve".

```
http://127.0.0.1:8010/lab        (ou o IP da LAN, ver [[WebUI]])
```

## O que a tela faz (passo ①)

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

## O que a tela faz (passo ②)

| elemento | por quê |
|---|---|
| **a frase como matriz** | cada token vira uma linha desenhada (o vetor resumido, azul/laranja na mesma escala da ficha, norma ‖e‖ à direita) — "a frase é uma matriz `n × 896`" deixa de ser figura de linguagem; clicar na linha abre a ficha |
| **histograma das normas** | os 16.387 vetores numa barra só, com a média marcada — a distribuição é estreita (σ ≈ 0,10) *exceto* pelos tokens de chat, que despencam |
| **extremos com nome** | os vetores mais longos e mais curtos da tabela, clicáveis — o aluno vê quem é quem sem abstração |
| **a tabela lida de volta** | produto escalar `v·W` do vetor inicial: as notas que um `lm_head` tied daria, lado a lado com o cosseno — a ordem **muda** porque a nota inclui a norma. É o gancho anunciado do passo ⑤ |
| chips de escolha | no ② o aluno escolhe qual token da frase quer "ler de volta" |

## O que a tela faz (passo ③)

| elemento | por quê |
|---|---|
| **seletores camada(12) × cabeça(14) + chip do token** | a atenção só existe *dentro* de uma cabeça de uma camada; trocar seletor é trocar de microscópio sem recarregar a página |
| **o relógio do RoPE** | 32 ponteiros num círculo: cada um é um **par de coordenadas** `(xⱼ, xⱼ₊₃₂)` do Q capturado; cinza = como o vetor chega, colorido = girado à posição do slider. O ponteiro 1 dá uma volta a cada ~6,3 posições; o último, ~47 mil. É a matemática literal do `apply_rope`, não uma metáfora aproximada |
| **slider de posição + ▶ girar** | o aluno "passeia" o token pela frase e vê o vetor girar em tempo real; a velocidade relativa dos ponteiros é a lição |
| **ponto verde de conferência** | quando o slider bate na posição real do token, o ponteiro coincide com o `q` **capturado do modelo** — a aula não é figura, é conferência (o teste trava essa equivalência) |
| **radar do token** | 14 linhas (cabeças) × n colunas (tokens): quanto o token escolhido olhou para cada um dos anteriores; o futuro vem escuro (máscara causal). Clique numa linha abre a cabeça em sparkline com rótulos — o formato escolhido com o autor (calor + detalhe ao clicar) |
| **matriz n×n** | todos olhando todos, numa camada e cabeça: o triângulo superior vazio de propósito, a linha do token escolhido contornada — é o radar deitado |

## As oito decisões que importam

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

**2. O laboratório não encosta na GPU — nos passos ①–②.** A ficha do token precisa da tabela
de embeddings, e o modelo mora na VRAM disputada com o Strata. Então
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

**5. A matriz resume sem apagar os picos.** Desenhar 896 colunas por linha é ilegível; a
faixa resume cada vetor a 128 valores (`FAIXA_BINS`), e em cada grupo de dimensões sobrevive
o de **maior módulo**, não a média — a média apagaria justamente os picos que dão forma à
linha. A matriz desenha no máximo 96 linhas (`MATRIZ_LINHAS`; `cortado_em` avisa com o total)
e o pedido de ids é limitado a `MAX_LAB_TOKENS`. Detalhe de canvas que morde duas vezes: o
`clientWidth` do `<canvas>` é 0 antes do `appendChild` — primeiro cola na página, depois
desenha (ver [[Gotchas]]).

**6. Nota do lm_head ≠ cosseno — e a tela mostra os dois.** `lida_de_volta()` multiplica o
vetor pela tabela inteira (`pesos @ pesos[id]`: a nota que um lm_head tied daria *naquele*
vetor) e devolve duas ordenações: `por_pontos` (produto escalar, **inclui a norma** do
vizinho) e `por_cos` (direção pura). A ordem muda na tela, e essa é a lição. A honestidade
obrigatória: com o vetor *inicial* isso é prévia do passo ⑤, não o logit real — os logits de
verdade saem depois das 12 camadas; a nota diz isso no rodapé da seção.

**7. O ③ captura sem contaminar: `capture=[{} por bloco]`.** `PingoLM.forward(..., capture=)`
é uma carona opcional: cada dict sai com `q_lin/q/k_lin/k` (pré/pós-RoPE, `[B,H,T,D]`) e `w` —
as probabilidades de atenção **refeitas à mão**, porque `F.scaled_dot_product_attention` é
fused e nunca devolve pesos. Sem `capture` o forward não muda uma linha: o teste exige
logits **bit a bit** iguais com a captura pedida, e o `y` continua vindo do SDPA. A captura é
só do prefill (B=1, sem padding nem cache — `ValueError` fora disso). No servidor,
`Server.captura_atencao()` é a **única rota do lab que carrega o modelo**: roda sob
`gen_lock` (429 se o chat está gerando), teto de 64 tokens (`MAX_ATTN_TOKENS`, pesos são
14·n² por camada) e cache de uma captura — chave `modelo+ids`; trocou o texto, refaz o
forward (0,66 s para 11 tokens na CPU). Com o professor servindo, o servidor em
`--device cpu` serve igual; ver [[Gotchas]].

**8. O par que gira é `(xⱼ, xⱼ₊₃₂)`, não vizinhos.** O `apply_rope` do Pingo usa o estilo
`rotate_half` (HF): o `head_dim=64` divide em duas metades e o par rotacionado é a coordenada
`j` com a coordenada `j+32`. Um relógio que girasse `(x₀,x₁), (x₂,x₃)…` mostraria outra
matemática. O front-end pega `head_dim/2` da própria captura e o teste confere o `q` capturado
contra `apply_rope(q_lin)` — se divergir, a aula está mentindo. O ponto verde do relógio é a
versão visual desse teste.

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
POST /api/lab/matriz {modelo, ids[]}          // POST: 512 ids não caberiam numa URL
   → {dim, bins, escala, cortado_em, linhas:[{id, token, texto, norma, std, faixa[bins]}]}
GET  /api/lab/stats?modelo=sft
   → {vocab, dim, norma:{media, mediana, min, max, std}, histograma:[{desde, ate, cont}],
      maiores[6], menores[6], norma_vetor_medio}
GET  /api/lab/lida?modelo=sft&id=4560        // a tabela "lida de volta" (passo ②)
   → {id, token, texto, norma, pontos_consigo,
      por_pontos:[{id, token, texto, pontos, cos, norma}], por_cos:[…]}
POST /api/lab/attn {modelo, ids[], camada}    // ③: pesos reais de uma camada (gen_lock!)
   → {modelo, camada, n_camadas, heads, n, pesos[heads][n][n]}
GET  /api/lab/rope?modelo=sft&ids=4,9,…&camada=11&head=0&i=5   // ③: os vetores do relógio
   → {pos, head_dim, theta, inv_freq[32], q_lin[64], q[64], k_lin[64], k[64]}
```

Tetos: 4.000 caracteres no pedido, 512 chips desenhados (`cortado_em` avisa), 96 linhas de
matriz (`MATRIZ_LINHAS`) e 64 tokens na captura de atenção (`MAX_ATTN_TOKENS`). A lógica toda mora em `pingo/didatic.py` (puro, testado em CPU); a página é `web/lab.html`; as rotas em
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

No passo ② (medidos hoje na mesma tabela):

- **a tabela é estreita, com três buracos**: normas com média 1,139, mediana 1,126 e σ só
  **0,103** — e as **três menores normas dos 16.387 vetores são exatamente os tokens de
  chat**: `<|user|>` 0.235, `<|assistant|>` 0.236, `<|end|>` 0.382. É a mesma prova de que o
  SFT mal os moveu, agora visível como três barras solitárias no começo do histograma. Os
  maiores são sufixos e fragmentos — `mente` 1.577, `Ã¡c` 1.478, `es` 1.471 — não palavras.
- **na matriz da frase, o chat aparece achatado**: nas 11 linhas de `Quanto a Companhia vai
  investir até 2028?` os nove tokens de conteúdo têm norma 0,94–1,29 e os dois de chat são
  duas linhas visivelmente baixinhas (0,235/0,236).
- **a lida de volta ensina a diferença nota × cosseno**: para `Ġinvestir` o primeiro colocado
  pela nota é *ele mesmo* (1,027 = ‖e‖² — o vetor inicial se lê a si mesmo primeiro), depois
  `Ġapostar` 0,51 · `Ġgastar` 0,43; pelo cosseno a ordem sai sem si e com pesos diferentes.
  Para `<|assistant|>` a lista por nota é **lixo** (`<|pad|>` 0,18 e bytes puros — a norma é
  pequena demais para pontuar), mas pelo cosseno o nº 1 é `<|user|>` com **0,986**: os dois
  tokens de chat são quase o mesmo vetor — mais uma pista da [[Avaliação SFT|recusa fraca]].
- o vetor médio da tabela tem norma só 0,253 (contra média de normas 1,139): os vetores **não
  apontam para o mesmo lado** — a média cancela. Bom para desmanchar a ideia de que
  "um token é igual ao outro".

No passo ③ (medidos no checkpoint SFT, forward de 11 tokens na CPU em 0,66 s):

- **a rotação é fiel**: `‖q_lin‖ = ‖q‖ = 8,7258` para `Ġinvestir` (rotação não estica nada), e
  o par 0 na posição 5 calculado no navegador `(0,2747, −0,0189)` bate com o `q` capturado
  `(0,27472, −0,01886)` em 4 decimais — é o ponto verde do relógio.
- **as frequências cobrem 4 ordens de magnitude**: `inv_freq` vai de **1,0** (par 0: uma volta
  a cada 2π ≈ 6,28 posições) a **0,00013335** (par 31: ~47 mil posições por volta). O ctx do
  Pingo é 2.048 — os últimos ponteiros são quase decorativos nesta frase, e é isso que dá ao
  RoPE a noção de distância relativa ([[RoPE]]).
- **atenção que muda com a profundidade**: na camada 1, cabeça 1, o `Ġinvestir` divide a
  atenção quase igualmente (`<|user|>` 0,232, `Ġvai` 0,227, `ĠCompanhia` 0,138…); na camada 12
  ele dá **0,477 a si mesmo** e 0,189 ao `Ġvai` — a última camada concentra, a primeira
  dispersa. Slide pronto para a aula de [[Atenção causal]].

## Próximas etapas da esteira (critérios no `PLAN.md`, passo 12)

`④ RMSNorm + SwiGLU` · `⑤ lm head — a mesma tabela` (a prévia já roda no ②) · `⑥ amostragem`
· `⑦ loop`. O andaime do ③ já resolveu a parte difícil: o `capture` entrega os
intermediários que ④–⑥ precisam (é só estender a carona para `norm_attn(x)`, o SwiGLU e o
logit final), e as rotas do ③ já deixaram de ser intocáveis — carregam o modelo sob
`gen_lock` com teto de tokens. O GPU pesado continua proibido com o professor servindo
([[Infra e GPU]]): a prática é `--device cpu` ou desligar o Strata com aviso antes.

## Ver também

- [[WebUI]] — o chat (Flask + SSE); o lab é a segunda página do mesmo servidor
- [[Tokenizer BPE byte-level]] — por que o vocab é de bytes e o que é byte fallback
- [[RoPE]] — os 32 ponteiros do relógio: de onde vem cada `inv_freq`
- [[Atenção causal]] — o `softmax(QKᵀ/√d)` que o radar e a matriz desenham
- [[Inicialização e embeddings tied]] — a tabela que a ficha desenha é também o lm head
- [[Gotchas]] — as armadilhas desta feature (offsets, base sem `tokenizer.json`, o par do RoPE)
