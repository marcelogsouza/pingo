---
tags: [pingo, sft, ferramenta, curso, tokenizer]
etapa: 0
atualizado: 2026-11-01
---

# Modo didático (`/lab`) — passos ①–⑥: tokenização, embeddings, RoPE + atenção, bloco, lm head, amostragem

O Pingo é um **projeto-curso** ([[Visão e filosofia]]): o valor está tanto no modelo quanto em
conseguir *mostrar* o que acontece dentro dele. O modo didático é a tela que faz isso — um
"laboratório" na mesma WebUI, com uma esteira de etapas no topo (tokenização → embeddings →
atenção → MLP → lm head → amostragem → loop) que **funciona como abas**: os passos ①–⑥
estão prontos, ⑦ aparece marcado "em breve".

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

## O que a tela faz (passo ④)

| elemento | por quê |
|---|---|
| **a régua do RMSNorm** (uma linha por token, 4 barras) | ‖x‖ entra torto, o **RMSNorm puro** devolve √896 = 29,93 para *todos* os tokens (a coluna verde sai com barras iguais — é a definição de normalizar, medida do forward, não figura), o **γ** modula dimensão a dimensão, e a última barra é o Δ que a camada somou (atenção + MLP) |
| **γ desenhado como faixa**, alternável norma-da-atenção / norma-do-MLP | o γ é o modelo opinando sobre quais dimensões valem; com média/min/max/σ ao lado, o aluno vê que a opinião da camada 1 é quase unânime (0,86–0,93) — e procura as camadas onde ela não é |
| **as três faixas do SwiGLU** — gate, up e o produto, na *mesma escala*, uma sobre a outra | `down(silu(gate)·up)` vira uma imagem só: o produto tem picos onde gate e up concordam e vales quase zero onde o gate fechou — a esparsidade aparece entre as três |
| **contas do token** | "portas abertas: 151 de 2.304" (força ≥ 10% da mais forte) e "‖Δ‖ num resíduo de ‖x‖" — a lição de que o modelo guarda fato como um punhado de neurônios, e de que cada camada soma pouco (por isso são 12) |
| controles camada(12) × token, **sem novo forward** | a captura do ③ (chave `modelo+ids`) serve os dois: trocar camada no ④ custa 12 ms na CPU |

## O que a tela faz (passo ⑤)

Em palavras simples: depois da última camada, cada token da frase é **um vetor só**. O
`lm_head` multiplica esse vetor pela **mesma tabela de 16.387 × 896 do passo ①** — e cada
linha do vocab recebe uma **nota**, o *logit*. Só isso: nenhum peso novo (isso é o *tied*),
e nenhuma escolha — quem escolhe entre as notas é o passo ⑥. A aba mostra as notas da posição
escolhida e prova que a tabela é a mesma de onde o curso começou.

| elemento | por quê |
|---|---|
| **chips de posição** | os logits dependem da posição: cada linha da frase prediz o próximo token, e a última (`<\|assistant\|>`) é o cardápio da geração — é o ponto onde o chat está quando o ⑥ entrar |
| **a última régua** (‖x₂‖ → RMSNorm puro → com γ) | o modelo ainda aplica **um RMSNorm final** antes da tabela; a lição do ④ vale para o vetor que o lm head vai ler, e a faixa do **γ final** desenhada mostra a opinião dele |
| **a régua de 16.387 logits** | o vocab inteiro de uma vez em 128 pingos (cada um, o maior \|valor\| de um grupo de ~129 linhas) — a imagem de que o lm head não é mais uma camada: é o produto escalar com a **mesma tabela do passo ①** |
| **os 10 que ganharam** | os candidatos que o passo ⑥ vai pesar; clicar num token abre a ficha dele — a mesma ficha/tabela do ①, fechando o ciclo do curso |
| **o próximo token real: nota e posto** | professor-forcing honesto: "o que o modelo achava que vinha aqui" vs o que a frase tinha — em `Ġinvestir → até`, o sucessor real ficou em **5º de 16.387** |
| **a mesma tabela lida duas vezes** | lado a lado: a prévia do ② (vetor inicial × tabela) e o logit real do ⑤ (vetor final × tabela) — se as ordens fossem iguais, as 12 camadas não teriam feito nada |

## O que a tela faz (passo ⑥)

Em palavras simples: o lm head deu notas, mas nota não é escolha. O passo ⑥ é a **roleta**:
cada token do vocab recebe uma fatia de **360° × a sua probabilidade**, o aluno gira (o
sorteio é o `torch.multinomial` do `sample_next` do servidor, não um `Math.random` da
página) ou **escolhe o token com a mão** — e o card do **desdobramento** mostra onde aquela
escolha vai dar, deixando o modelo girar sozinho até o `<|end|>`.

| elemento | por quê |
|---|---|
| **sliders T / top-k / top-p / penalidade + semente** | os filtros do chat, na tela do curso; mexer num slider refaz a roda em ~2 ms (a captura está em cache) — o efeito aparece na hora. A semente torna o giro **reproduzível**: mesma semente, mesmo token |
| **a roleta (canvas)** | a distribuição de UM token desenhada em graus; o ponteiro dourado em cima, a roda gira ~2 voltas e para na fatia sorteada; cada fatia grande mostra o token e o %; o resto vira a **fatia cinza** `+N tokens` — e a soma dos graus fecha em 360° (teste) |
| **o cardápio clicável** | os 24 primeiros candidatos com logit, probabilidade e **graus**; clicar **força** o token na história — inclusive um cortado pelo filtro (fatia 0°, o `p_bruto` conta a história, e o posto denuncia: rank > retidos) |
| **a história com 🎲/✋** | cada token gerado volta como entrada (é o ⑦ em embrião) com a marca de quem escolheu; desfazer/recomeçar deixam navegar a árvore |
| **o desdobramento** | `generate_stream` puro (KV cache incluso) dali para frente, com os MESMOS parâmetros: o card mostra o que a escolha puxa até o `<\|end\|>` ou o teto — é o "mostro um passo, mas conto o que vem depois" |
| **fim honesto** | se o sorteado for `<|end|>` (id 16.386), a roda para com o aviso — foi o SFT que ensinou isso |

É a única aba que **escreve** estado no servidor (uma captura nova por posição), mas não
segura sessão entre requests: a roleta é *stateless* — o cliente manda a história inteira e
a `captura_atencao` (chave `modelo+ids`) faz o resto do cache. Entrar no ⑤ com a mesma frase
não refaz o forward nenhum, e vice-versa.

## As dez decisões que importam

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
fused e nunca devolve pesos (e desde o ④, também com as chaves do bloco inteiro — decisão 9).
Sem `capture` o forward não muda uma linha: o teste exige
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

**9. O ④ não pediu um forward novo — a carona foi estendida, não bifurcada.** O mesmo
`capture=[{} por bloco]` do ③ ganhou as chaves do bloco: `x0/n1p/n1/a/x1/n2p/n2/g/u/s/m/x2`
(fluxo residual, as duas normas — `n*p` = **antes do γ** — e o SwiGLU). A conta não foi
duplicada: `RMSNorm`, `SwiGLU` e o bloco recebem o dict e o preenchem **na passagem real**,
sem recalcular nada — e o teste continua exigindo logits *bit a bit* iguais com a captura
pedida, mais `x2` do bloco *k* == `x0` do bloco *k+1* (bit a bit) e `‖n1p‖ = √d` para todos
os tokens. Capturar o RMSNorm **antes do γ** é o ponto: a lição é "todo vetor passa a medir
√d", e reconstruir esse vetor dividindo `n1` por γ seria desonesto com vetores minúsculos
(ε vira piso). Consequência de projeto: a captura cacheada por `modelo+ids` serve o ③ e o ④
com um único forward — a segunda camada do ④ responde em 12 ms na CPU.

**10. O ⑤ também é carona — e o tie sai provado com número.** `PingoLM.forward` com `capture`
agora escreve `nfinp/nfin` (a norm final **antes/depois do γ**, pelas mãos do próprio
`RMSNorm`) no **último** dict, e o `Server.captura_atencao` guarda também os **logits do
próprio forward** (`out.logits`, `[T, vocab]`): o ⑤ não recomputa o lm head, mostra o que o
modelo produziu. O teste exige `logits == nfin @ embed.weightᵀ`, e no checkpoint real a
recomputação com a tabela lida do `safetensors` na CPU bate com os logits do forward
**bit a bit (maxdiff 0,0)** — "o lm head é a tabela do passo ①" deixou de ser frase de aula e
virou asserção da suíte. A captura continua UMA por `modelo+ids` (agora serve ③, ④ e ⑤): o
forward de 11 tokens custa 0,09 s na CPU quente e cada posição nova do ⑤, 16 ms. O lm head
não escolhe nada — só dá notas; a escolha é o ⑥, e é por isso que o card mostra o **rank** do
sucessor real da frase (professor forcing): em `investir → até`, o token certo ficou em 5º de
16.387 — quase sempre o modelo sabia, e dá para *mostrar* o quase.

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
POST /api/lab/bloco {modelo, ids[], camada, i}   // ④: a MESMA captura do ③ (gen_lock!)
   → {n, d, hidden, bins, raiz_d, i, camada, n_camadas,
      tokens:[{i, id, token, texto, x0, n1p, n1, a, x1, n2p, n2, m, x2}],  // ‖·‖ de cada passo
      gamma:{attn:{media,min,max,std,faixa[bins]}, mlp:{…}},
      mlp:{g[bins], u[bins], s[bins], escala, maior{j,valor}, abertas, forca_max,
           delta, residuo}}
POST /api/lab/lmhead {modelo, ids[], i, k?}      // ⑤: a MESMA captura do ③ (gen_lock!)
   → {n, d, vocab, bins, i, raiz_d, modelo, n_camadas,
      tokens:[{i, id, token, texto, x2, nfinp, nfin}],   // a régua da norm final
      gamma:{media,min,max,std,faixa[bins]},             // γ da norm final
      top:[{id, token, texto, logit}],                   // os k (10) maiores logits
      proximo:{id, token, texto, logit, rank}|None,      // o sucessor real na frase (⑥ pide)
      faixa[bins], escala, estat:{media,min,max,std}, norma:{maior_x2, maior_nfin}}
POST /api/lab/amostra {modelo, ids[], acao:menu|girar|forcar, token?, temperature?, top_k?,
                        top_p?, repetition_penalty?, seed?}   // ⑥: a MESMA captura (gen_lock!)
   → {modelo, n, cortado_em, acao, vocab, retidos, p_retidos,
      filtros:{temperatura, top_k, top_p, repeticao}, semente?,
      candidatos:[{id, rank, token, texto, kind, logit, prob, graus, p_bruto}],
      outros:{n, prob, graus}|None, maior:{…},
      sorteado:{modo:roleta|mão, dentro, …}|None, end_id?}   // end_id só em giro/forço, no SFT
POST /api/lab/desdobrar {modelo, ids[], max_new_tokens?(≤96), …mesmos parâmetros}
   → {modelo, parada:max_tokens|end_token, n_novos, segundos, texto,   // generate_stream puro
      tokens:[{id, token, texto}]}
```

Tetos: 4.000 caracteres no pedido, 512 chips desenhados (`cortado_em` avisa), 96 linhas de
matriz (`MATRIZ_LINHAS`) e 64 tokens na captura de atenção (`MAX_ATTN_TOKENS`; ③, ④ e ⑤
dividem o teto e a captura). O ⑥ sobe o teto da captura para **160 tokens** entre frase e
gerados (`MAX_AMOSTRA_TOKENS`) e o desdobramento a 96 novos — a captura do ⑥ é a mesma do
③–⑤, só com chave maior. A lógica toda mora em `pingo/didatic.py` (puro, testado em CPU);
a página é `web/lab.html`; as rotas em `scripts/webui.py` são finas.

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

No passo ③ (medidos no checkpoint SFT; 11 tokens × 12 camadas × 14 cabeças: **0,66 s** na CPU,
**0,64 s** na GPU incluindo o load, e **0,004 s** para a 2ª camada porque a captura fica em
cache por `modelo+ids` — os valores batem dígito a dígito entre CPU e GPU, tudo em fp32):

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

No passo ④ (medidos no mesmo checkpoint e na mesma frase de 11 tokens; 1ª chamada 0,17 s —
o modelo e o `safetensors` já quentes — e **12 ms** para a 2ª camada, pois a captura do ③
serve os dois):

- **a régua é literal**: na camada 1 os tokens de conteúdo entram com ‖x‖ de 0,94 a 1,29 e os
  de chat com **0,235/0,236** (a norma baixinha do passo ②); depois do RMSNorm puro todos
  medem **29,92** — menos os dois de chat, que dão **29,69**: num vetor tão pequeno a média
  quadrática (6×10⁻⁵) é só ~60× o `eps` (10⁻⁶), e o piso de 1,6% aparece na tela. É o ε dos
  [[RMSNorm]] trabalhando à vista.
- **o γ da camada 1 quase não opina**: norma-do-MLP média 0,891 (min 0,858, max 0,932, σ só
  **0,011**) e norma-da-atenção 0,864 (0,754–0,934) — a faixa sai praticamente chata, e é uma
  pergunta pronta para o aluno: *em qual camada o γ deixa de ser chato?*
- **cada camada soma pouco — por isso são 12**: na camada 1 o Δ da atenção é 0,46–1,16 e o do
  MLP 3,1–5,3 sobre um resíduo de ~1; na camada 12 o resíduo já cresce **100×** (‖x‖ de 19 a
  107!) e o MLP soma 30,2 num resíduo de 34,4 — a última camada não é delicada. Bônus
  honesto: quem chega com o **maior** resíduo na última camada é o `<|user|>` (107).
- **o gate abre pouco**: para `Ġinvestir` na camada 1, **151 de 2.304** portas têm força
  ≥ 10% da mais forte (#1027, valor 2,23); na camada 12, 185 abertas e a mais forte vale
  14,07. O `<|user|>` abre **490** — o token mal treinado é também o mais difuso.

No passo ⑤ (medidos no mesmo checkpoint e na mesma frase; forward+captura **0,09 s** na CPU
quente — os 0,66 s do ③ a frio continuam o teto — e **16 ms** por posição nova, com a
captura em cache; `MAX_ATTN_TOKENS` continua 64):

- **a última régua também é literal, e aqui o ε não salva ninguém**: os vetores que chegam
  ao lm head medem ‖x₂‖ de **28,8 a 57,7** (o maior é o `<|user|>`, o mesmo torto do ④), e o
  RMSNorm puro devolve **29,93 a todos os 11 tokens** — sem o piso 29,69 dos tokens de chat
  do ④: depois de 12 camadas nenhum vetor é pequeno o bastante para o ε pesar.
- **o γ final é o botão de volume do modelo**: média **1,610** (min 1,276, max 1,695, σ só
  **0,061**) — quase não distingue dimensões (σ), mas estica o vetor todo ~1,6×: todas as
  normas saem 29,93 → **46,3–47,6**. Como a nota do lm head é produto escalar, isso multiplica
  todos os logits: é o oposto do γ da camada 1 (≈0,89, que quase encolhia).
- **o tie provado com número**: recomputar `nfin @ tabelaᵀ` com a tabela lida do
  `safetensors` na CPU devolve os logits do forward **bit a bit (maxdiff 0,0)**. A frase
  "o lm head é a mesma tabela do passo ①" agora tem teste na suíte.
- **a posição da geração** (o `<|assistant|>` final): campeão `A` logit **13,34**, depois
  `O` 10,62 e `NÃ£o` (= `Não`) 10,40; a massa tem média −3,40, σ 2,40 e mín −12,07. O cardápio do ⑥.
- **professor forcing**: para `Ġinvestir`, o sucessor real da frase (`ĠatÃ©`, o ` até`) levou
  logit **9,50 — posto 5º de 16.387**; o favorito do modelo era `ĠR` (11,88 — o fantasma do
  `R$`, o resto do top-3 é `ĠUS` e `Ġde`). O modelo quase sempre sabia a palavra e o rank
  mostra o quase.
- **② × ⑤, a mesma tabela lida duas vezes**: para `Ġinvestir`, o campeão da prévia do ②
  (vetor inicial) é *ele mesmo* (1,027 = ‖e‖²); no vetor final (⑤) o campeão é outro mundo
  (`ĠR`, 11,88). As camadas não criaram pesos novos: mexeram no vetor.

No passo ⑥ (medidos no mesmo checkpoint e na mesma frase, com o servidor em `--device cpu` e
o professor de pé — o modelo tiny de teste da suíte também exercita as duas rotas):

- **a roda fecha**: no cardápio do `<|assistant|>` final (T=1, sem filtro), os 24 candidatos
  + a fatia cinza somam **360,02°** (o erro é dos graus arredondados). O campeão `A` leva
  **285,4°** de 360° (p 0,793, logit 13,34 — o mesmo logit que o ⑤ anunciou); `O` leva
  18,8°, `Não` 15,2°, e os **16.363 restantes cabem em 8,1°** de fatia cinza. A pergunta
  pronta para o aluno: *quantos tokens cabem naquela fatia que mal se vê?*
- **o filtro visto na roda**: T=0,7 + top-p 0,9 → **retidos 1** — com a temperatura afiada, o
  nucleus do campeão já soma 0,93 sozinho e a roleta vira uma fatia só (a lição de que
  temperatura e top-p agem *em conjunto*, não cada um no seu quadrado). Forçar `Ġinvestir`
  (id 4.560) com top-k=3: fatia 0°, `prob` 0, mas `p_bruto` 8×10⁻⁶ e posto 4 — a tela mostra
  o token *morreu no filtro, não no sorteio*.
- **velocidade** (CPU, professor ocupando a VRAM): primeiro menu 0,07 s (forward+captura de
  11 tokens quentes); **girar 35 ms** (a captura do ⑤ está em cache, chave `modelo+ids`);
  mexer nos sliders **2 ms** (só re-faz o softmax sobre os 16.387 — a roda acompanha o dedo);
  menu da posição seguinte 0,07 s (forward novo de 12 tokens). Desdobramento greedy de 32
  novos: parou no `<|end|>` com **29 tokens em 0,64 s** — o KV cache carregando a história.
- **a história que o modelo contou sozinho**: girando T=0,7/top-p 0,9 e depois desdobrando
  com penalidade 1,15, da pergunta `Quanto a Companhia vai investir até 2028?` saiu
  `A Companhia de Assistência Técnica e Extensão Rural do Maranhão (Cedae) vai investir
  R$ 1,5 milhão em 2027.<|end|>` — e com o `<|end|>` sorteado no loop interativo a roda
  fecha com o aviso, exatamente como no chat.
- **bug consertado a caminho do ⑥**: o `sample_next` fazia o acumulado do top-p sobre os
  **logits**, não sobre as **probabilidades** — com logit positivo, `top_p=0,9` cortava tudo
  menos o favorito (com negativo, não cortava nada). A roleta obrigou a conta certa; o chat
  (que passava `top_p` do slider direto para o `sample_next`) ganhou filtro de verdade. O
  teste `test_sample_next_top_p_e_nucleus_de_probabilidade` (em `test_model.py`) trava o
  nucleus {0,1} de uma linha em que o jeito velho devolveria só {0}. O padrão do chat
  (`top_p=1`) nunca era afetado — os runs e o relatório de 2,507 não mudam.

## Próximas etapas da esteira (critérios no `PLAN.md`, passo 12)

`⑦ loop` — o token escolhido volta como entrada e o **KV cache** entra em cena de verdade: o
⑥ já mostra a história crescendo token a token (com `generate_stream` no desdobramento — a
conta e o cache estão prontos e testados); falta o passo didático de mostrar o *cache* — o
que o modelo reusa e o que ele recomputa a cada token novo. O andaime do ③–⑥ já resolveu a
parte difícil: o `capture` entrega os intermediários, e as rotas que carregam o modelo já
sabem obedecer ao `gen_lock` com teto de tokens. O GPU pesado continua proibido com o
professor servindo ([[Infra e GPU]]): a prática é `--device cpu` ou desligar o Strata com
aviso antes.

## Ver também

- [[WebUI]] — o chat (Flask + SSE); o lab é a segunda página do mesmo servidor
- [[Tokenizer BPE byte-level]] — por que o vocab é de bytes e o que é byte fallback
- [[RoPE]] — os 32 ponteiros do relógio: de onde vem cada `inv_freq`
- [[Atenção causal]] — o `softmax(QKᵀ/√d)` que o radar e a matriz desenham
- [[Inicialização e embeddings tied]] — a tabela que a ficha desenha é também o lm head
- [[Gotchas]] — as armadilhas desta feature (offsets, base sem `tokenizer.json`, o par do RoPE)
