---
tags: [pingo, projeto, decisoes, adr]
etapa: 0
atualizado: 2026-10-30
---

# Decisões (ADR)

Registro das decisões que **custam caro reverter**. Formato leve de ADR: contexto → decisão →
alternativas → consequência. As decisões de desenho fino (por que `head_dim` 64, por que RoPE
θ=10k) estão nas notas de [[Arquitetura do Pingo]].

---

## ADR-01 · Arquitetura própria, HF só como scaffold (2026-10-29)

**Contexto.** O objetivo do projeto é *aprender*, não entregar um produto rápido.

**Decisão.** Escrever `RMSNorm`, `RoPE`, atenção, `SwiGLU` e `PingoLM` à mão. Herdar de
`PreTrainedModel`/`PretrainedConfig` apenas para ganhar `save_pretrained`, `from_pretrained`
e um `config.json` que o Hub/ecossistema entende.

**Alternativas.** (a) `GPT2LMHeadModel`/`LlamaForCausalLM` prontos — descartados: o projeto
viraria "fine-tuning". (b) Copiar `modeling_llama.py` — descartado: não se aprende nada.

**Consequência.** Herdamos também os *quirks* do HF v5 (tied weights declarados, buffer
non-persistent voltando lixo no `from_pretrained`) — ver [[Gotchas]].

---

## ADR-02 · Fase 1 = Pingo-120M, não 350M (2026-10-29)

**Contexto.** Restrição do usuário: **nada de run contínuo de 3 dias**. A RTX tem 32 GB.

**Decisão.** 12 camadas, d_model 896, 14 heads (head_dim 64), SwiGLU 2304 → 127.555.456
params ([[Contagem de parâmetros]]). Treino de ~1 noite.

**Alternativas.** Pingo-350M (24×1024) ≈ 3 dias de treino — fica para a Fase 2.

**Consequência.** O modelo é pequeno de propósito: a qualidade vem de acertar o pipeline, e
um ciclo curto permite erro → aprendizado → correção no mesmo dia. Foi o que aconteceu.

---

## ADR-03 · Dataset: Community OSCAR `pt`, dump `2024-10` (2026-10-29)

**Contexto.** O OSCAR-2301 original ficou suspenso para revisão manual (não é mais servido).

**Decisão.** `oscar-corpus/community-oscar`, gate **automático** (aceitar os termos 1×),
dump **`2024-10`** — o mais recente que ainda publica o idioma `pt` (os dumps seguintes só
trazem `pt_meta`).

**Alternativas.** `eduagarcia/cc_news_pt_v2` (só notícia), `statmt/cc100` `pt`, Wikipedia
(complemento). Ver [[Escolha do dataset]].

**Consequência.** Web **cru** → ruído (cassino, SEO, espelhos) que depois volta a assombrar
no SFT ([[Geração do dataset]]).

---

## ADR-04 · Sem filtro pt-BR × pt-PT (2026-10-29)

**Decisão.** Não separar variantes. O pt-BR domina naturalmente o corpus web em português.

**Consequência.** O modelo mistura "estava"/"está a ser" — visível nas amostras de
[[Perplexidade e KV cache]]. Aceito como característica, não como bug.

---

## ADR-05 · Tokenizer BPE byte-level, vocab 16.384 (2026-10-29)

**Decisão.** BPE byte-level (regex GPT-2 + `byte_fallback=True`), 3 especiais
`<|bos|> <|eos|> <|pad|>`, treinado em 1 GB do `split=train`.

**Por quê 16k.** Trade-off parâmetros × compressão: com d_model 896, cada 1k de vocab custa
896k params de embedding (e o mesmo no `lm_head`, mas *tied*, então uma vez só). 16k dá 4,091
chars/token em pt; um vocab de 32k comprimiria mais, mas comería ~14M params (11% do modelo)
em embeddings.

**Consequência.** A decisão de adicionar tokens de chat no SFT vira uma cirurgia real no
embedding ([[Treino SFT]]).

---

## ADR-06 · Amostrar 2,75B tokens dos ~15B disponíveis (2026-10-29)

**Decisão.** `prepare_tokens.py` embaralha os parquets (`seed=13`) e para em 2,75B tokens.
O `split=val` entra **inteiro** (292M).

**Razão.** Regra de bolso de ~20 tokens por parâmetro: 2,75B / 127,6M ≈ **21,6 tok/param**.
Ver [[Binarização e shards]].

**Consequência.** Sobram ~12B tokens não usados (e o dump reserva `2023-40` nunca foi
preciso). Espaço para escalar depois sem re-baixar nada.

---

## ADR-07 · Treino: bf16, AdamW wd 0.1 (sem decay em norm/embed), 0,52M tok/step (2026-10-29)

**Decisão.** `B=16 × ctx 2048 × accum 16` = 524.288 tokens por step; lr 3e-4 → 3e-5
(warmup 400, cosine); betas (0.9, 0.95); clip 1.0; `max_steps` = 1 época sobre os tokens.

**Consequência.** 5.248 steps, 7,56 h. Detalhes em [[Loop de treino]].

---

## ADR-08 · `torch.compile` ligado no run real (2026-10-29)

**Contexto.** Medido em `runs/bench-b16-compile`: 55k tok/s sem compile × **102k** com.

**Decisão.** Ligar no run. Também cai a VRAM (28,5 GiB → 18,8 GiB), porque a fusão reduz
ativações vivas.

**Consequência.** Todo código que mexe em `state_dict`/`named_parameters` precisa do
`getattr(model, "_orig_mod", model)` — bug real, ver [[Benchmarks na GPU]].

---

## ADR-09 · **SFT antes do 350M** (2026-10-30) — o plano virou

**Contexto.** A Fase 1 fechou com um modelo que gera pt coerente mas não responde perguntas
(nem segue formato). Escalar para 350M pede ~6B tokens e mais GPU; o gargalo hoje **não** é
tamanho, é **falta de formato/instrução**.

**Decisão.** Suspender o Passo 8 e fazer **SFT** com dataset sintético gerado por um modelo
professor local, a partir do próprio corpus do pré-treino.

**Consequências.** Pipeline novo de 5 etapas ([[Por que SFT]]); a GPU passa a servir dois
papéis (professor e aluno); o "esquecimento catastrófico" vira um risco a medir
([[Avaliação SFT]]).

---

## ADR-10 · Tokens de chat como **tokens especiais** (2026-10-30)

**Decisão tomada pelo usuário.** `<|user|>`, `<|assistant|>`, `<|end|>` entram no vocabulário
como ids novos (16.384 → 16.387), com `resize_token_embeddings` e as linhas novas iniciadas na
**média** dos embeddings antigos.

**Alternativas.** (a) usar strings normais ("User:"/"Assistant:") — sem resize, mas o modelo
gasta tokens em delimitadores e pode confundi-los com texto; (b) tokens de controle em
posições reservadas — mais cirurgia ainda.

**Consequência.** `tokenizer/tokenizer.json` **continua com 16.384**: os 3 tokens são
adicionados em memória no treino e salvos junto do run (`runs/sft-*/tokenizer.json`).
Regenerar o tokenizer nunca é a solução (mudaria os merges). Ver
[[Treino SFT]] e [[Gotchas]].

---

## ADR-11 · 30k pares, 5% de recusas, mistura 50/50 ancorado × sem contexto (2026-10-30)

**Decisão.** `--max-examples 30000 --recusa-frac 0.05 --ancorado-frac 0.5`, taxonomia
50% extração / 30% explicação / 15% resumo.

**Razão.** Com 127M de params, 20–50k pares já saturam; o gargalo é **qualidade**. Metade
ancorada ensina o uso tipo mini-RAG; metade sem contexto evita que o modelo só saiba responder
"olhando o trecho".

**Consequência.** Duas regras diferentes de recusa e de formulação de pergunta
([[Formato do exemplo]]).

---

## ADR-12 · Filtro de fonte **duro** + blocklist + teto por domínio (2026-10-30)

**Contexto.** O filtro do pré-treino (55,6% de aproveitamento) deixa passar spam adulto,
cassino e espelhos de filme pirata.

**Decisão.** Filtro bem mais duro para o SFT (tamanho ≥ 2000 chars, ≥ 4 parágrafos, dígitos ≤
25% das palavras, maiúsculas ≤ 30%, ≤ 4 links), **blocklist de URL e de conteúdo**, e teto de
5% dos exemplos por domínio.

**Consequência.** ~48 de 330 trechos do piloto foram vetados pelo próprio professor
(`status: invalido`) — o filtro de qualidade mais eficaz saiu de graça.

---

## ADR-13 · O juiz de recusas é **obrigatório**, não opcional (2026-10-30)

**Contexto.** Recusa é a única classe sem verificação automática: formato e evidência vazia
passam mesmo quando o trecho **responde** à pergunta. Aconteceu no dataset real.

**Decisão.** `validate_sft.py --judge-recusas` faz uma chamada extra por recusa (~5% do
dataset) e descarta as respondíveis. Detalhes em [[Juiz de recusas]].

---

## ADR-14 · Continuar com o professor **local** (2026-10-30)

**Contexto.** Comparação medida com 3 professores remotos no OpenRouter: o
`deepseek-v4.1-flash` aceita 97% dos trechos (contra ~75%) e custaria US$ 3–4 pelos 22k
pares.

**Decisão.** Manter o Qwen3.5-4B local, apesar de ~2–3× mais lento.

**Razão (o achado que decidiu).** Profesor melhor em prosa = **pior em veta spam**: o deepseek
aceitou página de cassino e de SEO, exatamente o que o veto do Qwen estava filtrando. Trocar
de professor injetaria lixo no dataset. Ver [[Professores remotos (OpenRouter)]].

---

## ADR-15 · Backup read-only do checkpoint (2026-10-30)

**Pedido explícito do usuário.** `runs/pt-120m-BACKUP-2026-10-30/` = cópia `dr-xr-xr-x` de
`runs/pt-120m/` com `SHA256SUMS.txt` conferido. Nenhum script escreve ali; o SFT grava em
`runs/sft-120m/`. **Preservar com carinho o modelo que funciona.**

---

## Ver também

- [[Visão e filosofia]] — o pano de fundo
- [[Linha do tempo]] — quando cada decisão foi tomada
- [[Gotchas]] — o preço de algumas delas
- [[Pingo]] — voltar ao índice
