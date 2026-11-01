---
tags: [pingo, referencia, gotchas, bugs]
atualizado: 2026-11-01
---

# Gotchas

A nota mais útil do vault: tudo que já nos custou tempo, organizado por área. Quase todos
viraram teste ou comentário no código.

> [!tip] Regra
> Encontrou um erro novo? Anote aqui **na hora**, com o sintoma. Um gotcha sem sintoma não
> ajuda ninguém.

---

## Ambiente e dados

**Idioma `pt` só existe em `2024-10` e `2023-40`.** Dumps mais recentes publicam apenas
`pt_meta`. Não insista. → [[Escolha do dataset]]

**Dataset é *gated* (gate auto).** Precisa `uv run hf auth login` + aceitar os termos 1× na
página. Sem token, o CDN responde **401**.

**Caminhos do repo HF já começam com `data/`.** Compor `data/` + caminho cru gera
`data/data/2024-10/...`. Use `path.removeprefix("data/")`. → [[Download do OSCAR]]

**`Path("x.jsonl.zst").stem` == `"x.jsonl"`.** `.stem` remove só **um** sufixo. Use
`name.removesuffix(".jsonl.zst")`.

**HF hub 2.x CLI:** o comando é `uv run hf auth whoami` (não `hf whoami`).

**`pt_part_N` não tem tamanho uniforme.** Muitas parts têm ~10 MB, não ~427 MB. Valide pelo
tamanho real reportado pelo Hub, nunca por média.

**Dedup do `process_data.py` é por execução.** O set `seen` vive em memória; rodar duas vezes
não barra duplicatas entre execuções.

**RAM é o gargalo, não a VRAM.** O Strata segura ~52 GB dos 62 GB → **sempre** processar por
row group, nunca carregar parquet inteiro. → [[Infra e GPU]]

**Disco:** `data/` = ~69 GB. Os `.jsonl.zst` (40 GB) são apagáveis com segurança depois da
tokenização; os `.bin` e os parquets **não** (o SFT ainda lê os parquets).

---

## Tokenizer

**`BpeTrainer` (tokenizers 0.23) não aceita `workers`/`chunk_size`.** Paralelismo é interno;
kwarg desconhecido só emite aviso e é ignorado. `initial_alphabet` é do **trainer**, não do
`models.BPE`. Contagem via `get_vocab_size()` (não `.vocab_size`).

**Byte-level mostra `Ġ`, `Ċ`, `Ã©`.** É o mapa byte→unicode do GPT-2; `decode()` faz roundtrip
exato. Cosmético, não bug. → [[Métricas de compressão]]

**Texto "spun" de SEO comprime igual a texto bom.** `chars/token` do nosso tokenizer deu 3,93
no lixo contra 4,13 de mediana: **não serve como detector de spam**. O que funcionou foi filtro
duro + veto do professor + blocklist de URL. → [[Geração do dataset]]

---

## Modelo e HF v5

**Tied weights precisam ser declarados.**
`_tied_weights_keys = {"lm_head.weight": "embed.weight"}` na classe, senão `save_pretrained`
estoura *"shared tensors not properly defined"*.

**`from_pretrained` materializa buffers non-persistent sem inicializar.** O `inv_freq` do RoPE
voltava **lixo** → logits errados mesmo com pesos bit-exatos. Solução: calcular on-the-fly, sem
buffer. → [[RoPE]]

**`post_init()` chama `_init_weights` em cada módulo.** Init custom vive em `_init_weights` (via
flag `_residual_proj`), nunca no `__init__` do bloco — senão é sobrescrito.

**`torch.Tensor.ones_()` não existe.** Para $\gamma=1$: `fill_(1.0)`.

**HF v5 devolve `CausalLMOutputWithPast`, que também indexa como tupla.** `out[0]` é a loss, mas
`out[0, :4]` estoura `TypeError: tuple indices must be integers`. Use `out.logits[...]`.

**`attention_mask` e `PingoCache` são mutuamente exclusivos.** O modelo levanta `ValueError`.
Cache é batch 1 sem padding; máscara é treino com padding.

**Máscara de padding com todas as chaves falsas → NaN.** Com pad à direita, uma query no padding
teria a linha inteira mascarada; por isso `attn_mask[:, :, :, 0] = True` (a coluna 0 é sempre um
token real). Não remova. → [[Atenção causal]]

**Não passe `attn_mask` **e** `is_causal=True` ao SDPA.** O resultado é ambíguo entre backends.

**`PingoCache`: todas as camadas escrevem na MESMA posição.** Se cada camada incrementasse
`cur_len`, a camada 2 escreveria deslocada. O avanço é único, em `commit()`. → [[KV cache]]

**Cache só aceita prefill (vazio) ou step de 1 token.** `t > 1` com cache cheio levanta erro
(exigiria máscara deslocada).

---

## Treino

**`--compile` prefixa os nomes (`_orig_mod.*`).** Quebra (a) o `no_decay` do `embed.weight` e
(b) as chaves do checkpoint. Sempre `getattr(model, "_orig_mod", model)`.
→ [[Benchmarks na GPU]]

**`--compile` é mais rápido E mais leve** (fusão reduz ativações): 55k → 101k tok/s e
28,5 GiB → 18,8 GiB. Não é só velocidade.

**O lr não está no checkpoint.** É recalculado por `lr_at(step, max_steps)`; `--resume` com
`--max-steps` diferente faz a curva de lr saltar **sem aviso**. → [[Loop de treino]]

**Resume precisa dos RNGs.** Python, `np.random.Generator.bit_generator.state`, torch CPU **e**
torch CUDA. Esquecer um e a "curva idêntica" quebra silenciosamente.

**Testes de `train.py` precisam de `--device cpu`.** A GPU existe nesta máquina e o teste de
resume exige determinismo de último bit.

**Val do OSCAR tem ruído** (cassino/SEO passam). ppl absoluta alta é esperada; serve como régua.

**`ppl` varia ~5% com o número de batches amostrados.** 20×B16 → 12,3; 25×B8 → 11,67. Comparar
ppl só com a mesma régua.

---

## Geração e avaliação

**Greedy em 120M entra em loop de repetição.** `repetition_penalty=1.15` resolve.

**Bench com `ctx == max_position_embeddings` não gera nada.** O cache para no limite e os dois
caminhos saem na hora (speedup 1,0×, sem erro). Corte para `max_ctx - n_new` e marque
`ctx_ajustado`. → [[Perplexidade e KV cache]]

**O ganho do KV cache NÃO é em batch=1.** Em B=1 o decode é limitado por latência de kernel
(pode dar 0,84×!). O ganho aparece em B=16 / ctx longo. Sempre compare `mesmos_tokens`.

**`generate(stop_token=...)` inclui o token de parada na saída.** Remova antes de medir
F1/apoio; meça `terminou` no texto cru. → [[Avaliação SFT]]

**Geração "vazia" é sintoma de modelo não treinado**, não de bug: um modelo sem SFT repete
`<|assistant|>` em loop.

**Métrica lexical de fidelidade não é prova.** `apoio` compara radicais de 5 letras: paráfrase
legítima derruba o número e invenção gramatical pode passar. Sem o `_stem`, "investe" ×
"investimentos" já derrubava (bug real). Use como triagem + `--judge`.

**`sample.py` precisa do tokenizer do modelo quando ele existe.** Sem
`resolve_tokenizer()`, o SFT seria avaliado com o vocab de 16.384 e os tokens de chat virariam
bytes soltos.

---

## SFT

**`labels` deslocados (o bug mais caro do projeto).** O modelo prevê o próximo token:
`labels[np_-1 : len(s)-1] = ids[np_:]`. Com `labels[t] = ids[t]` o treino "funciona" (acc 56%) e
a loss fica em **12** em vez de ~2,6, porque só precisa copiar o token do residual stream.
→ [[Treino SFT]]

**`{"type":"json_object", "schema":...}` é ignorado pelo llama.cpp.** Use o formato completo
`json_schema` + `strict: true`. → [[O professor local]]

**Thinking ligado queima o budget.** Com `max_tokens` curto, `content` volta vazio e o
raciocínio vai em `reasoning_content`. Use `chat_template_kwargs: {"enable_thinking": false}`.
O LFM2.5-2.6B anterior gastava 1.661 tokens de raciocínio para ~180 de conteúdo e não tinha
switch — foi descartado por isso.

**`chat_template_kwargs` é do llama.cpp, não do OpenRouter.** No OpenRouter, troque por
`reasoning: {"enabled": false}`.

**Resize do embedding quebra o tie.** Trocar `model.embed` não atualiza `lm_head.weight`;
refaça `model.lm_head.weight = emb.weight` e atualize `out_features`/`config.vocab_size`.

**Nunca regerar `tokenizer/tokenizer.json` para os tokens de chat.** O SFT adiciona os 3 tokens
em memória e salva uma cópia (16.387) em `runs/sft-*/`. Regerar invalidaria todos os `.bin`.

**MinHash: amostre shingles por HASH, nunca por posição.** `words[i::step]` desalinha os dois
textos e mede Jaccard 0,38 num espelho quase idêntico. Use `h % step == 0`.
→ [[Validação, dedup e splits]]

**No LSH, o bucket tem de ser lista** (colisão não pode se perder) e **só se indexa o exemplo
ACEITO** (senão ficam índices órfãos em `signatures[i]`).

**Recusa não tem verificação automática.** Formato e evidência vazia passam mesmo quando o
trecho responde à pergunta. `--judge-recusas` não é opcional. → [[Juiz de recusas]]

**`is_pt` precisa aceitar palavra de 1 letra.** `[^\W\d_]{2,}` ignorava "a", "o", "e", "é" e
reprovava pergunta curta legítima. Agora conta stopwords **+** palavras acentuadas (2 sinais).

**O modo misto muda duas regras.** Sem contexto, a pergunta não pode citar
"texto/trecho/documento/artigo/fonte" e a recusa tem outra frase.

**`--resume` do `gen_sft.py` engana o contador do `tqdm` — e o teto é por processo.** A barra
conta só os pares deste processo. Em 2026-10-30 marcava 5.804 com **8.959** linhas no arquivo.
Confira com `wc -l`. Como `--max-examples` limita só o que **este** processo aceita, retomar e
deixar a barra fechar produz `N + max_examples`: ia dar 33.154 pares; parei por `wc -l` em
30.000. Para retomar contando o arquivo: `--max-examples $((30000 - $(wc -l < arq)))`.
→ [[Geração do dataset]]

**SFT com 3 épocas e `lr 1e-4` superajusta em ~200 steps.** Em 2026-10-30 o run real (26.479
exemplos, batch efetivo 32) teve val mínimo no **step 200** (2,02) e terminou em 2,16, com train
loss 0,40 — e ppl **6,94×** no val do pré-treino. Com `--epochs 1 --lr 5e-5` o val desceu até o
fim (1,76) e o fator caiu a **1,91×** com F1 igual. Regra prática para este dataset: **1 época,
lr ≤ 5e-5**; se for mudar, olhe o val a cada 100 steps antes de deixar rodar.
→ [[Run do SFT-120M]]

**`SIGINT` no `gen_sft.py` grava o `stats.json`, mas o processo pode não morrer.** O laço para,
o `finally` fecha o arquivo (nada mais é escrito) e o `stats.json` sai — porém as threads do pool
podem ficar penduradas por minutos. Depois do `stats.json`, `SIGTERM` é seguro; só verifique com
`wc -l` que a contagem parou onde você queria.

**Scripts importados via `importlib` precisam de `sys.modules[nome] = mod`.** Sem isso,
`@dataclass` estoura `AttributeError: 'NoneType' object has no attribute '__dict__'`.

**`TokenData.get_batch` usa amostragem com reposição.** "1 época" = 1× tokens, ~63% dos tokens
distintos vistos.

---

## Processo e convivência

**Nunca rodar CUDA sem avisar.** Se `nvidia-smi` mostra ~31 GB usados, o Strata está de pé.
→ [[Infra e GPU]]

**Nunca dois escritores no mesmo JSONL.** `gen_sft.py` é idempotente com `--resume`, mas só se
houver **um** processo escrevendo.

**Não rodar `train.py` sem `--resume` em `runs/pt-120m/`.** Sobrescreve o melhor modelo. Existe
backup read-only `runs/pt-120m-BACKUP-2026-10-30/` com `SHA256SUMS.txt` conferido.

**Se a loss der NaN:** cortar lr pela metade (1,5e-4) e recomeçar do zero (resume de um ckpt com
NaN só propaga o NaN).

**Se der OOM:** baixar `B` e subir `accum` na mesma proporção (batch global igual → curva
idêntica).

## WebUI e modo didático

**`decode([id])` não devolve o texto de um token.** Num BPE byte-level o token pode ser
metade de um caractere UTF-8, e o `decode` devolve `U+FFFD`. O texto humano vem dos
**`offsets` do `encode()`, que são posições de caractere** do texto original (não de byte) —
`tokenizar()` fatia por eles. → [[Modo didático]]

**A superfície do token não é o texto.** O vocab guarda `ĠpreÃ§o`; o humano lê ` preço`. A
ponte é o alfabeto byte-level (`pingo.didatic.bytes_para_superficie` /
`superficie_para_bytes`), não um `encode("utf-8")` da superfície.

**O `pre_tokenizers.Whitespace()` do fixture estilhaça `<|user|>`.** Ele separa palavras e
pontuação (`<|`, `user`, `|>`), então nenhum token de chat sai de um WordLevel de mentira e o
teste vira verde por acidente. Nos testes do lab use
`pre_tokenizers.Split(pattern=Regex(r"\s+"), behavior="removed")`.

**O checkpoint base não tem `tokenizer.json`.** `Server.arquivo_tokenizador()` cai em
`tokenizer/tokenizer.json` (16.384); o SFT traz o dele (16.387). Tokenizar com um e desenhar
a ficha com a tabela do outro mostra o vetor errado **sem erro nenhum** — por isso
`Server.tabela_de()` confere `embed.weight × get_vocab_size()`.

**O `save_pretrained` do HF v5 grava só `embed.weight`.** O `lm_head` é *tied*, não duplicado:
a tabela que o laboratório desenha é essa chave do `model.safetensors` (58 MB em fp32).

**O `/lab` não usa o prompt do treino.** Ele mostra `<|user|>texto<|assistant|>`, sem o
`Pergunta: ` do dataset — e por isso a primeira palavra cai num id diferente do que o
chat manda (`Quanto` 6.666 × `ĠQuanto` 8.579): **o espaço faz parte do token**. Para
gerar, o chat continua em `montar_prompt()`; o lab é leitura, não geração.

**Canvas só desenha depois de estar na página.** O passo ② do `/lab` desenha um
`<canvas>` por linha da matriz usando `canvas.clientWidth` para a resolução — antes do
`appendChild` isso dá 0, o código cai no fallback de 420 px e o CSS estica tudo: borrão.
Primeiro cola na página, depois pinta (`pintarFaixa` depois do `appendChild`). Vale para o
histograma e para o mapa de atenção do passo ③.

**O modo didático não pode chamar `Server.carregar()` — exceto ③–④, por contrato.**
`carregar()` sobe o modelo para a VRAM — exatamente o que a regra do professor proíbe. As rotas
`/api/lab/*` usam `tokenizador_de()` e `tabela_de()`, que ficam na CPU. A exceção deliberada é
`/api/lab/attn|rope|bloco` → `Server.captura_atencao()`: carrega o modelo no device do servidor sob
`gen_lock` (429 se o chat está gerando), teto de 64 tokens e cache de uma captura. Com o
professor servindo, rode o servidor em `--device cpu` — o forward de 11 tokens leva 0,66 s (na
GPU livre são 0,64 s já incluindo o load; a captura fica em cache por `modelo+ids`, então
trocar só camada/cabeça custa microssegundos). Nunca derrube o professor por conta própria.

**O RoPE do Pingo não gira pares vizinhos de coordenadas.** Com `rotate_half` (estilo HF), o
par que roda junto é **(xⱼ, xⱼ₊₃₂)** — 32 dimensões adiante, não (x₂ᵢ, x₂ᵢ₊₁). Um relógio que
gire coordenadas adjacentes mostra outra matemática; no `/lab` o `head_dim/2` vem da captura e
o teste de equivalência com o `RoPE` do modelo fecha a porta.

**O dict do `capture` é um só por bloco e cresceu duas vezes.** ③ escreveu `q_lin/q/k_lin/k/w`
e ④ escreveu o bloco (`x0/n1p/n1/a/x1/n2p/n2/g/u/s/m/x2`) **no mesmo dict**; ⑤ ainda escreve
`nfinp/nfin` (a norm final antes/depois do γ) **só no último dict**. Quem iterar as chaves ou
fixá-las em teste precisa esperar as famílias (o teste fixa os conjuntos inteiros de
propósito: crescer a carona é decisão consciente, não acidente — e o `nfinp/nfin` fora do
último bloco seria um bug silencioso).

**A captura serve ③, ④ e ⑤, mas há UMA vaga.** `Server._captura` guarda o último forward
(chave `modelo+ids`, ~60 MB de tensores fp32 com 64 tokens, já contando os logits
`[64×16.387]` ≈ 4 MB que o ⑤ consome). Duas abas com frases diferentes se calcaram: cada
troca refaz o forward. Trocar camada/cabeça/token/posição na MESMA frase é de graça (4–16 ms).
Os logits do ⑤ são os do **próprio forward** (`out.logits`), não uma recomputação: se alguém
"otimizar" o `captura_atencao` chamando o modelo sem guardar a saída, a aba ⑤ desenha número
inventado e nada quebra no resto — só a aula.

## Ver também

- [[Suíte de testes]] — o que já está protegido
- [[Decisões (ADR)]] — o porquê das escolhas que geraram alguns desses bugs
- [[Pingo]] — voltar ao índice
