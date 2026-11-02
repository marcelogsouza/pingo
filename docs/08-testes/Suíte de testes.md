---
tags: [pingo, testes, qualidade]
etapa: 0
atualizado: 2026-11-02
---

# Suíte de testes

```bash
uv run pytest -q      # 227 passed in ~85 s (medido em 2026-11-02; subiu de ~14 s quando os
                      # jobs de subprocess e o e2e do /treino entraram)
```

**227 casos** em 16 arquivos. Tudo roda em **CPU** (config tiny, tempo curto) — a GPU não é
necessária para validar o código, só para treinar. Os testes de job/disparo usam
`CUDA_VISIBLE_DEVICES=""` no filho: com o professor servindo, até o health check do
`opt.step()` do torch tenta `cuInit` e OOM-flaka.

## Por arquivo

| Arquivo | Casos | O que protege |
|---|---|---|
| `test_smoke.py` | 2 | deps importam, pacote carrega |
| `test_tokenizer.py` | 3 | especiais (0/1/2), roundtrip **byte-exato**, byte fallback, compressão em pt |
| `test_data.py` | 5 | memmap, shapes, `y = shift(x)`, determinismo por rng, janela que cabe no shard, integração com os `.bin` reais |
| `test_model.py` | 26 | **contagem de params 127.555.456**, tie, shapes, init escalado, **invariância relativa do RoPE**, equivalência do KV cache (`<1e-5`), buffers pré-alocados, overflow, overfit→loss<0.05, sampling (**nucleus do top-p = probabilidades**, o bug pego pela roleta do ⑥), generate, save/load bit-exato, **`capture` do ③**: logits **bit a bit** iguais com captura pedida, pesos que somam 1 com triângulo futuro zerado, `q` capturado == `apply_rope(q_lin)` do próprio modelo; **bloco do ④**: `x0` é o embedding, RMSNorm puro tem norma **√d**, `x2 = x1 + m`, `s = silu(g)·u`, e `x2` do bloco *k* == `x0` do bloco *k+1* (bit a bit); **lm head do ⑤**: só o último dict carrega `nfinp/nfin`, e **`logits == nfin @ embed.weightᵀ`** (o tied provado); **colunas congeladas do ⑦**: `snapshots()` devolve buffers cujas posições antigas ficam **bit a bit** iguais após steps de decode |
| `test_train.py` | 7 | lr warmup+cosine, grupos de decay, **decay com pesos de `--compile`**, **resume bit-exato** (6 == 3+3), payload do ckpt + logs, **parada graciosa**: `INTERROMPER` grava ckpt+evento sem `final/`, e **SIGTERM real** de subprocess → rc 130 → `--resume` conclui |
| `test_qa_dataset.py` | 7 | contrato JSONL aprovado/reprovado com motivos legíveis, campo candidato quando falta `content`, duplicata pelo **mesmo hash do `process_data.py`** (whitespace colapsado conta), avisos ≠ reprovação, amostragem reservatória, relatório JSON-safe |
| `test_jobs.py` | 7 | JobRunner: fila 1-por-vez (`JobOcupado`), SIGTERM no parar → `interrompido`, `falhou` com rc, persistência em `jobs.json`, **caça ao órfão no boot** (webui morreu com filho vivo → SIGTERM), `log_records` tolera linha pela metade, `hw()` sano |
| `test_treino_pipeline.py` | 2 | **corrente do treino-5 de ponta a ponta**: JSONL → `train_tokenizer --id --specials(chat) --jsonl` → `prepare_tokens --input jsonl` (split por hash estável, meta amarrado ao `tokenizer_id`) → `train.py` **derivando vocab do meta** → SFT com o mesmo tokenizer (`tokens de chat novos: 0`) |
| `test_treino_api.py` | 14 | contrato das rotas `/api/treino/*`: página, status/hw/log, datasets/QA (fora do repo = 404), tokenizadores/tokens listados, validações de disparo (nome, **runs protegidos 403**, out ocupado → 409 pedindo retomar, fila ocupada → **429 antes de qualquer validação**), mutex job↔chat (`Server.carregar` recusa), épocas→`--max-steps` conferido no argv, e um **pré-treino tiny real pela API** (subprocess CPU → `final/`, log, curva) |
| `test_sample.py` | 6 | ppl ≈ ln V no init, cache × naive com **mesmos tokens**, speedup reportado, corte de ctx, repetition penalty quebra o loop, relatório JSON |
| `test_gen_sft.py` | 21 | limpeza de parágrafos, filtro de fonte, chunking sem cortar frase, pesos dos tipos, todas as regras de `validate()`, reamostragem, `inacabável` não trava, prompt por modo, blocklists, HTTP fake, ponta a ponta |
| `test_validate_sft.py` | 24 | evidência tolerante a whitespace, `is_pt` curto, contrato do exemplo, shingles/Jaccard, **LSH acha quase-duplicata**, LSH barra trecho republicado, **não indexa rejeitados**, split determinístico por doc, ponta a ponta, idempotência dos tokens de chat, `render`, juiz fake |
| `test_train_sft.py` | 9 | **collate desloca labels** e mascara prompt/pad, loss ignora prompt/padding, padding não contamina posições reais, **resize na média + re-tie**, tokens de chat estáveis, descarte por tamanho, ckpt/best/final/log, **resume bit-exato**, dataset vazio avisa |
| `test_train_sft.py` (treino-1) | +2 | `INTERROMPER` → ckpt + evento sem `final/`; interromper → `--resume` conclui e grava o `tokenizer.json` do final |
| `test_eval_sft.py` | 12 | F1 extremos/caixa, `apoio` com stem, limpeza de tokens de chat, recusa por modo, `terminou`, agregação sem contaminar com `None`, lista vazia, resolve tokenizer do modelo, ponta a ponta com modelo tiny |
| `test_webui.py` | 47 | prompt igual ao do treino (**o teste que impede a UI de degradar o modelo**), protocolo SSE `meta`/`delta`/`done`, `429` da segunda geração, `400` de prompt grande, clamp dos parâmetros, `DeltaDecoder` incremental, troca de modelo que descarrega o anterior, rotas do `/lab` (invólucro `<\|user\|>…<\|assistant\|>` sem `Pergunta:`, embedding **sem carregar o modelo**, busca, vocab mismatch; passo ②: matriz linha-por-token com teto de ids, stats da tabela, lida tied nota×cosseno; passo ③: pesos reais causais por cabeça, cache da captura, `rope` confere a rotação, 429 do `gen_lock`; passo ④: régua/γ/faixas do bloco, `400`/`429`, e **a captura do ③ serve o ④** — um forward só; passo ⑤: régua final + top-k **conferido no `topk` da captura** + posto do sucessor real, `400`/`429`, e a **mesma captura serve ⑤**; passo ⑥: menu com os **logits da captura** e roda que fecha em 360°, **a mesma captura serve ⑥** (carona sem recomputar), semente que reproduz o giro, `forcar` dentro e fora do filtro, tetos de 160 ids e `429`, e o **desdobramento** — `generate_stream` até o `<\|end\|>`, reprodutível com semente; passo ⑦: o replay com o prefill + decode de `no_forward` 1, `entrou == escolhido` do passo anterior, `economia` com<sem cache, `fita` e `compara` conferidos na resposta, semente reproduz, tetos/`400`/`429`, e a rota do loop na página) |
| `test_didatic.py` | 33 | **alfabeto byte-level** (`ĠatÃ©` são os bytes de ` até`), categoria dos tokens, fatia pelos `offsets`, repetições e acentos, corte em 512, `embed.weight` lido do `.safetensors` na CPU, norma e vizinhos por cosseno ordenados, busca no vocab; passo ②: `faixa` preserva o **maior módulo** de cada grupo, matriz linha-por-token + corte honesto, histograma que soma o vocab, `lida_de_volta` = produto escalar conferido na mão; passo ④: `faixa_de` nos picos, `ficha_bloco` com a **régua √d**, γ com média/extremos e o SwiGLU do token; passo ⑤: `ficha_lmhead` com a **régua final √d**, `proximo` (posto do sucessor real conferido no `topk`) e `None` na última posição; passo ⑥: `menu_amostra` com a roda que **fecha em 360°**, T=0 = uma fatia, sementeira conferida no `sample_next` (o giro nunca sai do filtro; top-p = menor prefixo que soma p), semente reproduz, `forcar` fora do menu (fatia 0°, `p_bruto` > 0, rank > retidos) e a penalidade punindo no menu todo; passo ⑦: `roda_loop` **igual ao `generate_stream` token a token** (semente, com e sem penalidade), o caderno (`entrou` = escolhido anterior, `no_forward`/`escrito`/`kv_lidos`, decode cresce de 1 em 1), paradas `end_token`/`contexto`, a `fita` com ‖K‖/‖V‖ por camada e o `compara` sem cache (**maxdiff < 1e-4**, argmax igual), semente reproduz e os guardas (`[]`, frase maior que o contexto); integração com o tokenizer real (pula sem `runs/`) |

## Os cinco testes que valem por uma tarde de debugging

1. **`test_cache_equivalencia_prefill_e_steps`** — `max|logits_cache − logits_full| < 1e-5`.
   Se o cache estiver escrevendo deslocado, o benchmark de speedup seria uma ficção.
2. **`test_resume_reproduz_run_contínuo`** — pesos **bit-idênticos** entre 6 steps seguidos e
   3 + resume. É o que permite fatiar treino em noites.
3. **`test_collate_desloca_labels_e_mascara_prompt_e_pad`** — o bug mais caro do SFT (labels
   alinhados em $t$ em vez de $t+1$) fica impossível de reintroduzir sem falhar teste.
4. **`test_contagem_params_pingo120m`** — conta `sum(p.numel())` do modelo real, não uma fórmula
   paralela. Mudou o config, o teste grita.
5. **`test_lsh_barra_trecho_republicado`** — o bug de amostragem de shingles por posição
   (Jaccard 0,38 num espelho) só aparece em teste com dado sintético controlado.
6. **`test_sigterm_real_para_o_processo_e_resume_conclui`** — o sinal que a UI manda, dado a um
   `train.py` de verdade: rc 130, ckpt íntegro, resume conclui. Foi também o teste que revelou o
   flake do torch com a GPU alheia cheia: sem `CUDA_VISIBLE_DEVICES=""` no filho, o
   `opt.step()` chama `cuInit` no health check e morre de OOM **sem o teste pedir GPU**.

## Convenções

- **Config tiny** (`PingoConfig(hidden_size=64, layers=2, heads=4, intermediate=128)`) em tudo
  que constrói modelo: roda em segundos.
- **`--device cpu` explícito** nos testes de `train.py`: nesta máquina a GPU existe, e o teste
  de resume exige determinismo (kernels CUDA com atomics não são determinísticos no último bit).
- **Fixtures com `tokens_dir`** usam os `.bin` reais quando existem e **pulam** quando não
  (a M4 não tem `data/`).
- **Nada de rede**: professor e OpenRouter são `monkeypatch`ados. `test_ask_teacher_le_resposta_do_servidor`
  usa uma função fake, não HTTP.

## Scripts importados como módulo (`importlib`)

Os testes de `gen_sft.py` / `validate_sft.py` carregam o script com
`spec.loader.exec_module()`. **É obrigatório registrar em `sys.modules` antes:**

```python
sys.modules[nome] = mod
spec.loader.exec_module(mod)
```

Sem isso, o `@dataclass` do script estoura
`AttributeError: 'NoneType' object has no attribute '__dict__'`.

## Ver também

- [[Gotchas]] — os bugs que viraram teste
- [[Comandos e cheatsheet]] — como rodar
- [[Pingo]] — voltar ao índice
