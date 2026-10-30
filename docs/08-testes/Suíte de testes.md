---
tags: [pingo, testes, qualidade]
etapa: 0
atualizado: 2026-10-30
---

# Suíte de testes

```bash
uv run pytest -q      # 103 passed in ~13 s (medido em 2026-10-30)
```

**103 casos coletados de 95 funções de teste** em 10 arquivos. Tudo roda em **CPU** (config
tiny, tempo curto) — a GPU não é necessária para validar o código, só para treinar.

## Por arquivo

| Arquivo | Casos | O que protege |
|---|---|---|
| `test_smoke.py` | 2 | deps importam, pacote carrega |
| `test_tokenizer.py` | 3 | especiais (0/1/2), roundtrip **byte-exato**, byte fallback, compressão em pt |
| `test_data.py` | 5 | memmap, shapes, `y = shift(x)`, determinismo por rng, janela que cabe no shard, integração com os `.bin` reais |
| `test_model.py` | 16 | **contagem de params 127.555.456**, tie, shapes, init escalado, **invariância relativa do RoPE**, equivalência do KV cache (`<1e-5`), buffers pré-alocados, overflow, overfit→loss<0.05, sampling, generate, save/load bit-exato |
| `test_train.py` | 5 | lr warmup+cosine, grupos de decay, **decay com pesos de `--compile`**, **resume bit-exato** (6 == 3+3), payload do ckpt + logs |
| `test_sample.py` | 6 | ppl ≈ ln V no init, cache × naive com **mesmos tokens**, speedup reportado, corte de ctx, repetition penalty quebra o loop, relatório JSON |
| `test_gen_sft.py` | 21 | limpeza de parágrafos, filtro de fonte, chunking sem cortar frase, pesos dos tipos, todas as regras de `validate()`, reamostragem, `inacabável` não trava, prompt por modo, blocklists, HTTP fake, ponta a ponta |
| `test_validate_sft.py` | 16 | evidência tolerante a whitespace, `is_pt` curto, contrato do exemplo, shingles/Jaccard, **LSH acha quase-duplicata**, LSH barra trecho republicado, **não indexa rejeitados**, split determinístico por doc, ponta a ponta, idempotência dos tokens de chat, `render`, juiz fake |
| `test_train_sft.py` | 9 | **collate desloca labels** e mascara prompt/pad, loss ignora prompt/padding, padding não contamina posições reais, **resize na média + re-tie**, tokens de chat estáveis, descarte por tamanho, ckpt/best/final/log, **resume bit-exato**, dataset vazio avisa |
| `test_eval_sft.py` | 12 | F1 extremos/caixa, `apoio` com stem, limpeza de tokens de chat, recusa por modo, `terminou`, agregação sem contaminar com `None`, lista vazia, resolve tokenizer do modelo, ponta a ponta com modelo tiny |

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
