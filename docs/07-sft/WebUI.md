---
tags: [pingo, sft, ferramenta, flask]
etapa: 0
atualizado: 2026-11-01
---

# WebUI (Flask)

Uma página de chat em `localhost` para conversar com o Pingo sem escrever código: o SFT
(instrução) e o base (continuação) na mesma tela, com streaming, contexto opcional e os
hiperparâmetros de amostragem à mão.

```bash
uv run python scripts/webui.py                     # 0.0.0.0:8010 (LAN + localhost)
uv run python scripts/webui.py --host 127.0.0.1    # só esta máquina
uv run python scripts/webui.py --device cpu --port 8080
uv run python scripts/webui.py --sft runs/sft-120m-ep1/final --base runs/pt-120m/final
```

O host padrão é `0.0.0.0`: a página responde **na rede local** — nesta máquina,
`http://192.168.0.101:8010` (o script imprime o IP no start, via truque do socket UDP, que
não envia pacote nenhum).

É a única parte do projeto que usa **Flask** (`uv add flask` → 3.1.3). O resto continua
PyTorch + tokenizers.

## O que a tela faz

| elemento | por quê |
|---|---|
| seletor **SFT / base** | comparar instrução × continuação no mesmo prompt é o teste mais honesto das duas fases |
| **Contexto (opcional)** | cola um trecho e o modelo recebe `Contexto: … / Pergunta: …` — é o mini-RAG para o qual metade do dataset foi gerada |
| **parâmetros** (temp, top-p, top-k, penalidade, máx. tokens) | 120M é sensível: sem `repetition_penalty` o greedy entra em loop |
| rodapé de cada resposta | `modelo · tokens · segundos · tok/s · 1º token` — o número que diz se a geração está saudável |
| botão parar | aborta a leitura do stream (o servidor encerra o gerador) |

## As três decisões que importam

**1. O prompt é o do treino.** `montar_prompt()` monta exatamente
`<|user|>Contexto: …\n\nPergunta: …<|assistant|>` (ou só `Pergunta: …`). Usar outro template
("Você é um assistente…") degradaria a resposta sem que ninguém entendesse o motivo — o
modelo só viu esse formato. No base, ao contrário, o prompt é o texto **cru**: ele não
aprende instrução, aprende a continuar texto.

**2. Streaming por SSE, token a token.** `pingo.generate_stream()` é o `generate()` de sempre,
mas emite cada id conforme sai; a rota devolve `text/event-stream` com `meta` → N × `delta` →
`done`. Na prática: **primeiro token em ~17 ms** e o texto aparecendo a ~130 tok/s (base,
RTX), em vez de 2 s de tela parada.

**3. Um modelo por vez, com lock.** A VRAM é compartilhada com o treino, então o servidor
mantém **um** checkpoint carregado (trocar descarrega o anterior e chama `empty_cache`) e um
`threading.Lock` **não bloqueante** por geração: a segunda pergunta simultânea recebe `429` em
vez de brigar por memória.

## As armadilhas que apareceram

> [!bug] Acento quebrado no meio do stream
> Decodificar o id sozinho pode cortar um caractere multibyte no meio ("não" → `n�o`). O
> `DeltaDecoder` decodifica a lista inteira e só emite o que é **prefixo** do anterior; se o
> texto novo termina em `U+FFFD`, segura o pedaço até o próximo token. Sem isso a UI sai
> salpicada de `�`.

> [!bug] `<|end|>` não é texto
> O token de fim de turno existe para o modelo parar, não para aparecer na tela: o laço
> compara com `end_id` **antes** de decodificar. `end_id` só é definido para o checkpoint SFT —
> o base nunca foi treinado com ele, então ali ele não significa nada.

> [!note] Porta 8000 ocupada
> Nesta máquina a 8000 já tem dono (outra aplicação). O default da WebUI é **8010**.

## Números medidos (2026-10-30, RTX PRO 4500)

| | base (`pt-120m`) |
|---|---|
| tokens/s no stream | ~130 |
| 1º token (TTFT) | 17 ms (prompt de 6 tokens) |
| VRAM do servidor | ~2 GB (modelo fp32 + cache) |
| roda junto do treino? | sim — treino 8,6 GB + UI 2 GB em 32 GB |

## Limites conhecidos

- É o **servidor de desenvolvimento** do Flask, **sem autenticação e sem TLS**: qualquer um na
  mesma rede que abrir a porta 8010 usa o modelo (e o `429` só limita a *uma* geração por vez).
  Serve para LAN doméstica; não faça port-forward disso para a internet. Para fechar de novo:
  `--host 127.0.0.1`.
- **Sem memória de conversa**: cada pergunta é independente (o modelo não tem contexto de
  turnos anteriores; o histórico na tela é só visual).
- 120M erra fato com naturalidade ("São Paulo, no estado do Rio de Janeiro") e responde
  curto. A WebUI existe para **ver** isso, não para esconder.

## Testes (23 casos, `tests/test_webui.py`)

Sem rede e sem o tokenizer real: um checkpoint tiny + um tokenizer WordLevel de mentira. O
que está travado: o **formato do prompt**, o **protocolo SSE** (`meta`/`delta`/`done`), o
`429` da segunda geração, o `400` de prompt grande, o clamp dos parâmetros, o `DeltaDecoder`
incremental, a troca de modelo que descarrega o anterior e as rotas do laboratório.

## O modo didático (`/lab`)

A segunda página do mesmo servidor: **tokenização colorida, ids, bytes e o vetor inicial de
cada token** — o passo 1 de uma esteira de etapas para o curso. Ela não usa GPU nem o lock de
geração (lê `embed.weight` do `.safetensors` na CPU), então dá para abrir o laboratório
enquanto o chat gera uma resposta. Nota completa: [[Modo didático]].

## Ver também

- [[Treino SFT]] — o formato do prompt que a UI reproduz
- [[Geração (sampling)]] — os parâmetros que os sliders mexem
- [[Avaliação SFT]] — o mesmo modelo, medido com números
- [[Modo didático]] — a outra página do mesmo servidor: o modelo por dentro
- [[Pingo]] — voltar ao índice
