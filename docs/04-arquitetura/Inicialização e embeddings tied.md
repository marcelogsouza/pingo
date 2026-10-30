---
tags: [pingo, arquitetura, init, embeddings]
etapa: 5
atualizado: 2026-10-30
---

# Inicialização e embeddings tied

```python
def _init_weights(self, module):
    std = self.config.initializer_range           # 0.02
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if getattr(module, "_residual_proj", False):
            module.weight.data.mul_(1.0 / math.sqrt(2 * self.config.num_hidden_layers))
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
    elif isinstance(module, RMSNorm):
        module.weight.data.fill_(1.0)
```

## Regra 1 — `normal(0, 0.02)` em tudo que é matriz

A escolha "de família" (GPT-2 em diante): pesos $\mathcal{N}(0, 0{,}02^2)$, desvio **constante**,
independente do tamanho da camada. Isso funciona porque (a) a RMSNorm normaliza a escala na
entrada de cada bloco e (b) o `o_proj`/`down_proj` recebem a correção abaixo.

## Regra 2 — escala $1/\sqrt{2L}$ nas projeções residuais

```python
self.o_proj._residual_proj = True
self.down_proj._residual_proj = True
```

**O problema.** Cada bloco soma duas coisas no residual stream (saída da atenção e do MLP).
Se cada uma tem variância $v$, depois de $L$ blocos a variância do residual é

$$
\mathrm{Var}\big[x^{(L)}\big] \approx \mathrm{Var}[x^{(0)}] + 2L\,v .
$$

Com $L=12$ e $v$ constante, a ativação **cresce com a profundidade**, e o gradiente que volta
tem de atravessar essa escala — o treino fica sensível a lr e propenso a instabilidade.

**A correção.** Dividir por $\sqrt{2L}$ reduz a variância de cada soma por $2L$, deixando o
residual com variância aproximadamente constante ao longo da profundidade:

$$
\mathrm{Var}\big[x^{(\ell)}\big] \;\approx\; \mathrm{Var}\big[x^{(0)}\big] \quad \forall \ell .
$$

Com $L=12$: $\frac{1}{\sqrt{24}} \approx 0{,}204$. Ou seja, os pesos de `o_proj`/`down_proj`
nascem com std $\approx 0{,}0041$ — **5× menores** que os das outras matrizes.

> [!note] É testado
> `test_init_residuais_escalados` verifica exatamente isso: o desvio padrão dos pesos de
> `o_proj`/`down_proj` é $1/\sqrt{2L}$ vezes o das outras lineares da mesma shape.

> [!warning] Cuidado ao comparar com outros modelos
> Alguns modelos (Llama-3) põem a escala no *forward* (`/ sqrt(2L)`) em vez do init. O efeito
> no começo do treino é parecido, mas é diferente: a escala no init pode ser "desfeita" pelo
> otimizador, a do forward não. Aqui foi no init — como no GPT-2.

## Regra 3 — biased? Não.

Nenhum `nn.Linear` tem viés. Em pré-norm, o viés é redundante com o $\gamma$ do norm e com o
próprio residual; a prática moderna (Llama, PaLM, Mistral) o removeu. Economiza parâmetros e
uma operação de soma por linear.

## Embeddings tied (o ponto mais "de sistema" do modelo)

```python
self.embed   = nn.Embedding(vocab, d)
self.lm_head = nn.Linear(d, vocab, bias=False)
self.lm_head.weight = self.embed.weight        # MESMO tensor
```

| | sem tie | **com tie** |
|---|---|---|
| Params | 2 × 14.680.064 | **14.680.064** |
| Intuição | matriz "texto→espaço" e "espaço→texto" separadas | uma matriz: direção do token **é** a direção que o prevê |
| Efeito | mais capacidade | regularização + 11,5% menos params |

Matematicamente: o logit do token $v$ é $w_v^{\top} h$. Com tie, $w_v$ é o próprio embedding de
$v$ — o modelo aprende um espaço onde "apontar na direção de $v$" significa "o próximo token é
$v$".

### E o HF v5 exige declaração explícita

```python
_tied_weights_keys = {"lm_head.weight": "embed.weight"}
```

> [!bug] Sem essa linha, `save_pretrained` falha
> O v5 detecta tensores compartilhados e **recusa** salvar com *"shared tensors not properly
> defined"* a menos que a classe declare quais chaves são amarradas. É scaffolding do HF, não
> matemática — mas sem ela não há checkpoint em formato HF.

> [!bug] O `resize_token_embeddings` do SFT precisa re-amarrar
> Ao trocar `model.embed` por um `nn.Embedding` maior (16.384 → 16.387), o `lm_head.weight`
> **continua apontando para o tensor antigo**. `train_sft.resize_embeddings` refaz o tie
> explicitamente e atualiza `lm_head.out_features`. Sem isso, o modelo "treina" com dois
> vocabulários diferentes ([[Treino SFT]]).

## Onde `_init_weights` é chamado (e a pegadinha)

`post_init()` do HF chama `_init_weights` em **cada submódulo**. Por isso o init mora em
`_init_weights` (via flag), e não no `__init__` do bloco: se fosse no `__init__`, o
`post_init` sobrescreveria em seguida.

> [!bug] `torch.Tensor.ones_()` não existe
> Para o $\gamma$ do RMSNorm: `weight.data.fill_(1.0)`. Um `ones_()` acidental gera
> `AttributeError` no momento do `post_init`, ou seja, na construção do modelo.

## Ver também

- [[RMSNorm]] — o $\gamma$ inicializado em 1
- [[SwiGLU]] — quem carrega a flag `_residual_proj`
- [[Contagem de parâmetros]] — o efeito do tie na conta
- [[Gotchas]] — tied weights no HF v5
- [[Pingo]] — voltar ao índice
