---
tags: [pingo, arquitetura, mlp, swiglu]
etapa: 5
atualizado: 2026-10-30
---

# SwiGLU

```python
class SwiGLU(nn.Module):
    gate_proj: Linear(896, 2304, bias=False)
    up_proj:   Linear(896, 2304, bias=False)
    down_proj: Linear(2304, 896, bias=False)   # escala 1/√(2L) no init

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

## A fórmula

$$
\mathrm{SwiGLU}(x) = W_{down}\Big(\underbrace{\mathrm{SiLU}(W_{gate}x)}_{\text{portão}} \odot \underbrace{W_{up}x}_{\text{conteúdo}}\Big),
\qquad
\mathrm{SiLU}(z) = z\cdot\sigma(z)
$$

com $\sigma$ a sigmoide. A operação $\odot$ é produto elemento a elemento nas 2304 dimensões
internas.

## Por que "portão" ajuda

Um MLP comum é $W_2\,\phi(W_1x)$: a não linearidade é aplicada e o resultado é projetado. No
SwiGLU existe um **caminho multiplicativo**: o vetor `gate` decide, dimensão por dimensão,
*quanto* do vetor `up` passa. Isso dá à camada uma capacidade de "seleção condicional" que a
GeLU simples não tem — informalmente, é uma atenção consigo mesma por canal.

Empiricamente (Shazeer, 2020; e as ablations do PaLM e do Llama), SwiGLU bate GeLU com **o
mesmo orçamento de parâmetros**, medido em loss.

## Por que 2.304 (e por que o MLP ainda é ~3× maior que a atenção)

Um MLP de duas matrizes com fator $k$ custa $2dh$; SwiGLU com fator $h$ custa $3dh$. Para
empatar o custo:

$$
3 d h_{\text{swiglu}} = 2 d h_{\text{gelu}} \;\Longrightarrow\; h_{\text{swiglu}} = \tfrac{2}{3} h_{\text{gelu}}
$$

Como a prática de LLM usa $h_{\text{gelu}} \approx 4d$, sai

$$
h_{\text{swiglu}} \approx \tfrac23 \cdot 4d = \tfrac{8}{3}d = \tfrac{8}{3}\cdot 896 = 2389 \;\to\; \boxed{2304}
$$

(2304 = 18 × 128, múltiplo "redondo" para kernels de GEMM; o $8/3$ é o mesmo número que aparece
no `PLAN.md`.)

Conta de parâmetros por bloco:

$$
\underbrace{3 \cdot d \cdot h}_{\text{gate+up+down}} = 3 \cdot 896 \cdot 2304 = 6\,193\,152
\qquad\text{vs}\qquad
\underbrace{4 \cdot d^2}_{\text{q,k,v,o}} = 3\,211\,264
$$

Ou seja: **58% dos parâmetros do modelo estão nos MLPs** e 30% nas atenções
([[Contagem de parâmetros]]).

## Detalhe de implementação: `silu` e estabilidade

`F.silu` é $z\,\sigma(z)$, com $\sigma$ calculada internamente de forma estável (não há
overflow: para $z \to -\infty$, $\sigma\to 0$ e $\mathrm{SiLU}\to 0$; para $z\to+\infty$,
$\mathrm{SiLU}\approx z$). O produto com `up` é feito em bf16, como o resto do forward.

> [!note] `down_proj` é projeção "residual"
> Tanto `o_proj` quanto `down_proj` escrevem **de volta no residual stream**. É por isso que
> as duas recebem a escala $1/\sqrt{2L}$ no init — ver [[Inicialização e embeddings tied]].
> O atributo é só uma flag: `self.down_proj._residual_proj = True`.

## Ver também

- [[Arquitetura do Pingo]] — o bloco completo
- [[Contagem de parâmetros]] — a conta fechando
- [[Inicialização e embeddings tied]] — a flag `_residual_proj`
- [[Pingo]] — voltar ao índice
