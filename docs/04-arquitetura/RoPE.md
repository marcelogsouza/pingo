---
tags: [pingo, arquitetura, rope, posicao]
etapa: 5
atualizado: 2026-10-30
---

# RoPE (Rotary Position Embedding)

```python
class RoPE(nn.Module):
    def inv_freq(self, device):                      # θ^(-2i/d), calculado sob demanda
    def tables(self, position_ids):                  # pos · inv_freq → (cos, sin) [B,T,D]
```

## A ideia, em uma frase

Em vez de **somar** um vetor de posição ao embedding (posicional absoluto) ou **mascarar** com
um viés (relativo tipo T5), RoPE **rota** os vetores de Query e Key por um ângulo proporcional à
posição. O produto interno $q_m \cdot k_n$ passa a depender só de $m-n$.

## A matemática

Para cada par de dimensões $(2i, 2i+1)$ do head, a frequência angular é

$$
\theta_i = \Theta^{-2i/d}, \qquad \Theta = 10{,}000,\;\; i = 0,\dots,\tfrac d2-1
$$

e a posição $m$ gira o par por $\phi_i(m) = m\,\theta_i$. Em notação complexa (com o par
como $z = x_{2i} + i\,x_{2i+1}$):

$$
\tilde z = z \cdot e^{\,i\,m\theta_i}
$$

Na implementação real (real-valued), com
$c_i = \cos(m\theta_i)$ e $s_i = \sin(m\theta_i)$:

$$
\begin{pmatrix} \tilde x_{2i} \\ \tilde x_{2i+1} \end{pmatrix} =
\begin{pmatrix} c_i & -s_i \\ s_i & c_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

que é exatamente o que `rotate_half` + `cos/sin` fazem em bloco:

```python
rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])
out = x * cos + rotate_half(x) * sin
```

`tables()` monta `emb = cat(freqs, freqs)` para que `cos/sin` tenham a mesma forma de `x`
(`[B,T,D]`, broadcast sobre os heads via `unsqueeze(1)`).

## Por que isso dá posição *relativa* (a propriedade central)

Seja $R_m$ a matriz de rotação da posição $m$. Como rotações se compõem
($R_m R_n = R_{m+n}$) e $R_m^{\top} = R_{-m}$:

$$
q_m^{\top} k_n = \big(R_m q\big)^{\top}\big(R_n k\big)
= q^{\top} R_m^{\top} R_n\, k = q^{\top} R_{n-m}\, k
$$

Ou seja: o *score* de atenção é função **apenas da distância** $n-m$. Não existe nada
"absoluto" para o modelo decorar, e o mesmo padrão aprendido em distância 3 vale em qualquer
posição da sequência.

**É testado**: `test_rope_invariancia_relativa` verifica numericamente
$\langle q_m,k_n\rangle = \langle q_{m+\Delta}, k_{n+\Delta}\rangle$ na config tiny.

## Detalhes de implementação (e os bugs que eles evitam)

### 1. `inv_freq` calculado **sob demanda**, sem buffer

```python
def inv_freq(self, device):
    if device not in self._inv_freq_cache:
        e = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim
        self._inv_freq_cache[device] = self.theta ** (-e)
    return self._inv_freq_cache[device]
```

> [!bug] O bug caro do HF v5
> A implementação "óbvia" registra `inv_freq` como buffer (`register_buffer(..., persistent=False)`).
> O `from_pretrained` do **transformers v5** constrói o modelo sob `meta device` e materializa
> buffers non-persistent **sem inicializar** — o `inv_freq` voltava lixo, e os logits ficavam
> errados **mesmo com pesos bit-exatos**. A solução: recriar a tabela on-the-fly (custa um
> `arange` + `pow`, desprezível) e guardar num dict por device.

### 2. Nenhum parâmetro

RoPE não tem pesos → o estado de `self.rope` não aparece no `state_dict`, e não há risco de
salvar/carregar tabela desatualizada.

### 3. Aplicado só em Q e K — nunca em V

Rotacionar $V$ não muda nada (a atenção é linear em $V$ e a rotação se cancelaria na média
ponderada de forma errada). A posição afeta **onde olhar** (scores), não **o que é lido**.

### 4. `position_ids` vêm de fora quando há cache

Sem cache: `0..T-1`. Com `PingoCache`: `cache.cur_len .. cur_len+T-1`, para que o token novo
seja rotacionado na posição **global** dele ([[KV cache]]).

## Escolhas de hiperparâmetro

| Escolha | Valor | Implicação |
|---|---|---|
| $d_{head}$ | 64 | pares de dimensões: 32 frequências distintas |
| $\Theta$ | 10.000 | o "clássico" (Llama-3 usa 500k para contexto enorme) |
| comprimento de onda mais longo | $2\pi\Theta^{0} = 2\pi$ para $i=0$ … e $\theta_{31} = 10^{-62/32}$ | a dimensão mais lenta dá ~$2\pi\Theta \approx 62.832$ posições antes de repetir |

$$
\text{período da dimensão } i = \frac{2\pi}{\theta_i} = 2\pi\,\Theta^{2i/d}
$$

Com contexto de 2048, nada chega perto do período máximo — não há "aliasing" posicional.

> [!note] Contexto maior no futuro
> RoPE permite *extensão* (ex.: NTK-aware / YaRN, mudando $\Theta$ na inferência) sem
> retreinar do zero. Foi parte da razão de escolhê-lo sobre embedding aprendido.

## Ver também

- [[Atenção causal]] — onde as tabelas são aplicadas
- [[KV cache]] — de onde vêm os `position_ids` incrementais
- [[Gotchas]] — o bug do buffer non-persistent
- [[Pingo]] — voltar ao índice
