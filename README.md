# ALiBi, GQA
## 1. ALiBi(Attention with Linear Biases)
ALiBi는 상대적 위치 인코딩 방식 중 하나로 아래의 그림처럼 쿼리와 키 벡터를 곱한 어텐션 스코어 행렬에 오른쪽에서 왼쪽으로 갈수록 더 작은 값을 더하는 방식을 사용합니다. 

<div align="center">
  <img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/de2f0ffb-8755-444a-93a7-3c005108e012" />
</div>

이는 어텐션 스코어에 단순한 편향(bias)을 더하는 방식이며, 수식으로 나타낼 경우 다음과 같습니다.

$$
a_i = \text{softmax}\\left( \mathbf{q}_i \mathbf{K}^\top + m \cdot [- (i-1), \ldots, -1, 0] \right) \\
= \text{softmax}\\left( \mathbf{q}_i \mathbf{K}^\top + m \cdot [0, 1, \ldots, (i-1)] \right)
$$

여기서 $\mathbf{q}_i \mathbf{K}^\top$는 일반적인 어텐션 스코어, $m$은 각 head의 기울기(slope)로 미리 정해지는 값입니다. 그리고 [...]는 토큰 간의 상대적 거리를 의미합니다. 위의 식은 $i$ 번째 토큰에 대한 식입니다.

head-specifi slope $m$ 값은 어텐션 헤드(head)의 개수를 이용합니다. 

논문에서는 8개의 head를 사용하며, 각 head의 slope을 아래와 같이 $\frac{1}{2^{\tfrac{8}{n}}}$을 공비로 하는 등비급수로 설정합니다. 가장 이상적인 경우는 헤드의 수가 8, 16, 32처럼 2의 거듭제곱일 때입니다. 
<div align="center">
  <img width="100" height="50" alt="image" src="https://github.com/user-attachments/assets/74674df7-c34a-41e3-b3d7-92d7761df22f" />
</div>

먼저, 다음과 같이 어텐션 헤드의 개수(num_heads)에 가장 가까운 2의 거듭제곱을 구합니다. 
```
import math, torch

n = 2 ** math.floor(math.log2(num_heads))
```

아래의 m_0은 공비에 해당합니다. 파이토치의 pow()와 m_0을 이용해 등비수열 m을 생성할 수 있습니다. 
```
m_0 = 2.0 ** (-8.0 / n) # 2^{-8/n} 
m = torch.pow(m_0, torch.arange(1, 1+n))
```

이렇게 m은 num_heads에 따라 미리 정해지는 값입니다. 

다음으로, m과 곱해지는 상대 위치 행렬은 쿼리 토큰의 수와 키 토큰의 수를 이용하여 다음과 같으 구혆할 수 있습니다.
```
x = torch.arange(seq_length, device=device)[None, :]
y = torch.arange(seq_length, device=device)[:, None]

x-y
```
- alibi는 인코더/디코더 셀프 어텐션에서만 사용할 것이기 때문에 쿼리와 키의 시퀀스 길이는 동일합니다.

이 상대 위치 행렬의 원소는 음수와 양수의 값을 가지며 음수는 과거 시점, 양수는 미래 시점을 나타냅니다. 
- 디코더 셀프 어텐션에서 causal mask를 사용하기 때문에 미래 시점의 원소는 무시됩니다. 

어텐션 스코어 행렬(attn_scores)이 있다는 가정 하에, 위치 정보는 다음과 같이 주입됩니다.
```
attn_scores = attn_scores + m*(x-y)
```
- m*(x-y)는 파이토치 기준 requires_grad=False이므로 역전파를 통해 값이 업데이트 되지 않습니다. 

논문에 따르면, ALiBi는 아래의 그림처럼 학습 중 보지 못했던 긴 시퀀스에 대해서도 다른 인코딩 방식(사인파, RoPE, T5 Bias)과 비교했을 때,성능(PPL)이 거의 떨어지지 않습니다. 

<div align="center">
  <img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/b825ac2f-1a35-417f-8cc5-4ba112f93957" />
</div>

<div align="center">
<img width="517" height="171" alt="image" src="https://github.com/user-attachments/assets/d7b7debf-16ce-444d-ab9e-4737c29180fd" />
</div>

## 2. GQA(Grouped-query-Attention)



