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

여기서 $\mathbf{q}_i \mathbf{K}^\top$는 일반적인 어텐션 스코어, $m$은 각 head의 기울기(slope)로 미리 정해지는 값이며, 각 head는 서로 다른 slope를 가집니다. 그리고 [...]는 토큰 간의 상대적 거리를 의미합니다. 위의 식은 $i$ 번째 토큰에 대한 식입니다.

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

다음으로, m과 곱해지는 상대 위치 행렬은 쿼리 토큰의 수와 키 토큰의 수를 이용하여 다음과 같으 구현할 수 있습니다.
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
- 정리하면 m*(x-y)는 각 헤드마다 기울기(m)를 가진 거리 기반 편향으로서, 파이토치 기준 requires_grad=False이므로 역전파를 통해 값이 업데이트 되지 않습니다. 

ALiBi는 아래의 그림처럼 학습 중 보지 못했던 긴 시퀀스에 대해서도 다른 인코딩 방식(사인파, RoPE, T5 Bias)과 비교했을 때,성능(PPL)이 거의 떨어지지 않습니다. 

<div align="center">
  <img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/b825ac2f-1a35-417f-8cc5-4ba112f93957" />
</div>

또한, 간단한 인코딩 방식을 사용하기 때문에 학습과 추론 처리 시간(초당 처리 토큰(또는 단어) 수)도 매우 효율적입니다. 

<div align="center">
<img width="500" height="200" alt="image" src="https://github.com/user-attachments/assets/d7b7debf-16ce-444d-ab9e-4737c29180fd" />
</div>

- RoPE와 T5의 상대 위치 편향은 사인파나 ALiBi처럼 계산으로 처리하는 게 아니라, 추가적인 연산(예: RoPE의 경우 토큰 임베딩을 회전하는 처리, T5의 상대적 위치 편향은 룩업 테이블, 버킷 등)이 필요하지만,
- 사인파 방식과 ALiBi는 정해진 값을 단순 추가하는 방식이기 때문에 속도가 더 빠를 수밖에 없습니다. 사인파와 ALiBi의 속도가 거의 동일한 것을 볼 수 있습니다. 

## 2. GQA(Grouped-query-Attention)
ALiBi의 핵심은 'Train Short, Test Long' 을 가능하게 한다는 점입니다. 그리고 간단한 덧셈만으로 위치 정보를 처리하기 때문에 위에서 본 것처럼, 다른 방식(예: RoPE, T5 Bias)에 비해 더 빠른 추론 속도, 그리고 좋은 성능을 기대할 수 있습니다. 

그러나, 트랜스포머 모델 추론 속도의 병목이 발생하는 주된 원인은 위치 정보를 처리하는 방식이 아닌 메모리 대역폭과 순차적인 연산때문입니다. 

특히, 매 디코딩 스텝마다 파라미터와 KV-Cache에서 키와 벨류 벡터를 로드해야 하기 때문에 추론 속도가 저하됩니다. 
- 참고로 GPU를 사용한다고 했을 때, GPU 메모리에 올라가는 주요 데이터는 모델 파리미터와 KV-Cache이기 때문에, MHA의 경우 어텐션 헤드의 개수가 많아질수록 GPU 메모리를 더 많이 차지하게 됩니다. 

이 한계를 해결하고자 하나의 K/V만 사용하는 MQA가 제안되었으나, MQA는 하나의 K/V만 사용하기 때문에 MHA에 비해 성능 저하, 불안정한 학습이라는 문제가 있었으며, 이 MQA의 한계를 보완하고자 등장한 것이 GQA입니다.

<div align="center">
  <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/cb6f819f-e77e-4029-a637-ec5820dac2ab" />
</div>

GQA는 간단히 말하면, MQA와 MHA를 절충한 방법입니다. MHA가 H개의 헤드, MQA가 1개의 헤드를 사용했다면 GQA는 G개(1 < G < H)의 헤드를 사용하기 때문입니다. 

GQA는 위의 그림처럼 여러 개의 쿼리 헤드가 키/벨류 헤드를 그룹으로 공유하는 방식입니다. 이를 통해 MHA의 성능을 거의 유지하면서도 MQA처럼 KV Cache 크기를 줄일 수 있기 때문에, MHA에 비해 추론 과정에서 key/value 벡터를 로드하는 시간을 크게 줄일 수 있습니다. 

GQA를 선택한 이유는 성능 부분도 있었지만, ALiBi와 GQA의 구현이 서로 독립적이었기 때문입니다.
- ALiBi는 어텐션 스코어 계산 후에 적용되고 GQA는 어텐션 스코어를 계산하기 전, K/V 프로젝션 단계에서 구조를 변경합니다.

또한, 다음과 같이 GQA를 통해 계산한 어텐션 스코어에 간단히 차원만 변경하는 방법으로 ALiBi를 적용할 수 있기 때문입니다.
```
## attention scores
Q = Q / math.sqrt(self.query_head_dim)
gqa_scores = Q @ K.transpose(3, 4)
# gqa_scores.shape: [batch_size, num_groups, query_heads_per_group, query_length, kv_length]

alibi_bias = (self.m * get_relative_positions(q_len, query.device)).unsqueeze(0)
# alibi_bias.shape: [1, num_query_heads, query_length, query_length]

alibi_bias = alibi_bias.view(1, num_groups, query_heads_per_group, q_len, q_len)
# alibi_bias.shape: [1, num_groups, query_heads_per_group, query_length, query_length]

gqa_scores = gqa_scores + alibi_bias
```
- ALiBi는 인코더/디코더 셀프 어텐션에서만 사용하므로 query_length == kv_length입니다.
- 구체적으로, 그룹화된 헤드 구조에 맞게 alibi_bias도 num_query_heads차원을 그룹 수(num_groups)와 그룹당 쿼리 헤드 개수(uery_heads_per_group) 두 개의 차원으로 나누면 됩니다.
- 이렇게 하면 첫 번째 쿼리 헤드에는 첫 번째 alibi_bias가 N번째 쿼리 헤드에는 N번째 alibi_bias가 정확히 매핑됩니다.
- 즉, ALiBi의 헤드별 고유 bias라는 원칙을 GQA 환경에서도 그대로 유지시킬 수 있습니다. 

인코더의 셀프 어텐션은 병목 구간이 아니기 때문에 MHA를 사용해도 무방합니다.

그래서서 인코더의 셀프 어텐션에는 MHA + ALiBi를, 디코더의 셀프 어텐션에는 GQA + ALiBi, 그리고 디코더의 크로스 어텐션에는 GQA만 적용했으며, 이는 <code>attention.py</code>와 <code>transformer.py</code>에서 확인할 수 있습니다. 
