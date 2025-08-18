# ALiBi, GQA(Grouped-query-Attention)
## 1. ALiBi
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

ALiBi 구현에 있어 중요한 건 head-specifi slope $m$ 값을 어떻게 정하느냐입니다. 나머지 부분은 단순히 구현 문제입니다. 

논문에서는 8개의 head를 사용하며, 각 head의 slope을 아래와 같은 등비급수로 설정합니다. 
<div align="center">
  <img width="100" height="50" alt="image" src="https://github.com/user-attachments/assets/74674df7-c34a-41e3-b3d7-92d7761df22f" />
</div>

