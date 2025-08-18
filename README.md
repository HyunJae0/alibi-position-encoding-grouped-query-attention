# ALiBi, GQA(Grouped-query-Attention)
## 1. ALiBi
ALiBi는 상대적 위치 인코딩 방식 중 하나로 아래의 그림처럼 쿼리와 키 벡터를 곱한 어텐션 스코어 행렬에 오른쪽에서 왼쪽으로 갈수록 더 작은 값을 더하는 방식을 사용합니다. 

<img width="582" height="250" alt="image" src="https://github.com/user-attachments/assets/aae1cd83-4888-4f54-84ec-e4a638d525ce" />

$$
a_i = \text{softmax}\\left( \mathbf{q}_i \mathbf{K}^\top + m \cdot [- (i-1), \ldots, -1, 0] \right) \\
= \text{softmax}\\left( \mathbf{q}_i \mathbf{K}^\top + m \cdot [0, 1, \ldots, (i-1)] \right)
$$

$\mathbf{q}_i \mathbf{K}^\top$
