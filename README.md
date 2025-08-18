# ALiBi, GQA(Grouped-query-Attention)
## 1. ALiBi

$$
a_i = \text{softmax}\\left( \mathbf{q}_i \mathbf{K}^\top + m \cdot [- (i-1), \ldots, -1, 0] \right) \\
= \text{softmax}\\left( \mathbf{q}_i \mathbf{K}^\top + m \cdot [0, 1, \ldots, (i-1)] \right)
$$

$ \mathbf{q}_i \mathbf{K}^\top $
