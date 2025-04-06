---
id: itv812r7qw03r8e5f3uygv3
title: LatexNote
desc: ''
updated: 1743957543235
created: 1739857979364
---

- 手写体： $\mathcal{M}_d$
- 双线条： $\mathbb{R}^{1369\times768}$
- 服从分布：$q \sim P(Q)$
- 复杂括号要有 \left 和 \right：$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

## 对齐


$$
\begin{align*}
K_r &= \frac{1}{WH} \sum_{w=1}^{W} \sum_{h=1}^{H} \left[ \left( \frac{\mathcal{M}_{o,I}(w,h) - \mathrm{mean}(\mathcal{M}_{o,I})}{\mathrm{std}(\mathcal{M}_{o,I})} \right)^4 \right], \\
K_a &= \max(\mathcal{M}_{o,I}) - \mathrm{mean}(\mathcal{M}_{o,I}).
\end{align*}
$$
