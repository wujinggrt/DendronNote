---
id: 607ng4ib05dyd441uqnus6n
title: RoPE_旋转位置编码
desc: ''
updated: 1745945035448
created: 1745859217851
---

目的是为了引入相对位置信息。使得 $QK^T$ 时引入相对位置。

## 原 Transformer 的位置嵌入

暂时不考虑 batch。长度（seq_len）为 $N$ 的输入序列 $\mathbb{S}_N=\{\omega_i\}_{i=1}^N$，经过词嵌入层后，得到 $N$ 维向量 $\mathbb{E}_N=\{\boldsymbol{x}_i\}_{i=1}^N$，其中 $\boldsymbol{x}_i$ 为 $i$ 词的词嵌入向量，维度为 $d$。

处理自注意力之前，需要给词嵌入向量加入**位置信息**，随后计算 $\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{v}$：

$$
\begin{align*}
\boldsymbol{q}_m &= f_q(\boldsymbol{x}_m, m) \\
\boldsymbol{k}_n &= f_k(\boldsymbol{x}_n, n) \\
\boldsymbol{v}_n &= f_v(\boldsymbol{x}_n, n)
\end{align*}
$$

下标 m 和 n 分别表示位置，左式分别代表对应位置的 query，key 和 value 向量。

位置编码着重于构造合适的 $f(q,k,v)$ 函数形式。

观察自注意力和值的计算。默认每个向量是列向量，与 Transformer 论文的 $QK^T$ 表达不同，使用了 $Q^TK$ 来表达。通常，PyTorch 的张量中，向量用行向量来理解更方便。

$$
a_{m, n} = \frac{\exp\left(\frac{\boldsymbol{q}_m^{\mathbf{T}} \boldsymbol{k}_n}{\sqrt{d}}\right)}{\sum_{j=1}^{N} \exp\left(\frac{\boldsymbol{q}_m^{\mathbf{T}} \boldsymbol{k}_j}{\sqrt{d}}\right)} \\
\boldsymbol{o}_m = \sum_{n=1}^{N} a_{m, n} \boldsymbol{v}_n
$$

## 绝对位置编码：Sinusoidal Positional Embedding

Transformer 原论文将绝对位置编码信息直接添加到词嵌入向量，随后开始计算 query，key，value 向量。

$$
f_{t: t \in \{ q, k, v \}} (x_i, i) := \boldsymbol{W}_{t: t \in \{ q, k, v \}} (\boldsymbol{x}_i + \boldsymbol{p}_i)
$$

位置编码向量 $\boldsymbol{p}_i$ 计算使用了 sinusoidal 函数：

$$
\boldsymbol{p}_{i, 2t} = \sin \left( i / 10000^{2t/d} \right) \\
\boldsymbol{p}_{i, 2t+1} = \cos \left( i / 10000^{2t/d} \right)
$$

注意，$\boldsymbol{p}_i$ 的维数与词嵌入向量的维数相同，下标 $2t, 2t+1$ 分别代表 $d$ 维中的偶数维和奇数维的部分。

```py
# position 就对应 token 序列中的位置索引 i
# hidden_dim 就对应词嵌入维度大小 d
# seq_len 表示 token 序列长度
def get_position_angle_vec(position):
    return [position / np.power(10000, 2 * (hid_j // 2) / hidden_dim) for hid_j in range(hidden_dim)]

# position_angle_vecs.shape = [seq_len, hidden_dim]
position_angle_vecs = np.array([get_position_angle_vec(pos_i) for pos_i in range(seq_len)])

# 分别计算奇偶索引位置对应的 sin 和 cos 值
position_angle_vecs[:, 0::2] = np.sin(position_angle_vecs[:, 0::2])  # dim 2t
position_angle_vecs[:, 1::2] = np.cos(position_angle_vecs[:, 1::2])  # dim 2t+1

# positional_embeddings.shape = [1, seq_len, hidden_dim]
positional_embeddings = torch.FloatTensor(position_angle_vecs).unsqueeze(0)
```

## 二维旋转位置编码

观察到，计算自注意力分数时，softmax 函数需要求位置 m 的 query 向量与位置 n 的 key 向量的内积 $\boldsymbol{q}_m^{\mathbf{T}} \boldsymbol{k}_n$。于是，构造函数 $g$，满足如下条件：

$$
\langle f_q(\boldsymbol{x}_m, m), f_k(\boldsymbol{x}_n, n)\rangle = g(\boldsymbol{x}_m, \boldsymbol{x}_n, m - n)
$$

式子左侧，是 query 向量和 key 向量的内积，计算规则遵循 $f_q, f_k$，参数是词嵌入向量和绝对位置。

式子右侧的输入信息包含词嵌入向量和**相对位置信息** m - n。

理解关键抽象的概念：只要找到满足上式的 $f_q, f_k, g$ 函数，我们便找到了**引入相对位置信息到注意力的方法**，即 query 向量与 key 向量的内积蕴藏着词嵌入和相对位置信息，即函数 $g$ 的意义。改造计算 query 向量和 key 向量的方式为 $f_q, f_k$，即引入相对位置编码到注意力分数。

将情况细化到二维场景，即词嵌入向量维度为 $d=2$，苏剑林的论文提出了满足上述条件的式子如下：

$$
\begin{align*}
f_q(\boldsymbol{x}_m, m) &= (\boldsymbol{W}_q \boldsymbol{x}_m) e^{i m \theta} \\
f_k(\boldsymbol{x}_n, n) &= (\boldsymbol{W}_k \boldsymbol{x}_n) e^{i n \theta} \\
g(\boldsymbol{x}_m, \boldsymbol{x}_n, m - n) &= \operatorname{Re} \left[ (\boldsymbol{W}_q \boldsymbol{x}_m) (\boldsymbol{W}_k \boldsymbol{x}_n)^* e^{i (m - n) \theta} \right]
\end{align*}
$$

其中，Re 代表复数的实部。使用复数和欧拉公式等推导（省略），$f_q$ 可以表示为：

$$
\begin{align*}
f_q(\boldsymbol{x}_m, m) &= \begin{pmatrix} \cos m \theta & -\sin m \theta \\ \sin m \theta & \cos m \theta \end{pmatrix} \begin{pmatrix} W_q^{(1,1)} & W_q^{(1,2)} \\ W_q^{(2,1)} & W_q^{(2,2)} \end{pmatrix} \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \end{pmatrix} \\
&= \begin{pmatrix} \cos m \theta & -\sin m \theta \\ \sin m \theta & \cos m \theta \end{pmatrix} \begin{pmatrix} q_m^{(1)} \\ q_m^{(2)} \end{pmatrix}
\end{align*}
$$

矩阵 $\begin{pmatrix} \cos m \theta & -\sin m \theta \\ \sin m \theta & \cos m \theta \end{pmatrix}$ 的行列式为 1，是旋转矩阵。

$$
f_k(\boldsymbol{x}_m, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} W_k^{(1,1)} & W_k^{(1,2)} \\ W_k^{(2,1)} & W_k^{(2,2)} \end{pmatrix} \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \end{pmatrix} \\
= \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} k_m^{(1)} \\ k_m^{(2)} \end{pmatrix}
$$

最终 $g(\boldsymbol{x}_m, \boldsymbol{x}_n, m-n)$ 可以表示如下:

$$
g(\boldsymbol{x}_m, \boldsymbol{x}_n, m-n) = \begin{pmatrix} q_m^{(1)} & q_m^{(2)} \end{pmatrix} \begin{pmatrix} \cos((m-n)\theta) & -\sin((m-n)\theta) \\ \sin((m-n)\theta) & \cos((m-n)\theta) \end{pmatrix} \begin{pmatrix} k_n^{(1)} \\ k_n^{(2)} \end{pmatrix}
$$

理解复数 $e^{im\theta}$ 到旋转矩阵 $\begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}$ 的关系，比较复杂。但是旋转矩阵之间的等价比较容易推导。具体参考 [zhihu](https://zhuanlan.zhihu.com/p/642884818)。

## 扩展二维旋转矩阵到多维

对于任意维度的词嵌入向量，记为 d 维，则 $f_q, f_k$ 表示如下：

$$
f_{\{q,k\}}(\boldsymbol{x}_m, m) = \boldsymbol{R}_{\Theta,m}^d \boldsymbol{W}_{\{q,k\}} \boldsymbol{x}_m
$$

$$
\boldsymbol{R}_{\Theta,m}^d = \underbrace{
\begin{pmatrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1}
\end{pmatrix}}_{\boldsymbol{W}_m}
$$

$$
\Theta = \left\{\theta_i = 10000^{-2i/d}, i \in [0, 1, 2, \ldots, d/2 - 1]\right\}
$$

旋转矩阵可以提前计算出来。

使用 $f_{q,k}$ 计算 q 和 k 向量，在计算注意力时便可融入相对位置信息。具体如：

$$
\boldsymbol{q}_m^{\mathbf{T}} \boldsymbol{k}_n = \left( \boldsymbol{R}_{\Theta,m}^d \boldsymbol{W}_q \boldsymbol{x}_m \right)^{\mathbf{T}} \left( \boldsymbol{R}_{\Theta,n}^d \boldsymbol{W}_k \boldsymbol{x}_n \right) = \boldsymbol{x}_m^{\mathbf{T}} \boldsymbol{W}_q \boldsymbol{R}_{\Theta,n-m}^d \boldsymbol{W}_k \boldsymbol{x}_n \tag{14}
$$

其中，$\boldsymbol{R}_{\Theta,n-m}^d = \left( \boldsymbol{R}_{\Theta,m}^d \right)^{\mathbf{T}} \boldsymbol{R}_{\Theta,n}^d$。

值得指出的是，由于 $\boldsymbol{R}_{\Theta}^d$ 是一个正交矩阵，它不会改变向量的模长，因此通常来说它不会改变原模型的稳定性。

## 高效计算 RoPE

旋转矩阵 $\boldsymbol{R}_{\Theta,m}^d$ 是稀疏矩阵，通常使用如下方式高效计算：

$$
\boldsymbol{R}_{\Theta,m}^d \boldsymbol{x} = 
\begin{pmatrix}
x_0 \\
x_1 \\
x_2 \\
x_3 \\
\vdots \\
x_{d-2} \\
x_{d-1}
\end{pmatrix}
\otimes
\begin{pmatrix}
\cos m\theta_0 \\
\cos m\theta_0 \\
\cos m\theta_1 \\
\cos m\theta_1 \\
\vdots \\
\cos m\theta_{d/2-1} \\
\cos m\theta_{d/2-1}
\end{pmatrix}
+
\begin{pmatrix}
-x_1 \\
x_0 \\
-x_3 \\
x_2 \\
\vdots \\
-x_{d-1} \\
x_{d-2}
\end{pmatrix}
\otimes
\begin{pmatrix}
\sin m\theta_0 \\
\sin m\theta_0 \\
\sin m\theta_1 \\
\sin m\theta_1 \\
\vdots \\
\sin m\theta_{d/2-1} \\
\sin m\theta_{d/2-1}
\end{pmatrix}
\tag{15}
$$

$\otimes$ 代表逐位相乘，对应 PyTorch 的 `*` 操作，将两个张量的每个元素相乘，而非矩阵乘法。

## LLaMA 的实现

```py
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # freqs 包含了各个 θ
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.wq = Linear(...)
        self.wk = Linear(...)
        self.wv = Linear(...)

        self.freqs_cis = precompute_freqs_cis(dim, max_seq_len * 2)

    def forward(self, x: torch.Tensor):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, dim)
        xk = xk.view(batch_size, seq_len, dim)
        xv = xv.view(batch_size, seq_len, dim)

        # attention 操作之前，应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # scores.shape = (bs, seqlen, seqlen)
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(dim)
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, xv)  # (batch_size, seq_len, dim)

```

## Ref and Tag

十分钟读懂旋转编码（RoPE） - 绝密伏击的文章 - 知乎
https://zhuanlan.zhihu.com/p/647109286

一文看懂 LLaMA 中的旋转式位置编码（Rotary Position Embedding） - 梁德澎的文章 - 知乎
https://zhuanlan.zhihu.com/p/642884818