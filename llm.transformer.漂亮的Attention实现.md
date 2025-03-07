---
id: kcism5sjj3bvuntcovct65l
title: 漂亮的Attention实现
desc: ''
updated: 1741319555926
created: 1741315704596
---

公式参考 Attention is all you need。Q, K 和 V 是 X 经过权重矩阵处理后得来的。当使用多头时，那么权重矩阵中，out_feature 通常是 dim // num_heads，少于输入的 input_dim。实现时，通常使用一个 nn.Linear 表示这些多头的 Q 的权重矩阵，是合并起来了的权重矩阵，方便一并计算。所以输出的 Q 中，Q[..., i:(i+1)*dim] 对应每个头。注意力和多头注意力公式如下：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

这里只是展现了注意力机制，有了 Q, K 和 V 后，Attention() 计算的细节。传给 Attention 的参数 Q, K 和 V，需要输入 X 经过对应的权重矩阵处理才能得到，具体流程参考多头矩阵部分。

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

nn.Linear() 中的 weight 的形状是 (out_features, in_features)，$W_i^Q$ 对应 weight[i:(i+1)*dim, :]。为方便计算多头，输入和输出的最后一维看作由两维num_heads x head_dim 组成，准备处理注意力前拆分。转置后，执行一系列注意力操作，最后再转置回来，并 reshape，便可省去 Concat 操作。

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

```py
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            # 3 分别由 Q K V 组成，所以转置后方便 unbind
            # 最后一维由 dim 拆为 num_heads x head_dim，方便多头操作
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            # 转置后，方便使用多头操作，把 N 与 head_dim ,维度放到最后两维即可。
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            # (B num_heads N head_dim) * (B num_heads head_dim N)
            attn_scores = torch.matmul(q, k.transpose(-2, -1))
            # Add attention mask if provided
            if attn_mask is not None:
                attn_scores += attn_mask
            # Apply softmax to get attention weights (softmax is applied along the last dimension)
            attn_weights = F.softmax(attn_scores, dim=-1)
            # Dropout on attention weights (if dropout is used)
            attn_weights = self.attn_drop(attn_weights)
            # Apply attention weights to value tensor (V)
            # (B num_heads N N) * (B num_heads N head_dim)
            x = torch.matmul(attn_weights, v)
        # 把最后两维连接起来，num_heads 个 head_dim 拼接
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

## Ref and Tag

#Transformer