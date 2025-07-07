---
id: uq048k5m1knzv5egjr43oxn
title: Openpi_代码
desc: ''
updated: 1751736836505
created: 1751475629280
---

## Configs: 训练配置

包含模型，数据，权重加载，训练超参数等。

## LoRA

参考 src/openpi/models/lora.py 部分。LoRA 嵌入了大部分网络层，方便微调。类 LoRAConfig 指定了配置，包含 rank-stabilized LoRA 选项。

为了高效，操作高度使用 einsum，并且适配投影矩阵来支持 einsum 和 LoRA。主要用于注意力机制部分。

```py
import flax.linen as nn

class Einsum(nn.Module):
    shape: tuple[int, ...]
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            # delta_W 维度记为 (...,d,k)，那么 w_a 维度为 (...,d,rank)，w_b 维度为 (...,rank,k)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            # 生成 LoRA x A B 的 einsum 表达式
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        return result
```

Einsum 可以看做将 x 投影到 qkv 的权重模块，上面适配了 LoRA 的部分。比如，gemma_fast 的 Attention 部分，当 kv 头与 q 头相等时，即 MHA：

```py
self.qkv_einsum = lora.Einsum(
    shape=(3, self.num_heads, self.features, self.head_dim),
    name="qkv_einsum",
    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
    lora_config=self.lora_config,
)
```

3 对应 qkv 向量，比如 shape 可以为 (3,8,32,4)，投影的 x 为 (B,S,32)，rank 为 2，w_a 维度 (3,8,32,2)，w_b 维度 (3,8,2,4)。对应数学表达 $\Delta W x = B A x$ 的矩阵。只是实现时，以转置的方式实现。所以 w_a 的倒数第二维与 x 维度相同，最后一维才是 rank 对应的轴，w_b 最后两维也是对应 B 的转置。但是实现中，w_a 对应矩阵 B，先与。这是因为张量的乘法中，以是 $x B A$ 形式进行的，可以看做转置的形式。投影时操作如下：

```py
q, k, v = self.qkv_einsum("BSD,3KDH->3BSKH", x)
```

## Ref and Tag

具身智能：基于pi0和pi0 fast的vla代码讲解（纯眼无实操版） - lumosity的文章 - 知乎
https://zhuanlan.zhihu.com/p/1895856498615240391