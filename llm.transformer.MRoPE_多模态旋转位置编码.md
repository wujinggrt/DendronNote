---
id: 79m0yaj6thit46x22078d01
title: MRoPE_多模态旋转位置编码
desc: ''
updated: 1746723516931
created: 1746636769551
---

文本 LLM 使用 RoPE-1D。而 MLLM 需要 RoPE-2D。文本是 1D 序列，所以它的位置只是一个标量 n；图像是2D的（“宽”和“高”），所以表达它的位置需要一个二维向量(x,y) ；视频则在图像的基础上新增了一个时间维度（或者说“帧”），所以它的位置是一个三维向量(x,y,z) 。当我们希望用同一个模型去处理三种模态的数据时，就要想办法糅合这三种不同形式的位置信息。

有的工作，选择展开所有模态，使用 RoPE-1D，但是可能降低性能。

需要设计向后兼容的 MRoPE，既能够满足多模态要求，也能在单一文本模态时有 RoPE-1D 的能力。也就是 RoPE-3D 能够退化到 RoPE-2D/1D 的功能。

以图文混合模态为例，图像是 2D 模态，文本是 1D 模态，需要将文本升维到 RoPE-2D 后统一使用它。但是，需要考虑仅有文本输入时，与 RoPE-1D 完全等价。对比两者：

$$
\text{RoPE-1D} \quad (\mathcal{R}_n) = 
\left(
\begin{array}{ccccccccc}
\cos n\theta_0 & -\sin n\theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\
\sin n\theta_0 & \cos n\theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\
0 & 0 & \cos n\theta_1 & -\sin n\theta_1 & \cdots & 0 & 0 & 0 & 0 \\
0 & 0 & \sin n\theta_1 & \cos n\theta_1 & \cdots & 0 & 0 & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos n\theta_{d/2-2} & -\sin n\theta_{d/2-2} & 0 & 0 \\
0 & 0 & 0 & 0 & \cdots & \sin n\theta_{d/2-2} & \cos n\theta_{d/2-2} & 0 & 0 \\
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \cos n\theta_{d/2-1} & -\sin n\theta_{d/2-1} \\
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \sin n\theta_{d/2-1} & \cos n\theta_{d/2-1} \\
\end{array}
\right)
$$

$$
\text{RoPE-2D} \quad (\mathcal{R}_{x,y}) = 
\left(
\begin{array}{ccccccccc}
\cos x\theta_0 & -\sin x\theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\
\sin x\theta_0 & \cos x\theta_0 & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\
0 & 0 & \cos y\theta_1 & -\sin y\theta_1 & \cdots & 0 & 0 & 0 & 0 \\
0 & 0 & \sin y\theta_1 & \cos y\theta_1 & \cdots & 0 & 0 & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos x\theta_{d/2-2} & -\sin x\theta_{d/2-2} & 0 & 0 \\
0 & 0 & 0 & 0 & \cdots & \sin x\theta_{d/2-2} & \cos x\theta_{d/2-2} & 0 & 0 \\
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \cos y\theta_{d/2-1} & -\sin y\theta_{d/2-1} \\
0 & 0 & 0 & 0 & \cdots & 0 & 0 & \sin y\theta_{d/2-1} & \cos y\theta_{d/2-1} \\
\end{array}
\right)
$$

可以发现，$\mathcal{R}_{n,n}=\mathcal{R}_n$，即位置 n 的 RoPE-1D 与位置 (n,n) 的 RoPE-2D 相等。那么，只需要对纯文本的位置坐标取 (n,n) 即可。

但是，RoPE-1D 中，$\theta_i = b^{-2i/d}$，有 $\theta_{2j}$ 不同于 $\theta_{2j+1}$。但是 RoPE-2D 中，为了确保 x 和 y 的对称性，有 $\theta_{2j} = \theta_{2j+1}$，于是产生矛盾。

两种选择：一是放弃 x 和 y 的对称性，RoPE-2D 依旧取 $\theta_i = b^{-2i/d}$，自然地 $\theta_{2j} \neq \theta_{2j+1}$；二是将 RoPE-2D 取 $\theta_{2j} = \theta_{2j+1} = b^{-4j/d}$，此时纯文本的位置编码与 RoPE-1D **略有不同**。

两种方案各有优劣。

考察 Qwen2VL 的实现。ViT 部分用到了 RoPE-2D，在 LLM 部分用到了 RoPE-3D。

```py
class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
```

上面可以得到 freqs，shape 为 (seqlen, dim/2)，即 seq 与每个 inv_freq 元素相乘。freqs[seq] 对应 $\boldsymbol{R}_{\Theta,seq}^{dim}$。

$$
R_{\Theta, ids_x, ids_y}^d x = 
\left( \begin{array}{c}
x_1 \\
x_2 \\
\vdots \\
x_{d/2-1} \\
x_{d/2} \\
x_{d/2+1} \\
x_{d/2+2} \\
\vdots \\
x_{d-1} \\
x_d
\end{array} \right)
\otimes
\left( \begin{array}{c}
\cos ids_x \theta_1 \\
\cos ids_y \theta_1 \\
\vdots \\
\cos ids_x \theta_{d/4} \\
\cos ids_y \theta_{d/4} \\
\cos ids_x \theta_1 \\
\cos ids_y \theta_1 \\
\vdots \\
\cos ids_x \theta_{d/4} \\
\cos ids_y \theta_{d/4}
\end{array} \right)
+
\left( \begin{array}{c}
-x_{d/2+1} \\
-x_{d/2+2} \\
\vdots \\
-x_{d/2-1} \\
-x_{d/2} \\
x_1 \\
x_2 \\
\vdots \\
x_{d/2-1} \\
x_{d/2}
\end{array} \right)
\otimes
\left( \begin{array}{c}
\sin ids_x \theta_1 \\
\sin ids_y \theta_1 \\
\vdots \\
\sin ids_x \theta_{d/4} \\
\sin ids_y \theta_{d/4} \\
\sin ids_x \theta_1 \\
\sin ids_y \theta_1 \\
\vdots \\
\sin ids_x \theta_{d/4} \\
\sin ids_y \theta_{d/4}
\end{array} \right)
$$

```py
class Qwen2VLVForConditionalGeneration(...):
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        """
```

参数 image_grid_thw 后的 thw 分别代表 temporal height width。

可以看到，MRoPE-3D 针对纯文本、图像和视频输入。使用了三个维度的位置，temporal position_ids, height position_ids, width position_ids。当内容时文本时，即 input_ids 为 [T T T T T] 时，位置上三者相同。都是 [0, 1, 2, 3, 4]。当处理视频时，需要考虑时间、高和宽的位置。比如，视频输入为 3 个视频 patch，高和宽的 patches 分别为 2 个，于是 3 个视频帧得到了 3 * 2 * 2 = 12 个 V token；有 5 个文本输入，于是 input_ids 为 [V V V V V V V V V V V V T T T T T]，V 代表视频，T 代表文本。最明显区分体现在间维度上，不同模态都会对齐，视频和文本的都有统一位置 ids，比如前三帧为 0 1 2，文本自然延续下去 3 4 5 6 7。但是在高度和宽度上，按照各自的维度模态处理。比如，前两个视频 token，三个编码分别是 [0, 0, 0]，第二个则是 [0, 0, 1]，以宽度为最后一维处理。文本则全部一样，比如 [3, 3, 3]。

```py
class Qwen2VLRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2VLConfig, device=None):
        ...
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
```

inv_freq 可以看做是各个 $\theta_i$ 的值。

```py
    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

## Ref and Tag