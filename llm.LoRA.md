---
id: 3d70v3qtqyg2tjtnwuzus6g
title: LoRA
desc: ''
updated: 1751732696911
created: 1751690669800
---

## LoRA：`h_final = W * x + (alpha / r) * B * A * x`

LoRA (Low-Rank Adaptation) 是一种高效的微调（Fine-Tuning）技术，其核心思想是：**在对大型预训练模型进行微调时，我们不需要更新模型的所有参数，而只需要更新一个非常小的、“注入”到模型中的“旁路”参数。**

### LoRA 的核心思想与动机

1.  **问题背景**: 对一个像 GPT-3 (1750亿参数) 这样的模型进行全量微调（Full Fine-Tuning）成本极高：
    *   **训练成本高**: 需要巨大的 GPU 显存（VRAM）来存储模型参数、梯度和优化器状态。
    *   **存储成本高**: 每个微调任务都会产生一个完整的模型副本，如果微调 100 个不同任务，就需要存储 100 个巨大的模型文件。
    *   **部署/切换困难**: 在生产环境中，为不同任务加载不同的完整模型非常耗时且不灵活。

2.  **核心假设 (Low-Rank Hypothesis)**: 微软的研究人员发现，预训练好的大模型本身已经非常强大，微调所带来的参数变化其实是一个“低秩”的（Low-Rank）。换句话说，尽管整个权重矩阵 $W$ 是巨大的（例如 $4096 \times 4096$），但微调时它需要更新的“变化量” $ΔW$ 可以被分解为两个更小的矩阵的乘积。

    在数学上，一个大矩阵 $ΔW$ (维度 $d \times k$) 如果是低秩的，它可以被近似为 $B \times A$，其中 $B$ 的维度是 $d \times r$，$A$ 的维度是 $r \times k$。这里的 $r$ 就是“秩”（rank），并且 $r$ 远小于 $d$ 和 $k$ ($r << min(d, k)$)。

3.  **LoRA 的解决方案**:
    *   **冻结原始权重**: 在微调时，保持预训练模型的原始权重 $W$ 不变（frozen）。
    *   **注入低秩矩阵**: 在原始权重旁边，并联一个由两个小矩阵 $A$ 和 $B$ 组成的“旁路”。
    *   **只训练旁路**: 训练过程中，只更新矩阵 $A$ 和 $B$ 的参数。

    因此，模型的总更新量从巨大的 $W$ 变成了小得多的 $A$ 和 $B$。例如，如果 $W$ 是 $4096 \times 4096$（约1677万参数），我们选择 $r=8$，那么 $A$ 是 $8 \times 4096$，$B$ 是 $4096 \times 8$，总共需要训练的参数只有 $(4096 \times 8) + (8 \times 4096) ≈ 6.5万$，参数量减少了几个数量级。

---

### LoRA 是如何加载到 Transformer 块的？

这是问题的关键。LoRA 并非替换 Transformer 中的任何层，而是“附加”或“注入”到现有的权重矩阵上，通常是 `nn.Linear` 层。在 Transformer 块中，最常被注入 LoRA 的是**自注意力（Self-Attention）**模块中的四个关键线性层：

*   **Query (Wq)**
*   **Key (Wk)**
*   **Value (Wv)**
*   **Output (Wo)**

有时也会应用到前馈网络（FFN）的线性层中。

我们以一个标准的线性层（例如 Query）为例，来看 LoRA 的加载和计算过程。

**原始的计算方式:**
一个输入向量 `x` 经过线性层，计算方式为：
`h = W * x`
其中 `W` 是预训练的权重矩阵，`h` 是输出。

**LoRA 加载后的计算方式:**
LoRA 在旁边增加了一个“旁路”，计算流程变为：

1.  **冻结原始路径**: `h_orig = W * x` (这里的 `W` 不参与梯度更新)
2.  **计算旁路路径**:
    *   输入 `x` 先经过矩阵 `A`：`x_a = A * x`
    *   然后经过矩阵 `B`：`h_lora = B * x_a`
    *   所以旁路输出是：`h_lora = B * A * x`
3.  **合并输出**: 将原始路径的输出和旁路输出相加。
    `h_final = h_orig + h_lora = (W * x) + (B * A * x)`

为了方便，可以写成：
`h_final = (W + B * A) * x`

这里还引入了一个**缩放因子 `alpha`**，通常与 `rank (r)` 配合使用，公式变为：
`h_final = W * x + (alpha / r) * B * A * x`
这个缩放可以稳定训练，作用类似于一个学习率。

**图解加载过程:**

下面是一个简化的示意图，展示了 LoRA 如何“附加”到一个 `nn.Linear` 层上：

```
      +--------------------------------+
      |        Input Vector x          |
      +--------------------------------+
                  |
        +---------+---------+
        |                   |
        | (Frozen)          | (Trainable)
+-------------------+   +-------------------+
|  Original Linear  |   |    LoRA Adapter   |
|     Layer (W)     |   | +-------+ +-------+|
+-------------------+   | |  A (r,k)  | |  B (d,r)  ||
        |               | +-------+ +-------+|
        |               +-------------------+
        |                       |
   h_orig = W*x             h_lora = (B*A)*x
        |                       |
        +-----------+-----------+
                    |
                    + (Addition)
                    |
      +--------------------------------+
      |       Final Output h_final     |
      +--------------------------------+
```

### 训练与推理的区别

LoRA 的一个巨大优势在于它在训练和推理时的不同表现。

*   **训练时 (Training)**:
    *   如上图所示，模型包含原始权重和 LoRA 旁路。
    *   内存中需要加载 `W` (虽然是冻结的)、`A` 和 `B`。
    *   只有 `A` 和 `B` 的梯度和优化器状态需要存储，大大减少了 VRAM 占用。

*   **推理时 (Inference)**:
    *   为了消除任何额外的计算延迟，我们可以将 LoRA 适配器“合并”回原始权重中。
    *   我们计算一个新的权重矩阵 `W_new = W + (alpha / r) * B * A`。
    *   然后，我们可以丢弃 `A` 和 `B`，只使用这个新的、尺寸与原始 `W` 完全相同的 `W_new`。
    *   **结果**: 推理时的模型结构与原始模型完全一样，没有任何额外的层或计算，因此**没有推理延迟**。

### 总结 LoRA 的优点

1.  **参数高效**: 只需训练和存储极少量的参数（LoRA 权重 `A` 和 `B`）。
2.  **内存高效**: 训练时显著降低 GPU 显存需求。
3.  **无推理延迟**: 合并后的模型与原模型结构相同，速度一样快。
4.  **任务切换灵活**: 可以为多个任务训练不同的 LoRA 适配器。在部署时，只需加载基础模型，然后根据需要快速加载或切换不同任务的 LoRA 权重（几十MB），而无需加载整个几十GB的模型。这被称为“Adapter-based tuning”。

总而言之，LoRA 通过在 Transformer 的线性层旁边增加一个可训练的低秩旁路，实现了在冻结绝大部分模型参数的情况下进行高效微调，完美地平衡了性能、效率和灵活性。

### 变种

[rank-stabilized LoRA](https://arxiv.org/pdf/2312.03732) 提出，缩放因子部分使用 $\frac{\alpha}{\sqrt{r}}$ 而不是 $\frac{\alpha}{r}$，能提高稳定性，通常 r 选择 8 和 16。

## Ref and Tag