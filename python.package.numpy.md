---
id: 1ka9ot8e39esd6hg84b4mma
title: Numpy
desc: ''
updated: 1751737573817
created: 1751694639071
---

## einsum 函数

实现了爱因斯坦求和约定，用于多维数组的操作。通常效率会更高。标记法如下：
- 重复下标表示在该维度上进行求和
- 唯一下标表示保留该维度
- 下标顺序决定输出数组的形状

关键在于理解下标规则。左右两侧关注下标，不存在的下标则求和来压缩。

```py
numpy.einsum(subscripts, *operands)
```

subscripts 的语法中，输入操作数之间用逗号分隔，这些操作数会隐式地先执行对应元素的乘法（广播后），然后按照下标规则进行求和。

乘法执行顺序分为两个阶段：
1.  **广播乘法阶段**：所有输入数组在匹配维度上进行广播和元素级乘法，也就是说，只要两个数组在维度上有匹配的部分，就会执行乘法
2.  **求和约简阶段**：对未出现在输出下标中的维度进行求和，从而压缩此维度

下标表示：
-   **单字符**：每个字符代表一个维度（如 `i, j, k`）
-   **箭头左侧**：输入数组的维度
-   **箭头右侧**：输出数组的维度
-   **省略号**：`...` 表示多个维度

运算规则：
1.  **重复下标**：在输入中出现但不在输出中出现的下标会被求和，以压缩维度
2.  **唯一下标**：保留在输出中的下标决定输出形状
3.  **下标顺序**：输出下标顺序决定结果数组的维度顺序

### 矩阵

```py
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
```

矩阵乘法可以如下实现。k 是重复的，代表在 k 的维度求和。即在第一个矩阵 A 是第二维，矩阵 B 是第一维。

```py
# 矩阵乘法
# 看到 j 是重复的，
matmul = np.einsum('ik,kj->ij', A, B)
# [[19,22],
#  [43,50]]
```

计算过程：
1. 创建临时数组 temp[i,k,j] = A[i,k] * B[k,j]
   - 对于每个 i,j 组合：temp[i,0,j] = A[i,0] * B[0,j]
                temp[i,1,j] = A[i,1] * B[1,j]
2. 对 k 维度求和（k未出现在输出中）：
   result[i,j] = Σₖ temp[i,k,j]

迹（对角线求和）。ii 表示在两个位置重复出现（行和列维度）。箭头左侧没有维度即 0 维，代表只有一个数字，而非一个维度。

```py
trace = np.einsum('ii->', A)  # 1 + 4 = 5
```

逐元素乘法（哈达玛积）。ij 在输入输出都出现，没有重复下标被忽略，不用求和来压缩维度。

```py
hadamard = np.einsum('ij,ij->ij', A, B)
# [[5,12],
#  [21,32]]
```

### 张量

```py
T = np.random.rand(2, 3, 4)  # 三维张量

# 沿特定维度求和
sum_dim0 = np.einsum('ijk->jk', T)  # 沿第0维求和
sum_dim1 = np.einsum('ijk->ik', T)  # 沿第1维求和
```

张量缩并

```py
U = np.random.rand(4, 5)
contraction = np.einsum('ijk,kl->ijl', T, U)  # 结果形状: (2,3,5)
```

计算过程：
1. 广播维度：T[i,j,k] 和 U[k,l]
2. 元素乘法：temp[i,j,k,l] = T[i,j,k] * U[k,l]，通过最后两维来广播，广播总会从后往前
3. 求和约简：对k求和（k未出现在输出中）
   result[i,j,l] = Σₖ T[i,j,k] * U[k,l]

```py
# 双线性变换
bilinear = np.einsum('ijk,il,jm,kn->lmn', T, a, b, a)
```

### 常见用法

| 操作       | einsum 表达式 | 等效 NumPy 函数                                                |
| ---------- | ------------- | -------------------------------------------------------------- |
| 向量点积   | 'i,i->'       | np.dot(a, b)，$o=\sum_i a[i] b[i]$                             |
| 矩阵乘法   | 'ij,jk->ik'   | np.matmul(A, B)                                                |
| 迹         | 'ii->'        | np.trace(A)，两个 i 都没有出现，需要求和压缩。                 |
| 转置       | 'ij->ji'      | A.T                                                            |
| 按行求和   | 'ij->i'       | A.sum(axis=1)，$output[i]=\sum_j A[i,j]$                       |
| 按列求和   | 'ij->j'       | A.sum(axis=0)                                                  |
| 外积       | 'i,j->ij'     | np.outer(a, b), 扩充为 (i,1) 和 (1,j) 维度后，矩阵乘法，即广播 |
| 提取对角线 | 'ii->i'       | np.diag(A)，$o[i]=A[i,i]，无压缩$                              |
| 张量缩并   | 'ijk,kl->ijl' | np.tensordot(T, U, axes=([2],[0]))                             |

可以看到，箭头两侧就是表达式的下标，索引最终结果的下标。

比如 'ii->i' 左侧两个 i 对应矩阵 A 的索引下标，有两个，即 `A[i,i]`，右侧则是输出的索引下标 `output[i]`。联系起来，则 `output[i] = A[i,i]`。所以参数叫做 subscripts，下标之含义。

再比如，外积 'i,j->ij'，左侧由逗号分隔下标，对应元素相乘，得到 ij 维度的结果。

压缩的例子，比如张量缩并 'ijk,kl->ijl'，左侧有重复下标 k，右侧没有 k，所以 k 维度被压缩。具体来说：$O[i,j,l] = \sum_k T[i,j,k] U[k,l]$，k 轴求和压缩。

还可以指定超过两个张量的情况，具体见注意力机制部分。

### 注意力机制的例子

比如，需要投影 x 到 qkv 使用 LoRA 的方案时，需要执行 $B A x$ 部分。实现通常使用转置，比如 $x A^T B^B$ 方案如下：

```py
# ('BSD,3KDL->3BSKL', '3BSKL,3KLH->3BSKH')
import numpy as np
# K 是注意力头的数量，D 是嵌入空间维度，L 是 rank 所在轴，维度是 LoRA 维度
xA_eqn = "BSD,TKDL->TBSKL"
x = np.random.normal(0.0, 1.0, (8, 64, 32))
A = np.random.rand(3, 8, 32, 2)
print(np.einsum(xA_eqn, x, A).shape) # (3, 8, 64, 8, 2)
```

在 subscripts 中，两个数组只有 D 是对应起来的轴，所以 einsum() 对此轴上各元素一一对应地执行乘法。

einsum() 细节如下：

维度对齐与广播。为了对应输出 TBSKL，x 与 A 都会自动扩展如下：
- x 自动扩展为: (1, B=8, S=64, 1, D=32)   # 添加了T和K维度
- A 自动扩展为: (T=3, 1, 1, K=8, D=32, L=2) # 添加了B和S维度

沿着 D 维度计算点积，方程 xA_eqn 等价于 $output[t, b, s, k, l] = \sum_d ( x[b, s, d] * A[t, k, d, l] )$。

#### scaled-dot product 注意力机制

$A_{ij} = \text{softmax}(\frac{Q_{in}K_{ijn}}{d^{\frac{1}{2}}})$

```py
Q = torch.randn(8,10)  #batch_size,query_features
K = torch.randn(8,6,10) #batch_size,key_size,key_features
d_k = K.shape[-1]
A = torch.softmax(torch.einsum("in,ijn->ij",Q,K)/d_k,-1)
```

#### 分组查询注意力

num_heads 表示 Q 头的数量，num_kv_groups 表示 K 和 V 头的分组数量，通常小于 Q 头数量，且要求能被 Q 的头整除。

使用 einsum 来避免各种转置。

```py
import torch
import torch.nn.functional as F

def grouped_query_attention(
    Q: torch.Tensor,  # [batch_size, q_seq_len, num_heads, head_dim]
    K: torch.Tensor,  # [batch_size, kv_seq_len, num_kv_heads, head_dim]
    V: torch.Tensor,  # [batch_size, kv_seq_len, num_kv_heads, head_dim]
    num_kv_groups: int  # 分组数量 G (num_heads 必须能被 G 整除)
):
    batch_size, q_seq_len, num_heads, head_dim = Q.shape
    kv_seq_len = K.shape[1]
    num_kv_heads = K.shape[2]
    
    # 验证分组合理性
    assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
    assert num_kv_heads == num_kv_groups, "K/V must have num_kv_heads = num_kv_groups"
    
    # 1. 计算注意力分数
    # Q: [b, q, h, d] -> [b, h, q, d]
    # K: [b, kv, g, d] -> [b, g, kv, d] -> [b, h, kv, d] (通过广播)
    scores = torch.einsum("bqhd,bkgd->bhqk", Q, K)  # 输出: [b, h, q, kv]
    scores = scores / (head_dim ** 0.5)
    
    # 2. 计算注意力权重
    attn_weights = F.softmax(scores, dim=-1)  # [b, h, q, kv]
    
    # 3. 加权聚合值向量
    # V: [b, kv, g, d] -> [b, g, kv, d] -> [b, h, kv, d] (通过广播)
    output = torch.einsum("bhqk,bkvd->bqhd", attn_weights, V)  # [b, q, h, d]
    
    return output
```

## Ref and Tag