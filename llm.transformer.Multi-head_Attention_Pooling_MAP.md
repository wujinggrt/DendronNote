---
id: 6z48jv3ogmmyqqafv6rdbet
title: Multi-head_Attention_Pooling_MAP
desc: ''
updated: 1740141154025
created: 1740141096046
---

Multi-head Attention Pooling 利用了多头注意力机制的优势，旨在从输入序列中提取出最具代表性的特征表示。相比于传统的平均池化或最大池化方法，MAP 能够根据输入数据的具体内容动态地调整其关注的重点，从而提供更加丰富的特征描述。一般可以用在比如 VLM 生成的内容聚合工作。

工作流程
- 输入表示：输入可以是一个序列（如文本序列或时间序列），每个元素都由一个向量表示。
- 多头注意力计算：应用多头注意力机制，为序列中的每一个位置生成一组注意力权重。
- 加权求和：利用生成的注意力权重对原始输入序列进行加权求和，得到一个固定长度的向量作为该序列的整体表示。
- 输出：这个整体表示可以被进一步用于分类、回归等下游任务。

例子：
```py
import torch
import torch.nn as nn

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # 初始化权重矩阵
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        B, T, C = x.size()  # Batch size, Time steps, Channels
        
        # 计算 Query 和 Key
        Q = self.query_proj(x).view(B, T, self.num_heads, self.head_dim)
        K = self.key_proj(x).view(B, T, self.num_heads, self.head_dim)
        
        # 计算注意力分数
        scores = torch.einsum('bthd,bshd->bths', Q, K) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 加权求和得到最终的池化结果
        pooled_output = torch.einsum('bths,bshd->bthd', attention_weights, x.view(B, T, self.num_heads, self.head_dim))
        pooled_output = pooled_output.mean(dim=2)  # 平均所有头的结果
        
        return pooled_output

# 示例调用
model = MultiHeadAttentionPooling(input_dim=64, num_heads=8)
x = torch.randn(10, 20, 64)  # 假设输入是一个形状为 (batch_size, seq_length, feature_dim) 的张量
output = model(x)
print(output.shape)  # 输出应具有形状 (batch_size, seq_length, head_dim)
```