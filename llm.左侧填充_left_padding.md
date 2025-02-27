---
id: lxhiq8961aa24q0zpqgf8be
title: 左侧填充_left_padding
desc: ''
updated: 1740627999690
created: 1740627527490
---

比如 Qwen2-VL 使用了左侧填充，即 `self.padding_side = "left"`，否则在 Flash Attention 方面出现异常。

## Q：为什么需要在左侧进行 padding？比右侧有什么优势？

### 1. 左侧填充的优势

#### (1) 保持序列的右对齐
- 右侧对齐：左侧填充可以确保序列的右侧对齐，这对于某些模型（如 Transformer）非常重要，因为模型通常会**从序列的右侧开始生成输出**。
- 生成任务：在生成任务（如文本生成）中，模型通常从序列的末尾开始生成新的 token。如果使用右侧填充，填充的 token 会干扰生成过程，而左侧填充可以避免这个问题。

#### (2) 避免填充 token 干扰注意力机制
- 注意力机制：在 Transformer 模型中，填充 token 通常会被掩码（mask）掉，以避免它们参与注意力计算。如果使用右侧填充，填充 token 会出现在序列的末尾，可能会干扰模型对有效 token 的注意力。
- 左侧填充：左侧填充将填充 token 放在序列的开头，模型可以更容易地忽略这些 token，从而更专注于有效 token。

#### (3) 更适合自回归模型
- 自回归模型：在自回归模型（如 GPT）中，模型从左到右逐 token 生成序列。如果使用右侧填充，填充 token 会出现在序列的末尾，可能会干扰生成过程。
- 左侧填充：左侧填充将填充 token 放在序列的开头，模型可以更容易地忽略这些 token，从而更专注于生成有效 token。

#### (4) 更适合固定长度的输入

固定长度输入：在某些任务中，输入序列的长度是固定的（如批处理中的序列对齐）。左侧填充可以确保序列的右侧对齐，从而更容易处理固定长度的输入。

### 2. 右侧填充的劣势

- 干扰生成过程：在生成任务中，右侧填充的 token 会出现在序列的末尾，可能会干扰模型的生成过程。
- 干扰注意力机制：右侧填充的 token 会出现在序列的末尾，可能会干扰模型对有效 token 的注意力。
- 不适合自回归模型：在自回归模型中，右侧填充的 token 会出现在序列的末尾，可能会干扰生成过程。

### 3. 左侧填充的具体应用场景

- 文本生成任务：在文本生成任务中，模型通常从序列的末尾开始生成新的 token。左侧填充可以确保填充 token 不会干扰生成过程。
- Transformer 模型：在 Transformer 模型中，左侧填充可以确保填充 token 不会干扰注意力机制。
- 批处理中的序列对齐。在批处理中，左侧填充可以确保序列的右侧对齐，从而更容易处理固定长度的输入。

### 4. 示例

```py
from transformers import AutoTokenizer

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 输入文本
texts = ["Hello, how are you?", "I am fine."]

# 左侧填充
encoded_inputs = tokenizer(
    texts,
    padding="max_length",  # 填充到最大长度
    max_length=10,         # 最大长度
    truncation=True,       # 截断超长序列
    return_tensors="pt",   # 返回 PyTorch 张量
    padding_side="left"    # 左侧填充
)

print(encoded_inputs["input_ids"])
```

输出

```
tensor([[    0,     0,     0,  15496,    11,   703,   389,   345,  1371,    30],
        [    0,     0,     0,     0,     0,    40,   716,  1438,   627,    13]])
```

### 5. 总结

左侧填充的优势：
- 保持序列的右对齐。
- 避免填充 token 干扰注意力机制。
- 更适合自回归模型和生成任务。
- 更适合固定长度的输入。

右侧填充的劣势：
- 可能会干扰生成过程和注意力机制。
- 不适合自回归模型。

因此，在生成任务和 Transformer 模型中，左侧填充通常是更好的选择。

## Ref and Tag

#LLM