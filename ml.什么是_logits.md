---
id: tiivi6kto1v85pea6vpjk83
title: 什么是_logits
desc: ''
updated: 1740636083390
created: 1740635846410
---

logits 代表模型的最后一层输出且未经过归一化处理的原始预测值。即没有激活的的输出。一般不能直接使用。比如在 LLM 中，输出了还需要经过最后的词嵌入层映射到对应的 vocab_size 维度的输出。tokenizer 才能够解码。

## Ref and Tag