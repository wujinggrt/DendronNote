---
id: ymw3ox8p06lb7fbkm0xe2ir
title: Transformers_模型输出
desc: ''
updated: 1740648422091
created: 1740638713570
---

关注 dataclass `Qwen2VLCausalLMOutputWithPast` 和 `ModelOutput`。借助 LLM 理解。注意，`ModelOutput` 继承自标准库 `OrderedDict`。但是为什么要做这个封装？需要探究几个问题：
1. OrderedDict 用法
2. ModelOutput 封装了什么
3. Qwen2VLCausalLMOutputWithPast 封装了什么

但是，一般模型的 `forward()` 函数中，`return_dict` 默认 None，所以通常返回 `Tuple`。

## Ref and Tag

[[llm.huggingface.transformers_Trainer]]
[[llm.huggingface.Transformers库用法]]

