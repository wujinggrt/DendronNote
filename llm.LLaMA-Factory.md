---
id: lvhxgyuvwyw5soqj59gc045
title: LLaMA-Factory
desc: ''
updated: 1740381763036
created: 1740375991093
---

## 教程目标

功能包括：
1. 原始模型直接推理
2. 自定义数据集构建
3. 基于LoRA的sft指令微调
4. 动态合并LoRA的推理
5. 批量预测和训练效果评估
6. LoRA模型合并导出
7. 一站式webui board的使用
8. API Server的启动与调用
9. 大模型主流评测 benchmark
10. 导出GGUF格式，使用Ollama推理

## 检查 CUDA 环境

```py
>>> import torch
>>> torch.cuda.current_device()
0
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce RTX 4090'
>>> torch.__version__
'2.6.0+cu124'
```

## 下载模型

参考 Hugginface 教程。下载后，运行官方原始的推理 demo，验证模型文件的正确性和 transformers 库是否可用。

## 原始模型直接推理

开始工作之前，先试用推理模式，验证 LLaMA-Factory 推理部分是否正常。LLaMA-Factory 带了

## Ref and Tag
[Github](https://github.com/hiyouga/LLaMA-Factory)
[知乎教程](https://zhuanlan.zhihu.com/p/695287607)

[[llm.huggingface.DeepSpeed集成]]
[[llm.DeepSpeed_核心概念]]
[[llm.Megatron]]

#LLM