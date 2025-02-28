---
id: u8f5rxjjf1lq8qp7dqg52y9
title: FiLM_Feature_wise_Linear_modulation_多模态ronghe
desc: ''
updated: 1740712340991
created: 1740629832519
---

用于视觉推理和视觉问答，用于结合文本模态与图片模态。根据图片，给出回答：

![fig1](assets/images/llm.FiLM_Feature_wise_Linear_modulation_多模态融合/fig1.png)

FiLM 可以理解为在feature的层面上，对特征做了一次线性变换。具体做法如下：

FiLM learns functions $f$ and $h$ which output $γ_{i,c}$ and $β_{i,c}$ as a function of input $x_i$:

![公式](assets/images/llm.FiLM_Feature_wise_Linear_modulation_多模态融合/公式.png)

![film](assets/images/llm.FiLM_Feature_wise_Linear_modulation_多模态融合/film.png)

架构如下：

![architecture](assets/images/llm.FiLM_Feature_wise_Linear_modulation_多模态融合/architecture.png)

## QA

Q:  我们要讨论的论文是FiLM: Visual Reasoning with a General Conditioning Layer，链接是  https://arxiv.org/pdf/1709.07871  ，已有的FAQ链接是  https://papers.cool/arxiv/kimi?paper=1709.07871  。请以此为基础，继续回答我后面的问题。

这篇论文是 2017 年的，现在融合都有哪些更常用的工作？

## Ref and Tag

[arxive](https://arxiv.org/abs/1709.07871)

#MLLM
#Paper