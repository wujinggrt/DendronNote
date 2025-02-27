---
id: u8f5rxjjf1lq8qp7dqg52y9
title: FiLM_Feature_wise_Linear_modulation_多模态ronghe
desc: ''
updated: 1740630514296
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

## Ref and Tag

[arxive](https://arxiv.org/abs/1709.07871)

#MLLM
#Paper