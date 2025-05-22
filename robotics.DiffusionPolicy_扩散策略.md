---
id: 5547aabs82dmefk71h8g4bm
title: DiffusionPolicy_扩散策略
desc: ''
updated: 1747906272099
created: 1747903019367
---

## 项目实现

以 pusht_real 数据集为例。

### Action Chunk 的概念

Action chunking 将连续的动作序列划分为固定长度的“块”（chunk），每个块对应未来一段时间执行的动作序列。

Action Chunk 是 Action Chunking 中划分的连续动作序列，即策略一次性预测的 k 个时间步的动作集合。例如，若 k=100，策略会直接生成接下来 100 帧的机器人关节目标位置，而非逐帧生成。

## ALOHA Unleashed

### Policy

策略生成绝对关节角位置。使用长度为 50 的 action chunks，策略生成张量的形状为 (50,14)。

## Ref and Tag

