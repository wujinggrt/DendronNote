---
id: vzl08s4sfll3bv4cg3guooe
title: 调参经验
desc: ''
updated: 1741847501065
created: 1741847198753
---

## 动态学习率

初期，使用较大的学习率，可以加速学习。但是，一段时间后，可能会出现震荡。所以，进一步降低学习率，能够得到更稳定的学习效果。常见的方法有 warmup 加上余弦退火。

在强化学习方面，加上噪声来探索动作，退火思路类似。比如 ManiSkill-ViTac 2025 比赛中，训练初期需要较大的噪声，否则在训练集中，几乎没有成功的 episodes。训练出成功率较高的权重后，降低噪声，降低学习率，慢慢收敛。

## Ref and Tag

#Tricks