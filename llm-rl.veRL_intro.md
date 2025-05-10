---
id: jn81tvhrrys05bjr7hd20kj
title: veRL_intro
desc: ''
updated: 1743225349451
created: 1739175432693
---

[link](https://arxiv.org/abs/2409.19256v2)

字节团队提出了 LLM 的 RL 框架 veRL。DataFlow，数据流是重要的计算模式抽象。神经网络是典型的数据流，可以用计算图描述，节点代表计算操作，边代表数据依赖。大模型的 RL 更复杂，比如 RLHF 需要设计多个模型的训练，比如 Actor, Critic, Reference Policy, Reward Model 等，它们之间需要传递大量的数据，涉及了不同的计算类型，比如前向反向传播、优化器更新和自回归生成等，可以采用不同并行策略来加速。

Insights：传统的分布式 RL 在单张 GPU 训练，或者使用数据并行，将控制流和计算流合并在同一进程。在小规模模型时效果良好，大模型需要多维度并行和大量分布式计算，难以应对。**HybridFlow** 解耦控制流和计算流，兼顾灵活高效。

大模型 RL 本质是

提高性能，整合，hybrid；提高扩展能力，封装。

## ref and tags

吞吐量最高飙升20倍！豆包大模型团队开源RLHF框架，破解强化学习训练部署难题 - 新智元的文章 - 知乎
https://zhuanlan.zhihu.com/p/4461725991

https://verl.readthedocs.io/en/latest/start/install.html