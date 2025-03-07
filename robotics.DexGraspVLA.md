---
id: szrfzxm3muecjxl00crlgc2
title: DexGraspVLA
desc: ''
updated: 1741330456866
created: 1741077608202
---

DexGraspVLA 是一个分层框架，使用预训练的 VLM 作为高层次任务规划器，使用扩散策略作为低级动作控制器。

## Method

domain-variance 指在不同环境、条件下，输入数据（如图像、语言指令）的变化或差异。domain-invariance 指模型能够提取与环境和条件无关的特征表示，在不同环境下，这些特征保持一致。

## 附录细节

对于头部相机获取的图片，使用 Qwen-VL-Chat 获取边框，提供给低级控制器。直接使用现有的 VLM 作为 planner，本框架可以十分轻松地替换 VLM。

### Prompts 设计

使用 prompts 提示 planner。作者设计的 prompts 分为几类。主要包含功能如下：
- understanding the user prompt
- proposing an object as the current grasping instruction，建议当前指令对应的抓取的物体
- marking the target object bounding box
- checking if the grasp has succeeded
- assessing whether the current instruction is completed
- evaluating whether the entire user prompt is fully fulfilled

### Controller

头部和腕部相机收集的图像，都会转换为 640x480x3 的分辨率，掩码则为 640x480x1.使用 DINOv2 ViT-B/14 提取头部图像特征 $\phi^h$，使用 DINOv2 ViT-L/14 提取腕部特征 $\phi^w$。在使用 DINOv2 提取特征前，将图像 resize 到 518x518x3。训练时，使用 color jittering 随机化图像。最后，归一化图像，再送往 DINOv2 模型。最后得到头部和腕部的特征 $\bold{z}_t^h\in \mathbb{R}^{1369\times768}$ and $\bold{z}_t^w\in \mathbb{R}^{1369\times1024}$。求遮掩特征时，使用随机初始化的 ViT，提取特征为 $\bold{z}^m_t\in\mathbb{R}^{1369\times768}$。通过逐 patch 拼接 $\bold{z}_t^m, \bold{z}_t^h$，得到特征 $\bar{\bold{z}}_t^h\in\mathbb{R}^{1369\times1536}$。随后，使用 MLP 投影 $\bold{\bar{z}}_t^h, \bold{z}_t^w, \bold{s}_t$ 到共同的特征空间，维度为 1024，各自得到 $\bold{\tilde{z}}_t^h\in\mathbb{R}^{1369\times1024}$, $\bold{\tilde{z}}_t^w\in\mathbb{R}^{1369\times1024}$ and $\bold{\tilde{z}}_t^s\in\mathbb{R}^{1\times1024}$。拼接得到全部的观察特征序列 $\bold{\tilde{z}}_t^{obs}=\left(\bold{\tilde{z}}_t^h,\bold{\tilde{z}}_t^w,\bold{\tilde{z}}_t^s\right)\in\mathbb{R}^{2739\times1024}$

### DiT 实现

将时间步嵌入到和 $\bold{\tilde{z}_t^{obs}}$ 相同的 hidden space，得到 $\bold{\tilde{z}_t^d}\in\mathbb{R}^{1\times1024}$，随后和观察的特征拼接起来，得到 $\bold{\tilde{z}_t}=\left(\bold{\tilde{z}_t^{obs}}, \bold{\tilde{z}_t^d}\right)\in\mathbb{R}^{2740\times1024}$。

将 noise chunk 投影到相同的空间，其中，action chunk horizon H=64，于是 noised action chunk $\bold{\hat{A}}\in\mathbb{R}^{64\times13}$，最后得到特征 $\bold{\tilde{z}_t^A}\in\mathbb{R}^{64\times1024}$。随后，送到 DiT。每个 DiT 层对 action tokens 执行双向注意力，对 condition sequence 执行交叉注意力，再包含一个 MLP 投影。最后，输出投影回动作空间，即模型对噪声的预测。使用 DDIM 扩散和去噪。

Controller 使用了 163M 的可训练参数。为了加速训练，使用 bfloat16 混合精度训练，减少存储和提升训练。对我们的数据集，训练了 84 个 epochs，使用 8-A800 GPU 服务器训练，在一天之内训练完成。

## Ref and Tag

[灵初智能发布端到端VLA模型Psi R0.5，仅需两小时数据实现物品、场景全面泛化](https://mp.weixin.qq.com/s/55l129vnMl3ysoXRFBpp3w)

[主页](https://dexgraspvla.github.io/)
[paper](https://arxiv.org/abs/2502.20900)