---
id: 4gb9ottxmfh95i6654zy8hq
title: DexVLA复现
desc: ''
updated: 1740158833865
created: 1740053039805
---

## 数据准备
项目使用了 act 工作的数据格式，act 每个 timestep 数据格式如下：

observations
- images
    - each_cam_name     (480, 640, 3) 'uint8'
- qpos                  (14,)         'float64'
- qvel                  (14,)         'float64'

action                  (14,)         'float64'

随后，作者使用 rlds_to_h5py，把数据转换为 h5py 格式。

## VLM
使用 [Qwen2-2B-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) 作为主干网络。也许可以尝试 [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)。

模型结构，在 VLM 末尾增加一个 policy head；而 Helix 直接输出 token，当作 policy 模型的 latent vector。

## 视觉编码器条件化
两个方案：
- FiLM 层 (CNN 架构)：在 EfficientNet 的隐藏层
- 交叉注意力 (Trasnformer 架构)：在自注意力后插入跨注意力。

## TODO
制作 PPT，复现此项目。

Figure 的 Helix 思路与此十分相似。Helix 使用了较大的 VLM (7B) 作为主干，使用较小的策略模型 (80M) 生成动作。解耦了大模型和小模型。大模型

Helix 和 HiRT 并未开源代码和模型，DexVLA 开源了代码，复现可能更大。但是，在动作生成方面，还是使用了 action head 层动作学习的网络接在 VLM 模型，VLM 使用 Qwen2-3B-VL。

HiRT 发表了论文，解决了 VLM 模型与策略模型生成速度不匹配的情况。主要使用异步的方案，把 VLM 当做一个大型的编码器，编码视觉和自然语言指令。

问题分析：大模型生成较慢，动作策略的小模型生成较快，DexVLA 并没有解决此问题，还是使用同步的方案；泛化场景问题。

可行性：DexVLA 开源，有框架遵循，有复现可能。使用的 VLM 模型是 2B，最近，千问发表了 Qwen2.5-VL 系列。可以使用可能更优秀的 Qwen2.5-3B-Instruct，使用 DeepSpeed，两张显卡猜测能够微调。在数据收集方面，有 pny 做过数据收集，使用的数据格式类似。

下一步打算：先复现，后修改，不断逼近 Helix 的方案。

## Tag and Ref
[[robotics.DexVLA]]
[[robotics.Helix：用于通才人形机器人控制的_VLM]]
[[robotics.HiRT_使用分层机器人Transformer提示机器人控制]]

#复现