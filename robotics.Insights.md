---
id: 4dxai00llv4i5bruqnkpxnv
title: Insights
desc: ''
updated: 1740132257937
created: 1740130941606
---

## HiRT, Helix, DexVLA
[[robotics.HiRT_使用分层机器人Transformer提示机器人控制]]，[[robotics.Helix：用于通才人形机器人控制的_VLM]]，[[robotics.DexVLA]]

HiRT 和 Helix 解耦了机器人使用大模型和策略模型，在处理时延时，使用异步策略。把 VLM 当作大型的视觉-语言编码器，动作策略用 token 当作条件来训练。如果一次编码生成多个 token，是否意味着是规划的多个动作，需要序列执行？或者是并行执行？多个 token，可能是双臂协作，四臂协作。

想法：在视频方面，特别是腕部，使用一张图像，不断放大，模拟机械臂接近物体。随后可以用这些放到某个物体的图像制作伪视频，类比机械臂接近物体，用于强化学习。借鉴 [[rl.LongShortTermImagination]]。