---
id: eszoks0gixjd0fjyul4wh10
title: Robotics
desc: ''
updated: 1740416004206
created: 1740293600917
---

## HiRT, Helix, DexVLA
[[robotics.HiRT_使用分层机器人Transformer提示机器人控制]]，[[robotics.Helix：用于通才人形机器人控制的_VLM]]，[[robotics.DexVLA]]

HiRT 和 Helix 解耦了机器人使用大模型和策略模型，在处理时延时，使用异步策略。把 VLM 当作大型的视觉-语言编码器，动作策略用 token 当作条件来训练。如果一次编码生成多个 token，是否意味着是规划的多个动作，需要序列执行？或者是并行执行？多个 token，可能是双臂协作，四臂协作。

想法：在视频方面，特别是腕部，使用一张图像，不断放大，模拟机械臂接近物体。随后可以用这些放到某个物体的图像制作伪视频，类比机械臂接近物体，用于强化学习。借鉴 [[rl.LongShortTermImagination]]。


### 关注 VLM 推理和规划的能力

DexVLA 本质还是 VLM 预测并调用技能库。重要的还是让模型具有更强大的规划能力。DexVLA 提出使用 sub-step 标注的方法，提高长范围规划能力。

提供规划能力，是推理模型？使用微调，还是强化学习，还是修改网络结构？资源有限，探究小模型的可能。

多模态中，图片是一个模态，语言是一个模态，可以查看 Qwen2.5-VL 如何增强此推理能力，还有关于 Math/Coding 的推理能力增强工作。

是否需要一个动作的编码器？

## 动作指令需要简洁，规划需要有序
可以规定，动作指令不应该超过多少文本 (比如 20 个 tokens)，以细粒度来保证精简。训练 policy 时可以把此文本作为 condition 加入起来。

此文本可以有两个选择：(1) 是直接给 policy 嵌入一个可学习的嵌入层；(2) 是直接用 VLM 编码的 token 作为潜在表示来学习。

规划层，则把动作指令分解为一系列动作，包含一个停止动作。

探索式地，自增长技能，补充技能库。我想，这是把动作指令编码为 token，才有可能实现。但是，平滑工作十分重要，离散的情况，不能够有效和稳定地学习。基于技能库，不断探索，修改。这部分如何证实正在做，如何观测，是一个难点。思路：token 比文本可能蕴含更懂信息，压缩得更准确。那么，需要保证两个模型，policy 模型和 VLM 都能一直理解，则要求 VLM 能够解译 token，代表理解，比如 prompt 解释这个 token 对应什么动作，需要完成什么目的，即 What and How，应该再来个 why，为什么能成功，是基于什么推理，Why 则对模型提出了推理能力的要求。而 policy 就像小脑，和脊柱神经，快速执行。

## VLA 解释 token，policy 理解和执行 token
需要微调 (全量 or LoRA) VLM 来解释 token。每个 token 就像加密的暗号，需要 VLM 和 policy 能够解译，就像古代的虎符，对应权力。比如，在图中看到了 XXX， Xxx 是目标，token 要求，需要怎样操作它。以前的 VLM 只关心输出，而不关心解释，这个 idea 引入了自解释。自解释使用另一个 reward model，增强潜向量生成能力。

以 VLA 为出发点去探索。

最后会只需要 Decoder 吗？当数据足够大，是否可以参考语音工作 ([Step-Audio](https://github.com/stepfun-ai/Step-Audio)) 的对齐。

DexVLA 附录的消融实验指出，对扩散专家的预训练是十分重要的。所以对 policy 要预训练。扩散专家引入了文本，可以引入 token 作为 Condition，但是粒度更小。粒度越小，policy 的动作 horizon 越小，那么要求 VLM 规划的范围越大 (即包含更多的 step，潜 token 更多）。把规划输出文字变为规划输出 token，可以使得控制动作的粒度更小，这在动作上是一个 tradeoff，因为动作难以对应多个文本，文本表示内容太多了，应当压缩为 token 来给 policy 提供条件。token 要可解释，起到精简的效果。token 就像神经递质。

安全和可靠性，验证 token 是否正确生成；token 是否正确执行。

## Ref and Tag