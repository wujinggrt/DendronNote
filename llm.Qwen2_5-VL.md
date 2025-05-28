---
id: 2nbimo1w05qd1lpgkghblc7
title: Qwen2.5-VL
desc: ''
updated: 1748316208806
created: 1740130046405
---

## 摘要

此工作旨在提升大型视觉语言模型（LVLM）在细粒度视觉感知、文档解析、物体定位和长视频理解方面的能力。

## 核心思路

Qwen2.5-VL 的核心思路在于通过以下几个方面提升 LVLM 的性能：
- **优化视觉编码器**：引入窗口注意力机制，提高推理效率。
- **动态 FPS 采样**：扩展动态分辨率到时间维度，实现对不同采样率视频的全面理解。
- **绝对时间对齐的 MRoPE**：在时间域升级 MRoPE，通过与绝对时间对齐，促进更复杂的时间序列学习。
- **高质量数据**：投入大量精力整理高质量的预训练和监督微调数据，并将预训练语料库从 1.2 万亿 tokens 扩展到 4.1 万亿 tokens。
- **原生动态分辨率**：模型可以直接处理不同分辨率的图片，无需 resize。通过原生动态分辨率和绝对时间编码，Qwen2.5-VL 能够处理不同大小的图像和长时间视频，并能精确定位到秒级的事件。
- **增强 Agent 能力**：通过高级的定位、推理和决策能力，Qwen2.5-VL 提升了在智能手机和电脑等真实场景中的 Agent 功能。

## 方案与技术

**模型架构**：
- **LLM**：采用 Qwen2.5 LLM 的预训练权重作为初始化，并修改了 1D RoPE 为多模态旋转位置嵌入（Multimodal Rotary Position Embedding Aligned to Absolute Time, MRoPE），并与绝对时间对齐。
- **视觉编码器**：重新设计的 Vision Transformer (ViT) 架构，集成了 2D-RoPE 和窗口注意力机制，以支持原生输入分辨率并加速计算。
- **基于 MLP 的 Vision-Language Merger**：采用基于 MLP 的方法**压缩**图像特征序列，将空间相邻的四个 patch 特征组合并通过两层 MLP 投影到与 LLM 文本嵌入对齐的维度。

**快速高效的视觉编码器**：
- 在大多数层中引入窗口注意力机制，确保计算成本与 patch 数量呈线性关系。
- 采用 2D 旋转位置嵌入（RoPE）来有效捕获 2D 空间中的空间关系。
- 对于视频数据，将连续两帧分组在一起，减少输入到语言模型的 tokens 数量。
- 采用 RMSNorm 进行归一化，并使用 SwiGLU 作为激活函数，以提高计算效率和视觉与语言组件之间的兼容性。

MRoPE 位置编码把 text, image, video 三种模态统一作位置编码，作用于 LLM。每个 token 用 [t, h, w] = (frame_idx, height_idx, width_idx) 三个位置表示。

**原生动态分辨率和帧率**：
- 在空间域，动态地将不同大小的图像转换为具有相应长度的 tokens 序列。
- 对于视频输入，采用动态帧率（FPS）训练和绝对时间编码。
- 将 MRoPE IDs 直接与时间戳对齐，使模型能够通过时间维度 IDs 之间的间隔来理解时间节奏。

**多模态旋转位置嵌入（MRoPE）**：
- 将位置嵌入分解为时间、高度和宽度三个不同的组成部分。
- 通过利用时间 IDs 之间的间隔，模型能够学习跨不同 FPS 采样率视频的一致时间对齐。

**数据**
- 预训练数据：预训练数据量从 1.2 万亿 tokens 增加到约 4 万亿 tokens，包括图像描述、交错图像 - 文本数据、OCR 数据、视觉知识、多模态学术问题、定位数据、文档解析数据、视频描述、视频定位和基于 agent 的交互数据。
- 指令数据：使用 ChatML 格式构建指令遵循数据，包括纯文本数据和多模态数据（图像 - 文本和视频 - 文本组合），并采用双阶段优化范式，包括监督微调（SFT）和直接偏好优化（DPO）。

**训练方法**
- 预训练：分三个阶段训练，首先训练 ViT，然后训练所有模型参数，最后增加序列长度并加入视频和基于 agent 的数据。
- 后训练：采用监督式微调（SFT）和直接偏好优化（DPO）的双阶段优化模式，冻结 ViT 参数。

Grounding Data with Absolute Position Coordinates. 原生尺寸图像更能接近真实世界信息。如何增强 bbox 和 point 定位能力？

## QA

窗口注意力是什么？

### 什么是绝对时间对齐？

### 相比原来的 RoPE，MRoPE 有什么不同？

### 在多模态融合中采用简单的 MLP 投影器，这是否成为性能瓶颈？相比更复杂的跨模态注意力机制，这种设计有何优劣？

MLP 投影器在效率与性能间取得平衡，但在需要深层次模态交互的任务中可能成为瓶颈。未来可探索混合架构（如局部跨模态注意力 + 全局 MLP）。

**优势**：
- 计算高效：MLP 投影的复杂度为 O(N×D)，远低于跨模态注意力的 O(N×M×D)（M 为文本 token 数），适合处理长视觉序列。
- 训练稳定：避免跨模态注意力中梯度消失或对齐困难的问题。

**劣势**：
- 信息损失：MLP 可能无法充分建模视觉与语言 token 的细粒度交互，例如空间位置关联（如「图片左上的红色物体」）。
- 模态鸿沟：复杂任务（如需要视觉推理的数学问题）可能受益于更深的模态交互，而 MLP 投影器可能力有未逮。

## 启动 Service

```bash
# pip install 'vllm>0.7.2'
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5
```

使用 chat API：
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustrate?"}
    ]}
    ]
    }'
```

## 模型结构

```
Qwen2_5_VLForConditionalGeneration(
  (visual): Qwen2_5_VisionTransformerPretrainedModel(
    (patch_embed): Qwen2_5_VisionPatchEmbed(
      (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
    )
    (rotary_pos_emb): Qwen2_5_VisionRotaryEmbedding()
    (blocks): ModuleList(
      (0-31): 32 x Qwen2_5_VLVisionBlock(
        (norm1): Qwen2RMSNorm((1280,), eps=1e-06)
        (norm2): Qwen2RMSNorm((1280,), eps=1e-06)
        (attn): Qwen2_5_VLVisionSdpaAttention(
          (qkv): Linear(in_features=1280, out_features=3840, bias=True)
          (proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (mlp): Qwen2_5_VLMLP(
          (gate_proj): Linear(in_features=1280, out_features=3420, bias=True)
          (up_proj): Linear(in_features=1280, out_features=3420, bias=True)
          (down_proj): Linear(in_features=3420, out_features=1280, bias=True)
          (act_fn): SiLU()
        )
      )
    )
    (merger): Qwen2_5_VLPatchMerger(
      (ln_q): Qwen2RMSNorm((1280,), eps=1e-06)
      (mlp): Sequential(
        (0): Linear(in_features=5120, out_features=5120, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=5120, out_features=3584, bias=True)
      )
    )
  )
  (model): Qwen2_5_VLModel(
    (embed_tokens): Embedding(152064, 3584)
    (layers): ModuleList(
      (0-27): 28 x Qwen2_5_VLDecoderLayer(
        (self_attn): Qwen2_5_VLSdpaAttention(
          (q_proj): Linear(in_features=3584, out_features=3584, bias=True)
          (k_proj): Linear(in_features=3584, out_features=512, bias=True)
          (v_proj): Linear(in_features=3584, out_features=512, bias=True)
          (o_proj): Linear(in_features=3584, out_features=3584, bias=False)
          (rotary_emb): Qwen2_5_VLRotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=3584, out_features=18944, bias=False)
          (up_proj): Linear(in_features=3584, out_features=18944, bias=False)
          (down_proj): Linear(in_features=18944, out_features=3584, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((3584,), eps=1e-06)
    (rotary_emb): Qwen2_5_VLRotaryEmbedding()
  )
  (lm_head): Linear(in_features=3584, out_features=152064, bias=False)
)
```

## 官网例子：

## special tokens

微调时，看到 Qwen2TokenizerFast 如下：

```
tokenizer: Qwen2TokenizerFast(name_or_path='/data1/wj_24/huggingface/Qwen/Qwen2.5-VL-3B-Instruct', vocab_size=151643, model_max_length=131072, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']}, clean_up_tokenization_spaces=False, added_tokens_decoder={ 151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151646: AddedToken("<|object_ref_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151647: AddedToken("<|object_ref_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151648: AddedToken("<|box_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151649: AddedToken("<|box_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151650: AddedToken("<|quad_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151651: AddedToken("<|quad_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151652: AddedToken("<|vision_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151653: AddedToken("<|vision_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151654: AddedToken("<|vision_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151655: AddedToken("<|image_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151656: AddedToken("<|video_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True), 151657: AddedToken("", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False), 151659: AddedToken("<|fim_prefix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False), 151660: AddedToken("<|fim_middle|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False), 151661: AddedToken("<|fim_suffix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False), 151662: AddedToken("<|fim_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False), 151663: AddedToken("<|repo_name|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False), 151664: AddedToken("<|file_sep|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
```

看到特殊 token 为：

```py
special_tokens={
  'eos_token': '<|im_end|>', 
  'pad_token': '<|endoftext|>', 
  'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']
}, 
```

## 能力类别

各家VL模型或多模态模型的能力大幅偏向OCR/物体识别/HTML类任务，稍微带点逻辑的题目基本都是0分。

## Tag and Ref
[【多模态大模型】Qwen2.5-VL解剖 - Plunck的文章 - 知乎](https://zhuanlan.zhihu.com/p/24986805514)

[【论文解读】Qwen2.5-VL：更「真实」的全能视觉语言模型 - tomsheep的文章 - 知乎](https://zhuanlan.zhihu.com/p/25296368487)

[Qwen2.5-VL相比Qwen2-VL的主要改进 - 特里斯丹的文章 - 知乎](https://zhuanlan.zhihu.com/p/25692213650)

[官网博客介绍](https://qwenlm.github.io/blog/qwen2.5-vl/)

Qwen系列解读：回顾Qwen2.5-VL，目前最好的多模态开源算法之一 - 曾天真的文章 - 知乎
https://zhuanlan.zhihu.com/p/1897297231637357678

QWen2.5 VL 阅读记录 - 奇奇的文章 - 知乎
https://zhuanlan.zhihu.com/p/26113026053

#LLM