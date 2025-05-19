---
id: f05p6c9udsrbsu66kxhfesi
title: VACE_All_in_One_视频生成和编辑集成到一个模型
desc: ''
updated: 1747593950506
created: 1747559246932
---

使用了一个叫 VCU (Video Condition Unit) 的核心模块，这玩意就像一个翻译官，把不同任务的 “指令” 统一成模型能听懂的格式。对模型来说，就是一个【文、图、蒙版】的三元组。

如果用户只输入了文字指令，模型发现有文字，但图片和蒙版都没要求，就知道是文生视频指令；要是参考图生成视频，那输入里肯定会有参考图和提示词，模型看见参考图和文字，就理解要让它干图生视频的活儿了。

这样，模型不用为每个任务单独学习，而是通过这三个要素的排列组合，灵活处理各种需求。实现了一个模型顶多个用的效果。

![vcu_unit](assets/images/diffusion.VACE_All_in_One_视频生成和编辑集成到一个模型/vcu_unit.png)

## 论文总结

### 作者、团队信息、论文标题、论文链接、项目主页
- **作者**: Zeyinzi Jiang*, Zhen Han*, Chaojie Mao*, Jingfeng Zhang, Yulin Pan, Yu Liu  
- **团队**: Tongyi Lab, Alibaba Group  
- **论文标题**: VACE: All-in-One Video Creation and Editing  
- **论文链接**: 未直接提供（通常需通过 arXiv 或会议页面获取）  
- **项目主页**: [VACE Project Page](https://ali-vilab.github.io/VACE-Page/)

### 主要贡献
1. **统一框架 VACE**: 首次提出基于 Diffusion Transformer (DiT) 的统一视频生成与编辑框架，支持多任务（文本生成视频、参考生成视频、视频编辑、掩码编辑等）和任务组合。
2. **Video Condition Unit (VCU)**: 提出统一的多模态输入接口，将文本、图像、视频、掩码整合为标准化表示，支持跨时空维度的条件控制。
3. **Context Adapter 结构**: 通过可插拔的上下文适配器注入时空维度特征，实现任务概念的灵活解耦与融合。
4. **VACE-Benchmark**: 构建包含 12 种任务的评估数据集（480 样本），填补视频多任务评估的空白。

### 方法
#### 核心组件
1. **Video Condition Unit (VCU)**:
   - 输入格式: $V = [T; F; M]$（文本 $T$、视频帧序列 $F=\{u_1,u_2,\dots u_n\}$、掩码序列 $M=\{m_1,m_2,\dots m_n\}$）。u 是 RGB 的，归一化到 [-1, 1]。m 是二值的，1 和 0 分别代表编辑与不编辑（比如指定首尾帧，或者指定中间帧，则不预测和推理那几帧，只是参与模型预测）。F 和 M 的空间尺寸为 h x w，时间尺寸为 n。F 和 M 都是针对像素级的，所以有针对特定像素内容修改的能力。
   - 任务编码: 通过空帧（0）和全掩码（1）统一不同任务的输入（如表 1）。
2. **Context Adapter**:
   - **Concept Decoupling**: 将输入分解为反应帧（$F_c = F \times M$）和非活动帧（$F_k = F \times (1-M)$），分别处理需修改和保留的内容。
   - **Context Tokenization**: 将 $F_c$、$F_k$ 和 $M$ 编码为与噪声视频隐变量对齐的时空特征。
   - **Adapter Tuning**: 通过旁路 Transformer Blocks 注入任务特征到主 DiT 分支，支持参数高效微调。

输入通常只有 4 个模态：文本、图像、视频和掩码，整理出 5 种任务类型：
- Text-to-Video Generation (T2V): 不需要 context frame 和 mask，为了一致，全部设置为 0，即 $0_{h\times w}$，代表空白输入；mask 设为 $1_{h \times w}$，代表全部都需要重新生成
- Reference-to-Video Generation (R2V): 使用图像作为参考输入，假设有 l 张图像，前面作为参考。mask 的前 l 位也对应 0，代表这些像素不用修改或编辑，1 则对应需要生成的。
- Video-to-Video Generation
- Masked Video-to-Video Editing (MV2V): F 和 M 都需要显示提供，根据 3D region of interest (ROI)，仅对 ROI 部分像素修改。
- Task Composition: 
   - reference-inpaiting 任务的 context frames $\{r_1, r_2, \dots, r_l, u_1, u_2, \dots, u_n\}$，context masks $\{0_{h \times w}\} \times l + \{m_1, m_2, \dots, m_n\}$。用户可以指定修改视频中 l 个物体，基于此来生成视频

| Tasks | Frames (F's) & Masks (Ms)                                       |
| ----- | --------------------------------------------------------------- |
|  T2V  | $F = \{0_{h \times w}\} \times n \\ M = \{1_{h \times w}\} \times n$                               |
| R2V   | $F = \{r_1, r_2, ..., r_l\} + \{0_{h \times w}\} \times n \\ M = \{0_{h \times w}\} \times l + \{1_{h \times w}\} \times n$      |
| V2V   | $F = \{u_1, u_2, ..., u_n\} \\ M = \{1_{h \times w}\} \times n$                                    |
| MV2V  | $F = \{u_1, u_2, ..., u_n\} \\ M = \{\dot{m}_1, m_2, ..., \dot{m}_n\}$                                    |

可以看到，F 中的 0 代表空白，需要生成图像，通常伴随 mask 为 1。

![task_categories](assets/images/diffusion.VACE_All_in_One_视频生成和编辑集成到一个模型/task_categories.png)



Insights：是否可以通过输入视频模态，让机械臂决定从上抓或下抓。实现 In-Context Learning。

## Ref and Tag