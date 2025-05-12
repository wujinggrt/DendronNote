---
id: ofrxkx80oycjvdala8m9sik
title: RDT
desc: ''
updated: 1746981130614
created: 1746971411816
---

## 论文总结

### 作者、团队信息、论文标题、论文链接、项目主页

*   **作者：** Songming Liu, Lingxuan Wu*, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su†, Jun Zhu (*表示同等贡献, †表示通讯作者)
*   **团队信息：** 清华大学计算机科学与技术系、人工智能研究院、国家级研究中心 (BNRist Center)、清华-博世联合机器学习中心、清华大学智能产业研究院 (THBI Lab)
*   **论文标题：** RDT-1B: A DIFFUSION FOUNDATION MODEL FOR BIMANUAL MANIPULATION (RDT-1B: 面向双臂操作的扩散基础模型)
*   **论文链接：** [https://arxiv.org/abs/2410.07864](https://arxiv.org/abs/2410.07864)
*   **项目主页：** https://rdt-robotics.github.io/rdt-robotics/

### 主要贡献

1.  **提出了 RDT (Robotics Diffusion Transformer)：** 一个开创性的基于扩散和 Transformer 的双臂操作基础模型，有效表示多模态动作分布。
2.  **创新的可扩展 Transformer 设计：** 专门用于处理多模态输入的异构性，并捕捉机器人数据（如非线性、高频变化）的特性，通过 QKNorm & RMSNorm、MLP 解码器和交替条件注入 (ACI) 等改进实现。
3.  **引入 PIUAS (Physically Interpretable Unified Action Space)：** 物理可解释的统一动作空间，能够统一不同机器人的动作表示，同时保留原始动作的物理意义，促进了可迁移物理知识的学习和跨机器人预训练。
4.  **大规模预训练与模型扩展：** 在迄今为止最大的多机器人数据集（46 个数据集，超过 100 万个 episodes）上预训练 RDT，并将其参数量扩展到 1.2B，成为目前最大的基于扩散的机器人操作基础模型。
5.  **高质量微调数据集与卓越性能：** 在自建的包含超过 6000 个 episodes 的多任务双臂操作数据集上微调 RDT，并在真实机器人实验中证明其显著优于现有方法，展现了出色的零样本泛化能力（未见物体、场景、指令）、仅需 1-5 个示例的少样本学习能力，以及处理复杂灵巧任务的能力。

### 研究背景（研究问题，研究难点和相关工作）

*   **研究问题：**
    双臂操作在机器人完成真实世界任务中至关重要。开发一个能够泛化到未见场景（如未见物体和环境）的通用双臂操作策略，即基础模型，是一个极具潜力的研究方向。

*   **研究难点：**
    1.  **双臂协调的复杂性：** 协调两个机械臂的固有复杂性导致动作分布呈现多模态特性。
    2.  **训练数据稀缺：** 双臂系统的成本高昂导致训练数据严重不足，与基础模型对数据量的需求形成根本性冲突。
    3.  **架构限制：** 现有模型难以有效表达多模态动作，并且在处理异构多模态输入（文本、视觉、动作）及大规模稳定训练方面存在可扩展性问题。
    4.  **数据异构性：** 不同机器人间物理结构和动作空间的差异（数据异构性）给跨机器人学习带来了负迁移风险，牺牲了数据多样性。

*   **相关工作：**

    | 领域研究               | 已有方法                                                                                                             | 局限性                                                                                                                             | 本文改进                                                                                            |
    | :--------------------- | :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- |
    | 传统双臂操作           | 依赖任务特定的原语 (Mirrazavi Salehian et al., 2017; Rakita et al., 2019; Grannen et al., 2023a)                     | 泛化能力差，难以适应新场景                                                                                                         | 通过大规模数据学习通用策略                                                                          |
    | 小规模学习双臂操作     | 小模型、小数据、简单任务 (Krebs et al., 2021; Franzese et al., 2023; Grannen et al., 2023b; Zhao et al., 2023)       | 泛化能力窄，无法处理复杂任务                                                                                                       | 构建大规模基础模型 (RDT-1B)                                                                         |
    | 单臂机器人基础模型     | 跨机器人预训练 (Brohan et al., 2023; Kim et al., 2024)                                                               | 双臂动作空间更大，多模态性更强；现有方法在动作表示和数据异构性处理上仍有不足                                                       | 引入扩散模型处理多模态动作，PIUAS 处理数据异构性                                                    |
    | 现有机器人基础模型架构 | VLM 直接预测动作 (Brohan et al., 2022; Driess et al., 2023)，动作离散化                                              | 存在量化误差，在双臂任务中行为不协调；难以表达连续空间的多模态性                                                                   | RDT 采用扩散模型生成连续动作，Transformer 架构保证可扩展性                                          |
    | 机器人数据异构性处理   | 丢弃结构不一致的机器人数据，或仅保留跨机器人不变的特征 (Brohan et al., 2023; Ghosh et al., 2023; Shah et al., 2023a) | 损失了宝贵的数据多样性                                                                                                             | PIUAS 统一不同机器人的动作表示，保留物理意义，促进知识迁移                                          |
    | 机器人领域的扩散模型   | 用于连续控制 (Chi et al., 2023; Pearce et al., 2023; Ghosh et al., 2023)                                             | 模型规模相对较小 (如 Octo 93M)；未专门针对双臂操作的多模态性和大规模数据进行优化；机器人数据特性 (非线性、高频) 对模型设计提出挑战 | RDT 扩展至 1.2B 参数，并对 DiT 进行多项适应机器人数据特性的修改 (QKNorm, RMSNorm, MLP Decoder, ACI) |

*   **Mermaid 总结：**
    ```mermaid
    graph LR
        A[双臂操作基础模型研发] --> B{核心挑战};
        B --> C1[双臂协调复杂性：多模态动作];
        B --> C2[数据稀缺];
        B --> C3[模型架构局限];
        B --> C4[跨机器人数据异构性];

        D[相关工作] --> E1[传统方法: 任务原语];
        E1 --> F1[局限: 泛化差];
        D --> E2[小规模学习];
        E2 --> F2[局限: 任务简单/泛化窄];
        D --> E3[单臂基础模型];
        E3 --> F3[启发: 跨机器人预训练];
        E3 --> F3_Lim[局限: 双臂更复杂];
        D --> E4[现有VLM/离散化];
        E4 --> F4[局限: 量化误差/不协调];
        D --> E5[机器人扩散模型];
        E5 --> F5[启发: 连续控制];
        E5 --> F5_Lim[局限: 规模小/未针对双臂优化];

        G[本文RDT-1B贡献] --> H1[扩散模型 + Transformer: 处理C1, C3];
        G --> H2[PIUAS: 处理C4];
        G --> H3[大规模预训练+微调: 缓解C2, 提升泛化];
        G --> H4[针对机器人数据特性的架构改进];
    ```

### 方法

本文提出的 Robotics Diffusion Transformer (RDT) 是一个专为双臂操作设计的语言条件视觉运动策略。其核心思想是利用扩散模型强大的分布表示能力来捕捉双臂操作中固有的多模态动作，并结合 Transformer 的可扩展性处理异构输入。

1.  **RDT 模型 (Diffusion Modeling for Robotics)：**
    *   **目标：** 学习条件概率分布 $p(a_t | l, o_t)$，其中 $a_t$ 是动作，$l$ 是语言指令，$o_t$ 是观测。
    *   **扩散过程：** 与标准扩散模型类似，通过 K 步去噪过程从纯噪声动作 $a_t^K$ 恢复到干净动作 $a_t^0$。
    *   **网络 $f_θ$：** 学习一个去噪网络 $f_θ(l, o_t, a_t^k, k)$ 来预测干净动作，通过最小化 MSE 损失进行训练：$L(θ) := MSE (a_t, f_θ(l, o_t, sqrt(ᾱ_k)a_t + sqrt(1-ᾱ_k)ε, k))$。
    *   **动作分块 (Action Chunking)：** 实际中预测一个动作序列 $a_t:t+Ta$ 以增强时间一致性并减少误差累积。
    *   **针对机器人数据的架构调整：**
        *   **异构多模态输入编码：**
            *   **低维输入 (Low-Dimensional Inputs)：** 本体感知 $z_t$、噪声动作块 $ã_t:t+Ta$、控制频率 $c$、扩散时间步 $k$。使用带傅里叶特征的 MLP 进行编码。$z_t$ 和 $ã_t:t+Ta$ 先嵌入到 PIUAS。
            *   **图像输入 (Image Inputs)：** 历史图像 $X_t-Timg+1:t+1$ (Timg=2, 包含外部、左右腕部相机图像)。使用固定的 SigLIP 编码器，并引入多维位置编码和随机掩码。
            *   **语言输入 (Language Inputs)：** 语言指令 $l$。使用固定的 T5-XXL 编码器。
        *   **核心网络 $f_θ$ 结构 (基于 Diffusion Transformer - DiT)：**
            *   **QKNorm & RMSNorm：** 替换 DiT 中的 LayerNorm，以增强数值稳定性并更好地处理时间序列特性。
            *   **MLP 解码器 (MLP Decoder)：** 用非线性 MLP 替换 DiT 末端的线性解码器，以更好拟合非线性机器人动作。
            *   **交替条件注入 (Alternating Condition Injection - ACI)：** 在 DiT 块的交叉注意力层中，交替注入图像和文本 token，而非同时注入，以避免信息量大的图像 token 压制文本信息，从而提升指令跟随能力。

2.  **数据策略：**
    *   **物理可解释的统一动作空间 (Physically Interpretable Unified Action Space - PIUAS)：**
        *   为解决跨机器人数据异构性问题，设计了一个统一的动作空间 (128维，如 Fig. 3 左侧所示，Table 4 详细描述)。
        *   不同机器人的本体感知和动作向量根据其物理意义映射到该统一空间中的对应维度，其余维度填充。
        *   这使得模型能从不同机器人数据中学习共享的物理规律。
    *   **预训练 (Pre-Training)：**
        *   **数据集：** 收集了包含 46 个不同机器人数据集的迄今最大的机器人操作预训练数据集 (1M+ 轨迹，21TB)。对数据进行了清洗和预处理。
        *   **目标：** 从大规模、多样化的数据中学习通用的物理先验和表征。
    *   **微调 (Fine-Tuning)：**
        *   **数据集：** 在 ALOHA 双臂机器人上自建了一个全面的多任务双臂操作数据集 (6K+ 轨迹，300+ 任务，100+ 物体，15+ 场景)。使用 GPT-4-Turbo 增强指令多样性。
        *   **目标：** 进一步提升模型在目标机器人上的双臂操作能力，适应特定 embodiment。

*   **Mermaid 流程图：**
    ```mermaid
    graph TD
        A_Inputs[输入数据] --> B_Encoders[异构多模态编码器];
        A_Lang[语言指令 l] --> Enc_Lang[T5-XXL Encoder + MLP];
        A_Img[图像历史 X_t-1:t+1] --> Enc_Img[SigLIP Encoder + MLP + Multi-Dim Pos. Emb.];
        A_Prop[本体感知 z_t] --> Emb_Prop_PIUAS[嵌入 PIUAS];
        A_NoiseAct[噪声动作块 ã_t:t+Ta] --> Emb_Act_PIUAS[嵌入 PIUAS];
        A_Freq[控制频率 c] --> Enc_Freq[MLP];
        A_TimeStep[扩散时间步 k] --> Enc_TS[MLP];

        Emb_Prop_PIUAS --> Enc_Prop[MLP + Fourier Feat.];
        Emb_Act_PIUAS --> Enc_Act[MLP + Fourier Feat.];

        B_Encoders --> C_Tokens[生成各类 Tokens];
        Enc_Lang --> Lang_Tokens;
        Enc_Img --> Img_Tokens;
        Enc_Prop --> Prop_Tokens;
        Enc_Act --> NoiseAct_Tokens;
        Enc_Freq --> Freq_Token;
        Enc_TS --> TS_Token;

        Prop_Tokens --> LowDim_Input_Tokens[低维输入序列];
        NoiseAct_Tokens --> LowDim_Input_Tokens;
        Freq_Token --> LowDim_Input_Tokens;
        TS_Token --> LowDim_Input_Tokens;

        LowDim_Input_Tokens --> DIT_Blocks[DiT Blocks (L层)];
        Lang_Tokens -->|交替条件注入 (ACI)| DIT_Blocks;
        Img_Tokens -->|交替条件注入 (ACI)| DIT_Blocks;
        DIT_Blocks --QKNorm, RMSNorm--> Processed_Latent;
        Processed_Latent --> MLP_Decoder[MLP 解码器];
        MLP_Decoder --> Output[去噪后动作块 a_t:t+Ta];

        subgraph Preprocessing
            A_Inputs;
        end
        subgraph EncodersAndTokens
            B_Encoders;
            C_Tokens;
        end
        subgraph RDT_Core
            DIT_Blocks;
            Processed_Latent;
            MLP_Decoder;
        end
        subgraph OutputProcessing
            Output;
        end
    ```

### 实验与结论

*   **实验设置：**
    *   **任务：** 选择了 7 个具有挑战性的任务来评估 RDT 的泛化性和能力，包括：`Wash Cup` (未见物体)，`Pour Water` (未见场景)，`Pour Water-L-1/3 & -R-2/3` (指令跟随)，`Handover` (5-shot 学习)，`Fold Shorts` (1-shot 学习)，`Robot Dog` (灵巧操作)。
    *   **数据：** 使用第 4.2 节描述的预训练和微调数据集。
    *   **模型训练与推理：** RDT-1B (1.2B 参数) 在 48 个 H100 GPU 上预训练 1M步，微调 130K步。推理时采用 DPM-Solver++ 将扩散步数从 100 降至 5，实现 6Hz 的动作块推理频率。
    *   **对比基线：** ACT (VAE-based), OpenVLA (Transformer-based, discretization), Octo (Diffusion-based, 93M)。
    *   **评估指标：** 主要使用成功率。
    *   **硬件：** 所有实验在 ALOHA 双臂机器人上进行。

*   **实验结果与分析 (回答 Q1-Q5)：**
    *   **Q1 & Q2 (零样本泛化能力)：** RDT 在 `Wash Cup` (未见杯子) 和 `Pour Water` (未见房间) 任务中对未见物体和场景展现出强大的零样本泛化能力。在 `Pour Water-L-1/3` 和 `Pour Water-R-2/3` 任务中，RDT 能够准确理解并执行包含 "one-third" 或 "two-thirds" 等未在训练指令中见过的特定量词和手部指定的指令，性能远超基线。
    *   **Q3 (少样本学习能力)：** 在 `Handover` (5-shot) 和 `Fold Shorts` (1-shot) 任务中，RDT 仅需极少量演示就能学习全新的复杂技能，其动作模式与已知技能差异很大，而其他基线几乎无法完成。这得益于大规模预训练提供的先验知识。
    *   **Q4 (灵巧操作能力)：** 在 `Robot Dog` 任务中，RDT 能够精确控制推动操作杆的角度，实现对机器狗的直线行走控制。这表明 RDT 的扩散模型和强大网络架构能够精确建模多模态和非线性动作，满足灵巧任务的精度要求。
    *   **Q5 (模型设计要素的重要性 - 消融实验)：**
        *   **扩散建模 vs. 回归：** RDT (regress) 性能显著下降，证明扩散建模对表示多模态动作至关重要。
        *   **大模型 vs. 小模型：** RDT (small, 166M) 性能下降，说明大参数量带来的先验知识对泛化性很重要。
        *   **预训练 vs. 从头训练：** RDT (scratch) 在未见物体和场景上的表现非常差，突显了大规模预训练在学习泛化知识方面的关键作用。
        *   **架构改进 (QKNorm/RMSNorm, MLP Decoder, ACI)：** Fig. 4 显示，移除这些模块会导致训练不稳定或在特定任务上性能显著下降。

*   **结论：**
    论文成功地通过 RDT 模型解决了双臂操作中数据稀缺和操作复杂性带来的挑战。RDT 利用扩散模型处理多模态动作，通过 PIUAS 统一异构机器人数据进行大规模预训练，并在自建的高质量双臂数据集上微调。实验证明 RDT 在灵巧双臂操作、指令跟随、少样本学习以及对未见物体和场景的零样本泛化方面均取得了最先进的性能。

### 不足

论文本身未明确列出不足之处，但根据内容可以推测：
1.  **计算资源需求大：** 训练 1.2B 参数的模型需要大量的计算资源 (48 H100 GPUs)，这可能限制了其在资源受限环境下的复现和进一步研究。
2.  **对预训练编码器的依赖：** 模型依赖固定的、大规模预训练的视觉 (SigLIP) 和语言 (T5-XXL) 编码器，模型的性能可能受限于这些编码器的能力。
3.  **PIUAS 的普适性：** PIUAS 目前主要针对带夹爪的机器人手臂设计，对于其他类型的末端执行器（如吸盘、多指手）可能需要调整或重新设计。
4.  **对高质量演示数据的依赖：** 尽管通过预训练缓解了数据稀缺，但微调阶段仍依赖高质量的人类演示数据。探索更少依赖演示或能从次优数据中学习的方法是未来的方向。
5.  **推理速度：** 虽然通过 DPM-Solver++ 优化，6Hz 的动作块推理频率对于某些实时性要求极高的任务可能仍有提升空间。
6.  **任务范围：** 尽管覆盖了多种任务，但真实世界的复杂性是无穷的，模型的泛化能力仍有待在更广泛、更动态的环境中验证。


## Ref and Tag