---
id: oq6p9fjfcqnftunqdmtzob0
title: GeminiRobotics_VLA
desc: ''
updated: 1745636852277
created: 1741838502005
---

Gemini 具有多模态推理能力，能够处理文字、视觉和语音。

## 论文总结

### 作者、团队信息、论文标题、论文链接、项目主页

* **作者**: Gemini Robotics Team [cite: 13]
* **团队信息**: Google DeepMind [cite: 1]
* **论文标题**: Gemini Robotics: Bringing AI into the Physical World [cite: 1]
* **论文链接**: 论文中未直接提供 URL 链接，但提供了通讯邮箱 gemini-robotics-report@google.com [cite: 13]。
* **项目主页**: 论文中未明确提及项目主页。

### 主要贡献

* **Gemini Robotics 模型**: 提出了一种先进的视觉-语言-动作 (Vision-Language-Action, VLA) 通用模型 Gemini Robotics，能够直接控制机器人 [cite: 4]。该模型能执行平滑且反应灵敏的动作来处理复杂的操纵任务，对物体类型和位置的变化具有鲁棒性，能处理未见过的环境，并遵循多样化的开放词汇指令 [cite: 5]。
* **Gemini Robotics-ER 模型**: 引入了 Gemini Robotics-ER (Embodied Reasoning) 模型，该模型扩展了 Gemini 的多模态推理能力到物理世界，增强了空间和时间理解能力 [cite: 7, 8]。这使得模型具备了与机器人相关的能力，如物体检测、指向、轨迹和抓取预测，以及以多视图对应和 3D 边界框预测形式实现的 3D 理解 [cite: 9]。
* **模型特化与适应性**: 展示了通过额外的微调，Gemini Robotics 可以被特化以获得新能力，包括解决长时程、高灵巧性任务（如折纸狐狸或玩纸牌游戏），从少至 100 个演示中学习新的短时程任务，以及适应全新的机器人形态（包括双臂平台和高自由度人形机器人） [cite: 6]。
* **ERQA 基准**: 提出了 ERQA (Embodied Reasoning Question Answering)，一个开源基准，专门用于评估多模态模型的具身推理能力，解决了现有基准在评估超越原子能力的综合能力方面的不足 [cite: 41]。
* **负责任开发**: 讨论并解决了与此类新型机器人基础模型相关的重要安全考虑，并按照 Google AI 原则进行了负责任的开发 [cite: 11, 44]。

### 研究背景

* **研究问题**:
    * 如何将大型多模态模型（如 Gemini 2.0）在数字领域展现出的卓越通用能力，转化到物理智能体（如机器人）上，以创造出能够理解物理世界并与之进行有效、安全交互的通用机器人？ [cite: 1, 2, 26]
    * 具体来说，如何赋予最先进的数字 AI 模型所需的具身推理能力，以实现通用且灵巧的物理世界交互？ [cite: 26]

* **研究难点**:
    * **数字到物理的鸿沟**: 将数字能力转化为物理交互需要鲁棒的、人类水平的具身推理能力（例如，理解 3D 结构、物体间关系、直观物理学） [cite: 21, 22]。
    * **被动理解到主动交互**: 智能体不仅要被动理解空间和物理概念，还必须学会采取能直接影响外部环境的行动 [cite: 23, 24]。
    * **获取具身推理能力**: 需要模型能够理解物理世界丰富的几何和时空细节 [cite: 30]。
    * **物理世界的基础**: 必须将具身推理能力通过让模型“说出”物理动作的语言（理解接触物理、动力学和真实世界交互的复杂性）来扎根于物理世界 [cite: 31]。
    * **控制挑战**: 实现快速、安全、灵巧的机器人控制 [cite: 32]。
    * **评估挑战**: 缺乏能够评估在物理世界中行动所需的、超越原子能力的更广泛能力集的基准 [cite: 41, 61]。
    * **实时性挑战**: 大型模型（如 VLM）的推理延迟和硬件限制可能与实时机器人控制不兼容 [cite: 249, 250]。

* **相关工作**:

    | 领域研究                           | 已有方法                                                                                                                                                                                      | 局限性                                                                                                                                                                                                              | 本文改进                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
    | :--------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | 大型多模态模型 (LMMs/VLMs)         | 展示了在数字领域的通用能力 (文本、图像、音频、视频处理) [cite: 19]                                                                                                                            | 物理世界中的应用（机器人）仍是挑战 [cite: 1]                                                                                                                                                                        | 基于 Gemini 2.0 构建，专门为机器人设计，赋予物理世界理解和交互能力 [cite: 3, 28, 29]                                                                                                                                                                                                                                                                                                                                                                                       |
    | 机器人控制与具身智能 (Embodied AI) | 传统的特定任务机器人系统 [cite: 481]。需要组合多个模型来实现感知、规划、控制 [cite: 187]。基于扩散策略的多任务学习 [cite: 267]。代码生成策略 [cite: 190]。上下文学习 (ICL) [cite: 188, 210]。 | 缺乏通用性 [cite: 481]。组合模型复杂 [cite: 187]。现有 VLA 在抽象推理和泛化方面存在挑战 [cite: 367]。零样本/少样本控制能力有限 [cite: 189, 211]。实时控制延迟问题 [cite: 250]。泛化能力不足是部署瓶颈 [cite: 294]。 | 提出单一模型的 Gemini Robotics (VLA) 和 Gemini Robotics-ER (VLM) [cite: 3, 7]。Gemini Robotics-ER 统一感知、推理、规划能力 [cite: 187]，支持零样本（代码生成）和少样本（ICL）控制 [cite: 188]。Gemini Robotics 直接预测动作，实现低延迟（~250ms, 50Hz有效控制频率）灵巧控制 [cite: 4, 254, 255]。强大的泛化能力（视觉、指令、动作） [cite: 38, 242, 294, 312]。通过微调实现专业化（长时程、高灵巧性任务）和快速适应（新任务、新机器人形态） [cite: 6, 39, 333, 391, 398]。 |
    | 具身推理 (Embodied Reasoning)      | 现有 VLM 基准测试侧重于原子能力（如目标识别、计数） [cite: 61]                                                                                                                                | 缺乏评估物理世界交互所需更广泛能力的基准 [cite: 41, 61]                                                                                                                                                             | 提出 ERQA 基准 [cite: 41]。Gemini 2.0 和 Gemini Robotics-ER 在 ERQA 及其他空间理解基准上表现 SOTA [cite: 52, 74]。展示了 Gemini 2.0/ER 的多种 ER 能力（2D/3D 检测、指向、轨迹/抓取预测、多视图对应） [cite: 51, 108, 116]。                                                                                                                                                                                                                                                |
    | 机器人安全                         | 传统物理动作安全（避障、力控制等） [cite: 421]。内容安全策略 [cite: 437]。                                                                                                                    | 传统方法不足以应对 AI 驱动的通用机器人的语义动作安全挑战（开放域、非结构化环境中的复杂约束） [cite: 436, 443, 444, 445, 446]。                                                                                      | 继承 Gemini 内容安全策略 [cite: 438]。为新输出模态（如指向）增加内容安全层 [cite: 439, 442]。关注语义动作安全，使用 ASIMOV 数据集进行评估和改进 [cite: 448, 450]。采用 Constitutional AI 方法提升安全性 [cite: 453, 454]。结合安全关键的底层控制器 [cite: 423]。                                                                                                                                                                                                           |

    ```mermaid
    graph TD
        A[现有研究] --> B(大型多模态模型 LMMs/VLMs);
        A --> C(机器人控制与具身智能);
        A --> D(具身推理 Embodied Reasoning);
        A --> E(机器人安全);

        B --> B1{局限性: 物理世界应用挑战};
        C --> C1{局限性: 通用性差, 多模型复杂, 推理/泛化/实时性/延迟问题};
        D --> D1{局限性: 缺乏综合性评测基准};
        E --> E1{局限性: 传统方法难应对语义安全};

        B1 --> F[本文改进: Gemini Robotics 系列模型];
        C1 --> F;
        D1 --> F;
        E1 --> F;

        F --> G(Gemini Robotics-ER: 强具身推理 VLM);
        F --> H(Gemini Robotics: 通用灵巧 VLA);
        F --> I(ERQA 基准);
        F --> J(增强的安全框架);

        G --> K(支持零/少样本控制);
        H --> L(低延迟控制, 强泛化, 可特化/适应);
        I --> M(评估物理世界交互能力);
        J --> N(覆盖内容安全和语义动作安全);
    ```

### 方法

* **基础模型**: 基于 Gemini 2.0 多模态基础模型构建 [cite: 3, 33]。
* **Gemini Robotics-ER (VLM)**:
    * 通过针对具身推理任务（如 3D 感知、指向、状态估计、功能可见性预测）的特定训练来增强 Gemini 2.0，同时保留核心 VLM 能力 [cite: 7, 8, 35, 36]。
    * 使用新的 ERQA 基准等进行评估，展现了 SOTA 性能，尤其是在使用思维链 (Chain-of-Thought, CoT) 提示时 [cite: 34, 52, 74, 100, 102]。
    * 支持零样本（通过代码生成）和少样本（通过上下文学习 ICL）机器人控制 [cite: 10, 54, 186, 188]。零样本控制利用机器人 API 进行感知（使用 Gemini 自身）和动作执行 [cite: 192, 193, 195, 196]。少样本 ICL 则通过在上下文中提供演示轨迹（观察、动作、语言推理）来让模型直接生成末端执行器姿态 [cite: 212, 213, 221, 223, 224]。
* **Gemini Robotics (VLA)**:
    * 由 Gemini Robotics-ER 衍生而来，通过大规模机器人动作数据集（数千小时在 ALOHA 2 机器人上收集的、覆盖多样化任务的专家遥操作演示）以及非动作多模态数据（网页文档、代码、图像、视频、具身推理/视觉问答数据）进行微调 [cite: 4, 37, 239, 241, 258, 259, 260]。
    * 架构包含一个云端 VLA 主干（蒸馏版的 Gemini Robotics-ER，延迟 < 160ms）和一个运行在机器人板载计算机上的本地动作解码器（端到端延迟约 250ms，有效控制频率 50Hz） [cite: 234, 251, 252, 253, 254, 255]。
    * 直接输出动作块 (action chunks) [cite: 233, 239]。
* **特化与适应**: Gemini Robotics 的可选微调阶段：
    * **长时程灵巧性**: 针对特定的复杂任务（如折纸、打包午餐盒），使用小规模、高质量的数据集（2k-5k 条）进行微调 [cite: 333, 349]。
    * **增强推理**: 使用重新标记的动作数据集进行微调，将动作预测与具身推理能力（如轨迹理解和生成）联系起来，以提高在复杂 OOD 场景中的泛化能力 [cite: 366, 368, 369]。
    * **快速适应**: 使用少量数据（例如 100 个演示）对新短时程任务进行微调 [cite: 391, 393]。
    * **新机器人形态**: 使用少量数据对新的机器人平台（如 Franka 双臂机器人、Apollo 人形机器人）进行微调 [cite: 398, 405]。
* **安全框架**:
    * 遵循 Google AI 原则 [cite: 416]。
    * 继承了 Gemini 的内容安全策略 [cite: 438]。
    * 为新的输出模态（如指向）添加了额外的安全层（如偏见诱导指向查询的过滤） [cite: 439, 441, 442]。
    * 关注语义动作安全，使用 ASIMOV 数据集进行评估和改进，并采用 Constitutional AI 方法进行后训练和评估 [cite: 448, 450, 453]。
    * 设计为可与底层安全关键控制器接口 [cite: 423]。

* **方法流程图**:
    ```mermaid
    graph TD
        A[Gemini 2.0 Foundation Model] --> B(Robotics Specific Training);

        subgraph Gemini Robotics-ER (VLM for Embodied Reasoning)
            B --> C{Input: Images + Text Prompt};
            C --> D[Enhanced Embodied Reasoning: 2D/3D Detection, Pointing, Trajectory/Grasp Prediction, Multi-view Correspondence];
            D --> E{Output: Text (Coordinates, Code, Reasoning)};
            E --> F(ERQA Benchmark Evaluation);
            E --> G(Zero-shot Control via Code Gen);
            E --> H(Few-shot Control via ICL);
        end

        subgraph Gemini Robotics (VLA for Direct Control)
            I[Gemini Robotics-ER] --> J(Fine-tuning: Robot Action Data + Multimodal Data);
            J --> K[Cloud Backbone (Distilled ER)];
            J --> L[Local Action Decoder];
            M{Input: Images + Text Instruction} --> K;
            K --> L;
            L --> N{Output: Robot Action Chunks (Low Latency ~250ms)};
        end

        subgraph Specialization & Adaptation (Optional Fine-tuning)
            N --> O(Long-Horizon Dexterity);
            N --> P(Enhanced Reasoning & Generalization);
            N --> Q(Fast Task Adaptation);
            N --> R(New Embodiment Adaptation);
        end

        subgraph Safety Framework
            S[Google AI Principles] --> T(Content Safety Policies - Inherited);
            T --> U(New Modality Safety - Pointing Filter);
            S --> V(Semantic Action Safety - ASIMOV Datasets, Constitutional AI);
            V --> W(Post-training & Evaluation);
            S --> X(Interface with Low-Level Safety Controllers);
        end

        G --> Y[Robot Execution];
        H --> Y;
        N --> Y;
        O --> Y;
        P --> Y;
        Q --> Y;
        R --> Y;
        X --> Y;
    ```

### 实验与结论

* **Gemini Robotics-ER 表现**:
    * 在 ERQA、RealworldQA 和 BLINK 基准测试中取得 SOTA，展示了强大的具身推理能力 [cite: 74]。
    * 展示了包括 2D/3D 物体检测（在 SUN-RGBD 上达到 SOTA [cite: 179]）、指向（优于其他 VLM 和专门模型 [cite: 145, 146]）、轨迹/抓取预测和多视图对应在内的多种能力 [cite: 109, 110, 112, 114, 115, 116, 117, 171]。
    * 成功实现了对 ALOHA 2 机器人的零样本（代码生成，模拟任务平均成功率 53%）和少样本（ICL，模拟/真实任务平均成功率 65%）控制，并优于基础版 Gemini 2.0 [cite: 189, 203, 206, 217, 227]。
* **Gemini Robotics 表现**:
    * 开箱即用（无需特定任务微调）即可解决广泛的短时程灵巧任务，显著优于基线模型（𝜋0 re-implement, multi-task diffusion） [cite: 242, 265, 271, 283, 285]。
    * 展示了精确遵循自然语言指令的能力，尤其是在包含未见物体的新颖场景中 [cite: 287, 290, 292]。
    * 展示了在视觉、指令（包括释义、拼写错误、不同语言）和动作（OOD 位置、新物体实例）三个维度的强大泛化能力，持续优于基线模型 [cite: 294, 306, 312, 320, 321]。
* **特化与适应表现**:
    * **长时程任务**: 特化后的模型在复杂的长时程任务上取得了高成功率（例如，打包午餐盒任务 100%，六个任务平均 79%），显著优于特化的基线模型和从零开始训练的模型，证明了预训练通用模型和多样化数据的重要性 [cite: 333, 350, 353, 354, 360, 363, 364, 365]。
    * **增强推理**: 推理增强版本在需要推理和语义/空间理解的 OOD 任务上的表现显著优于普通版 Gemini Robotics 模型 [cite: 372, 387]。
    * **快速适应**: 仅用少量演示（≤100 个）即可实现快速任务适应（在 7/8 的任务上成功率超过 70%），并在较难任务上优于基线模型 [cite: 391, 393, 396, 401]。
    * **新机器人形态**: 成功地用少量数据将模型适应到新的机器人形态（Franka 双臂、Apollo 人形），在分布内任务上达到或超过 SOTA 单任务扩散策略的性能，并在泛化测试中表现显著更优 [cite: 398, 405, 406, 407, 414, 415]。
* **安全性表现**:
    * 通过 ASIMOV 基准测试展示了对物理安全的强大语义理解能力，该能力可通过 Constitutional AI 方法得到增强，并对对抗性提示具有鲁棒性 [cite: 450, 451, 452, 453, 454]。
    * 经过安全微调后，对诱导偏见的指向查询显示出高拒绝率 [cite: 442]。
* **结论**:
    * Gemini Robotics 系列模型（ER 和 VLA）通过有效地将像 Gemini 2.0 这样的大模型的推理能力带入物理世界，标志着朝着通用机器人迈出了实质性的一步 [cite: 12, 45, 479]。
    * 利用基于真实世界交互数据接地的互联网规模数据的推理能力，使机器人能够理解物理世界并有效行动 [cite: 46]。
    * 该方法实现了灵巧控制、强大的泛化能力以及高效的适应和特化 [cite: 466, 467, 468, 471]。

### 不足

* **Gemini 2.0/ER**:
    * 可能难以在长视频中准确地进行空间关系定位 [cite: 473]。
    * 数值预测（点、框）的精度可能不足以满足更精细的机器人控制任务 [cite: 473]。
* **Gemini Robotics**:
    * 在处理需要多步推理和精确灵巧运动相结合的复杂场景（尤其是在新颖情境下）方面仍需改进 [cite: 475]。
    * 将抽象推理与精确执行无缝结合的技术有待进一步开发 [cite: 476]。
* **未来工作**:
    * 加强推理与灵巧性结合的能力 [cite: 475, 476]。
    * 更多地利用模拟环境生成多样化、富含接触的数据，并开发相应的模拟到现实迁移技术 [cite: 477]。
    * 扩展多机器人形态实验，目标是减少适应新机器人类型所需的数据量，最终实现零样本跨机器人形态迁移 [cite: 478]。
    * 持续改进安全性和对齐性 [cite: 457, 487]。
    * 考虑更广泛的社会影响 [cite: 458, 485]。

## Ref and Tag

[Blog](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)
[Report PDF](https://storage.googleapis.com/deepmind-media/gemini-robotics/gemini_robotics_report.pdf)
[gemini robotics](https://deepmind.google/technologies/gemini-robotics/)

#VLA