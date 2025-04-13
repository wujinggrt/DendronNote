---
id: 24lm18im5j9eeujuegtyci8
title: ASAP
desc: ''
updated: 1744553009499
created: 1743090219279
---

## 论文总结

### 作者、团队信息、论文标题、论文链接、项目主页

* **作者**: Tairan He, Jiawei Gao, Wenli Xiao, Yuanhang Zhang, Zhengyi Luo, Guanqi He, Nikhil Sobanbab, Chaoyi Pan, Kris Kitani, Zi Wang, Zeji Yi, Jiashun Wang, Guannan Qu, Guanya Shi, Jessica Hodgins, Linxi "Jim" Fan, Yuke Zhu, Changliu Liu. (+ 表示贡献相同)
* **团队信息**: Carnegie Mellon University (1), NVIDIA (2)
* **论文标题**: ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills
* **论文链接**: arXiv:2502.01143v2 [cs.RO] (发布于 2025 年 2 月 8 日)
* **项目主页**: [https://agile.human2humanoid.com](https://agile.human2humanoid.com)
* **代码链接**: [https://github.com/LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP)

### 主要贡献

1.  **提出 ASAP 框架**: 引入了一个名为 ASAP 的框架，通过使用真实世界数据训练的 delta (残差) 动作模型，利用强化学习 (RL) 来弥合模拟到现实 (sim-to-real) 的差距。 [source: 50]
2.  **实现高难度动作**: 成功地在真实世界中部署了基于 RL 的全身控制策略，实现了以前难以完成的类人机器人敏捷全身动作，例如 Cristiano Ronaldo 的跳跃旋转庆祝动作、LeBron James 的单腿平衡庆祝动作以及 Kobe Bryant 的后仰跳投等。 [source: 1, 2, 51]
3.  **验证有效性**: 在模拟和真实世界环境中进行了广泛的实验，证明 ASAP 有效地减少了动力学不匹配，使机器人能够执行高度敏捷的动作，并显著减少了运动跟踪误差 (在 sim-to-real 任务中高达 52.7%)。 [source: 16, 52, 254]
4.  **开源代码库**: 开发并开源了一个多模拟器训练和评估代码库，以促进该领域的进一步研究。 [source: 53]

### 研究背景（研究问题，研究难点和相关工作）

* **研究问题**: 如何让类人机器人在真实世界中实现像人类一样敏捷、协调的全身技能。 [source: 4, 22]
* **研究难点**:
    * **Sim-to-Real Gap**: 模拟环境的动力学与真实世界物理之间存在显著的不匹配 (dynamics mismatch)，这是阻碍模拟训练策略直接应用于真实机器人的主要障碍。 [source: 4, 22]
    * **现有方法局限**:
        * 系统辨识 (System Identification, SysID) 方法需要预先定义参数空间，可能无法完全捕捉 sim-to-real 差距，且常依赖难以获取的地面真实扭矩测量。 [source: 5, 24, 25, 26]
        * 域随机化 (Domain Randomization, DR) 方法虽然能提升鲁棒性，但可能导致策略过于保守，牺牲了敏捷性。 [source: 5, 27, 28, 29]
        * 学习动力学模型的方法在无人机等低维系统上取得成功，但在高维、复杂的类人机器人系统上的有效性仍有待探索。 [source: 30, 31]
    * **硬件限制**: 类人机器人的硬件本身也存在局限性。 [source: 22]
* **相关工作**:
    * **基于学习的类人机器人控制**: 利用 RL 在模拟器中学习运动技能，如行走、跳跃、跑酷、跳舞、操作等。 [source: 233, 234, 235]
    * **物理模拟角色动画**: 在物理模拟中实现富有表现力和敏捷性的全身运动。 [source: 236]
    * **系统辨识 (SysID)**: 包括离线和在线方法，用于校准模拟器或机器人模型参数。 [source: 239, 240, 241, 242, 243, 244, 245]
    * **残差学习 (Residual Learning)**: 在机器人学中用于改进控制器、修正动力学模型或建模残差轨迹。 [source: 247, 248, 249, 250]

### 方法

ASAP 是一个两阶段框架，旨在对齐模拟与真实世界的动力学，以学习敏捷的类人机器人全身技能。 [source: 6, 32]

* **第一阶段：预训练 (Pre-training)** [source: 7, 33]
    1.  **数据生成**:
        * 从人类视频中捕捉运动数据 (例如使用 TRAM 将视频转为 SMPL 格式)。 [source: 63, 64]
        * 进行基于模拟的数据清洗 (例如使用 MaskedMimic 在 IsaacGym 中验证 SMPL 动作的物理可行性)。 [source: 55]
        * 将清洗后的 SMPL 动作重定向 (Retargeting) 到目标类人机器人模型 (例如 Unitree G1)。 [source: 56, 57, 60]
    2.  **运动跟踪策略训练**:
        * 在模拟器 (如 IsaacGym) 中，使用强化学习 (PPO 算法) 训练一个相位条件 (phase-conditioned) 的运动跟踪策略，使其模仿重定向后的机器人参考动作。 [source: 34, 35, 61, 69]
        * **关键技术**:
            * **非对称 Actor-Critic**: Critic 网络使用特权信息 (如参考运动的全局位置、根部线速度)，而 Actor 网络仅依赖本体感知信息和时间相位，以弥合模拟与现实的可观察性差异，且无需里程计。 [source: 71, 72, 73]
            * **终止课程 (Termination Curriculum)**: 训练过程中逐步收紧允许的运动跟踪误差容忍度 (从 1.5m 到 0.3m)，引导策略从学习基本平衡到精确跟踪高动态行为。 [source: 76, 77, 78, 79]
            * **参考状态初始化 (Reference State Initialization, RSI)**: 随机初始化训练回合在参考动作的不同时间点开始，使策略能并行学习动作的不同阶段，而非严格顺序学习。 [source: 84, 85, 86, 87]
        * 使用基础的域随机化技术增强策略鲁棒性。 [source: 91, 416]

* **第二阶段：后训练 (Post-training)** [source: 8, 33]
    1.  **真实数据收集**: 将预训练的策略部署到真实机器人 (或目标模拟器) 上执行任务，并收集真实轨迹数据 ($D^r = \{s_0^r, a_0^r, ..., s_T^r, a_T^r\}$)，包括运动捕捉数据和板载传感器数据。 [source: 37, 94, 95]
    2.  **Delta Action 模型训练**:
        * 将收集到的真实动作序列 ($a_t^r$) 在源模拟器中重放。由于 sim-to-real gap，模拟状态会偏离真实状态。 [source: 96]
        * 训练一个 Delta Action 模型 ($\pi_{\theta}^{\Delta}$)：输入当前模拟状态 $s_t$ 和真实动作 $a_t^r$，输出一个修正动作 $\Delta a_t = \pi_{\theta}^{\Delta}(s_t, a_t^r)$ (论文中图示和公式略有出入， Figure 2(b) 和公式 $s_{t+1} = f^{sim}(s_t, a_t^r + \Delta a_t)$ 显示输入 $a_t^r$，而 Section III-B 文字描述为 $\Delta a_t = \pi_{\theta}^{\Delta}(s_t, a_t)$，其中 $a_t$ 是策略动作)。 [source: 102, 104]
        * 目标是使模拟器应用修正后的动作 ($a_t^r + \Delta a_t$) 产生的下一状态 $s_{t+1}$ 尽可能接近真实的下一状态 $s_{t+1}^r$。 [source: 104, 108]
        * 使用 RL (PPO) 进行训练，奖励函数旨在最小化模拟状态与真实状态的差异，并加入动作幅度正则化项。 [source: 109, 105, 106]
        * 这个 Delta Action 模型学习补偿动力学差异。 [source: 13, 45, 101]
    3.  **策略微调 (Fine-tuning)**:
        * 将学习到的 Delta Action 模型 ($\pi^{\Delta}$) *冻结* 并整合到模拟器中，创建一个与真实世界动力学更对齐的模拟环境 ($s_{t+1} = f^{ASAP}(s_t, a_t) = f^{sim}(s_t, a_t + \pi^{\Delta}(s_t, a_t))$ )。 [source: 14, 47, 114]
        * 在这个增强的模拟器中，使用与预训练相同的奖励函数 (Table I) 微调第一阶段训练的运动跟踪策略。 [source: 114]

* **部署**: 将微调后的策略直接部署到真实世界机器人上，*此时不再使用 Delta Action 模型*。 [source: 12, 115]

### 实验与结论

* **实验设置**:
    * **任务**: 评估 ASAP 在多种敏捷全身运动跟踪任务上的性能，包括跳跃、单腿平衡、踢腿等。动作按难度分为 Easy, Medium, Hard 三级。 [source: 121, 138]
    * **环境**:
        * Sim-to-Sim: IsaacGym (训练) -> IsaacSim (测试), IsaacGym (训练) -> Genesis (测试)。 [source: 15, 116, 122]
        * Sim-to-Real: IsaacGym (训练) -> 真实 Unitree G1 类人机器人 (测试)。 [source: 15, 116, 125]
    * **基线方法**: Oracle (仅在 IsaacGym 中训练和测试), Vanilla (直接迁移预训练策略), SysID (辨识物理参数后微调), DeltaDynamics (学习残差动力学模型后微调)。 [source: 127, 128, 129, 131]
    * **评估指标**: 成功率 (Success Rate), 全局身体位置跟踪误差 ($E_{g-mpjpe}$), 均方根关节位置误差 ($E_{mpjpe}$), 加速度误差 ($E_{acc}$), 根速度误差 ($E_{vel}$)。 [source: 132, 133, 135]
* **实验结果**:
    * **Q1 (动力学匹配能力)**: 在开环重放实验中 (将测试环境轨迹在训练环境重放)，ASAP (Delta Action) 比 OpenLoop、SysID 和 DeltaDynamics 能更准确地复现测试环境轨迹，尤其在长时域内表现更优，表明其能更好地补偿动力学不匹配。 [source: 135, 141, 142, 143, 144, 145]
    * **Q2 (策略微调性能)**: 在闭环运动模仿任务中，经过 ASAP 微调的策略在 IsaacSim 和 Genesis 上的性能显著优于 Vanilla、SysID 和 DeltaDynamics，在所有难度级别上均取得更低的跟踪误差和更高的成功率。 [source: 118, 146, 148, 149, 150, 151, 152]
    * **Q3 (Sim-to-Real 迁移)**: ASAP 成功应用于真实的 Unitree G1 机器人。由于真实数据收集的挑战 (硬件损坏、耗时)，实验中训练了一个 4-DoF 的脚踝 Delta Action 模型。结果表明，ASAP 微调后的策略在真实世界中的跟踪性能优于 Vanilla 基线，无论对于训练 Delta Action 模型时使用的动作 (in-distribution) 还是未使用的动作 (out-of-distribution，如 LeBron James "Silencer")，都降低了各项跟踪误差。 [source: 157, 158, 159, 160, 161, 162, 163, 176, 177]
    * **Q4 & Q5 (Delta Action 模型训练与使用)**: 分析了数据集大小、训练时域长度、动作范数权重对 Delta Action 模型性能的影响。比较了不同使用 Delta Action 模型的方法 (固定点迭代、梯度优化、RL 微调)，发现 RL 微调效果最好。 [source: 178, 179, 180, 183, 184, 185, 186, 187, 188, 189, 190, 203, 209, 210]
    * **Q6 (ASAP 工作原理)**: 验证了 ASAP 微调优于简单的随机动作噪声微调。可视化 Delta Action 模型的输出，发现其学习到了结构化的动力学差异 (例如，下肢比上肢差异大，脚踝关节差异最显著，左右不对称)，这是均匀噪声无法捕捉的。 [source: 213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 226]
* **结论**: ASAP 框架通过学习 Delta Action 模型来捕捉和补偿模拟与现实之间的动力学不匹配，成功地将模拟环境中训练的策略迁移到真实世界的类人机器人上，实现了多样化且敏捷的全身技能，显著降低了运动跟踪误差。该工作为敏捷全身控制的 sim-to-real 迁移提供了有前景的方向。 [source: 17, 18, 252, 253, 254, 255]

### 不足

论文在 Section VIII 中指出了该框架的局限性：

* **硬件约束**: 执行敏捷全身运动对机器人硬件产生巨大压力，易导致电机过热甚至硬件损坏 (实验中损坏了两台 Unitree G1)，限制了可安全收集的真实世界数据的规模和多样性。 [source: 256, 257, 258]
* **依赖运动捕捉系统 (MoCap)**: 当前流程需要 MoCap 系统来记录真实世界轨迹，这限制了其在缺乏 MoCap 设备的非结构化环境中的实际部署。 [source: 259, 260]
* **Delta Action 训练的数据需求**: 尽管将 Delta Action 模型简化为 4-DoF 脚踝关节提高了样本效率，但在真实世界中训练完整的 23-DoF 模型仍然不切实际，因为它需要大量的运动片段数据 (例如模拟中需要 > 400 个片段)。 [source: 261]
* **未来方向**: 开发能感知损伤的策略以降低硬件风险；利用无 MoCap 的对齐方法；探索 Delta Action 模型的自适应技术以实现小样本对齐。 [source: 262]

## Ref and Tag

