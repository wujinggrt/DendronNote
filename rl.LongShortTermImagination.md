---
id: qon8owvfwcckh0l0fl9e8ee
title: LongShortTermImagination
desc: ''
updated: 1743958670926
created: 1739694710359
---


ICLR oral: Open-World Reinforcement Learning over Long Short-Term Imagination. https://qiwang067.github.io/ls-imagine, https://arxiv.org/pdf/2410.03618

insights: 做 US 时候，潜意识里人是能够估计人体的 affordance 图，知道大致范围。是否可以弄个 WorldModel？LLM 只能生成下一个 token，当做 short，嵌入另一个小模型来做 long term imagination。

训练 visual RL agent 时，model-based methods 通常是 short-sighted。作者提出 LS-Imagine，根据有限数量的状态转移步，扩展了 imagination horizon,，使得 agent 去探索行为，以到可能的长期反馈 (long-term feedback)。核心在于建立长短期世界模型 (long short-term world model)。作者模拟了以目标为条件的 jumpy 状态转移，并通过放大单张图像的具体区域来计算对应的 affordance map。

RL 中，instant state transitions（瞬时状态转移） 和 jumpy state transitions（跳跃状态转移） 是描述环境动态特性的两种现象。
* Instant State Transitions。智能体执行动作后，环境状态立即发生变化，且这种变化在时间上是**离散的**或**瞬间完成**的。无中间过程和中间状态。这是大多数强化学习问题的默认假设。比如棋类游戏，做出动作立刻更新棋盘。
* Jumpy State Transitions。状态变化在时间或空间上表现出不连续性或突变性。这种转移可能是由于环境动态的复杂性、传感器噪声或数据采样率不足引起的。不连续性：状态转移可能跳过某些中间状态，导致智能体观察到“跳跃”现象。时间或空间突变：状态可能在短时间内发生剧烈变化。挑战性：增加了智能体学习环境动态的难度。

最开始，需要一些成功的数据启动来学习 (是否需要模仿学习一部分，然后再强化学习探索式的学习增强泛化)，提供一些好的例子。就像 OS 的 Boostrap，需要嵌入一个启动 OS 的引导程序，再来启动 OS 以控制硬件。这也和 DeepSeek R1，基于强大的模型 V3 才能成功。就像鸡生蛋蛋生鸡问题，先加入一组数据，驱动这个循环启动就可以。

具体做法：先从一张图片，不断 Zoom in 得到一系列图片，作为成功的例子。

### 3.1 Overview
#### 3.2 AFFORDANCE MAP AND INTRINSIC REWARD
根据视觉观察和文本任务定义，引导 agent 的注意力**聚焦**于任务相关的区域。Affordance map 突出了区域与任务的关联性。

Affordance map computation via virtual exploration. 使用滑动边框扫描各个图片，不断在边框内放大，取出这些图片作为伪视频帧，以此对应长距离状态转移。根据文本描述的具体任务的目标，使用 MineCLIP 的奖励模型，评估视频切片与任务的相关性，生成 affordance map，用作探索的先验知识。

![fig2](assets/images/rl.LongShortTermImagination/fig2.png)

如 Fig 2(a)，模拟和评估探索，不依赖成功的轨迹。使用缩小了 15% 的宽和高的滑动边框，从左至右、从上至下的遍历当前观察的图像，覆盖了可能的区域。滑动边框朝着水平和垂直方向移动 9 步。对每个位置，不断放大感兴趣区域，得到 16 个如此图片，它们是边框标注的，经过调整尺寸到原来图像大小后，用来模拟 agent 向着目的位置移动的视觉转移，

首先，采用一个随机的 agent 与任务相关的环境交互，进一步收集数据。在观察的时间步 $t$ 得到的观察 $o_t$。使用一个滑动边框，其维度为观察图像的 15% 的宽和高，从左至右，从上至下地遍历整个观察图像。滑动边框水平和垂直地移动 9 步，在每个维度都覆盖每个潜在区域 (potential region)。作者每个观察 $o_t$ 的滑动边框中都囊括的位置，裁剪出 16 张图像。这些图像缩小了视场角，聚焦了区域，随后对这些图像重新调整到与观察图像大小的维度。如图 Fig 2(a) 上面部分，16 个 frames 模拟了探索。这 16 张有序的图像，用来模拟 agent 向着滑动边框确定的目的位置移动的视觉转移。重新调整的图像记为 $x_t^k (0 <= k < 16)$。使用 MineCLIP 模型计算这些图片集合 (视频帧) 与任务文本描述的相关性。随后量化了边框的 affordance value，最后得到潜在探索的感兴趣区域。Affordance value 根据感兴趣区域的边框数量求出。

#### 3.3 Long Short-Term World Model

定义世界模型为 long-term and short-term state transitions

## Reading

现有基于模型的RL方法（如 DreamerV3）因短期想象力（15步）受限，导致探索效率低下，难以处理长时程回报任务。LS-Imagination 扩展 imagination horizon，预测限定步数内的状态转移。作者构建了一个长短期世界模型。为了实现，作者使用以目标为条件的跳跃状态转移，计算对应的 affordance map。

LS-Imagination 使得世界模型高效学习并模拟特定行为的长期影响，并且不需要重复地 rollout step。模型学习后，能够预测即时状态转移（不跳步），也可以预测 jumpy state transition (可以跳步)。

Jumpy state transition: 允许 agent 绕过中间状态，使用一步来直接模拟一个任务相关的未来状态 $s_{t+H}$。此概念通常出现在世界模型。

![fig1_framework](assets/images/rl.LongShortTermImagination/fig1_framework.png)

如图，右侧蓝色箭头是预测的即时状态转移，红色箭头预测长期的状态转移。

| 数据类别       | 符号 | 描述                                   | 示例值或生成方法                 |
| -------------- | ---- | -------------------------------------- | -------------------------------- |
| 视觉观测       | `oₜ` | 64×64像素图像                          | Minecraft第一人称视角截图        |
| Affordance Map | `Mₜ` | 任务相关区域的热力图                   | U-Net预测或MineCLIP生成          |
| 跳转标志       | `jₜ` | 是否触发长时程想象                     | `True`若`P_{jump} >动态阈值`     |
| 跳转间隔       | `Δₜ` | 预测到目标的步数                       | 从真实交互数据匹配               |
| 复合奖励       | `rₜ` | $rₜ^{env} + rₜ^{MineCLIP} + αrₜ^{int}$ | α=1，高斯矩阵参数(σₓ=128, σᵧ=80) |

## 核心思路
- **关键思想**：  
  - **长短期世界模型**：融合短期单步转移和长期跳转状态模拟（如从当前状态直接预测接近目标的未来状态）。  
  - **Affordance Maps**：通过图像缩放和 MineCLIP 生成任务相关的空间先验，指导探索方向。  
  - **混合想象力训练**：在策略优化中联合使用长短期想象序列，直接估计长时程价值。  

![fig2](assets/images/rl.LongShortTermImagination/fig2.png)

### AFFORDANCE MAP AND INTRINSIC REWARD

核心思想是引导 agent 的注意力到视觉观察中，与任务相关的区域，提高探索效率。$\mathcal{M}_{o_{t,I}}(w,h)$ 代表 affordance map 中的值，代表观测图像位置 (w,h) 处与任务描述 $I$ (比如 "cut a tree") 的相关性。

#### 虚拟探索（virtual exploration）

- ​**滑动窗口裁剪**：在单张观测图像上，用尺寸为图像宽高15%的滑动边界框遍历所有区域（水平和垂直各分9步）。  
- ​**连续缩放生成伪视频**：对每个滑动窗口内的区域进行 16 次连续缩放（缩小视野模拟接近目标），生成伪视频帧序列 $\mathcal{X}_t = [x_t^0, x_t^1, \ldots, x_t^{15}]$。  
- ​**MineCLIP 相关性评估**：利用预训练的 MineCLIP 模型计算伪视频帧与任务文本指令（如“cut a tree”）的语义相关性得分，作为该区域的探索价值（affordance value）。

#### 快速生成：RAPID AFFORDANCE MAP GENERATION

- ​**原始方法缺陷**：  
  通过滑动窗口裁剪图像生成伪视频帧，并用MineCLIP计算任务相关性生成affordance map，存在**计算成本高**​（需遍历9×9窗口）和**实时性差**的问题。
- ​**核心目标**：  
  设计轻量模型替代原始流程，实现**实时生成高精度affordance map**，支撑强化学习的在线交互。

**网络架构**:
- ​**基础框架**：基于Swin-Unet（Cao et al., 2022）的U型编解码结构，适应高分辨率输出。  
- ​**多模态输入处理**：  
  - ​**图像分支**：Swin Transformer编码器提取多尺度特征（4×4 Patch → 线性嵌入 → 分层下采样）。  
  - ​**文本分支**：MineCLIP文本编码器生成512维文本特征。  
- ​**跨模态融合**：  
  - ​**TIA模块**​（Text-Image Attention）：以文本特征为Query，图像特征为Key/Value，通过多头注意力实现特征对齐。  
  - ​**桥接层**​（Bridge Layer）：融合编码器各阶段的多模态特征，传递至解码器。

**训练策略**
- ​**数据生成**：  
  - 随机策略采集2000张环境图像，通过原始滑动窗口方法生成affordance map作为标签。  
  - 构建数据集 $\{(o_t, I, \mathcal{M}_{o_t,I})\}$，其中$\mathcal{M}_{o_t,I}$为平滑后的affordance map。  
- ​**损失函数**：像素级L1损失 + 结构相似性损失（SSIM），约束输出与标签的一致性。  
- ​**优化细节**：  
  - 初始学习率：$5×10^{-4}$，每50轮衰减至0.1倍，共训练200轮。  
  - 输入图像分辨率：64×64，输出affordance map同分辨率。

#### AFFORDANCE-DRIVEN INTRINSIC REWARD

- ​**问题背景**：  
  开放世界中，稀疏奖励和长周期任务导致探索效率低下。传统方法（如MineCLIP奖励）依赖历史表现，难以直接捕捉未来目标的潜在价值。
- ​**核心目标**：  
  通过**空间先验知识**引导智能体关注任务相关区域，加速目标发现与策略优化。基于 affordance map和高斯分布设计奖励函数，鼓励目标居中。

**内在奖励公式**：  
$$
r_{t}^{\text{intr}} = \frac{1}{WH} \sum_{w=1}^{W} \sum_{h=1}^{H} \mathcal{M}_{o_t, I}(w,h) \cdot \mathcal{G}(w,h)
$$  
- $\mathcal{M}_{o_t, I}$：Affordance Map，表示图像$o_t$中各像素对任务指令$I$的关联性（值域$[0,1]$）。  
- $\mathcal{G}$：中心高斯矩阵，峰值在图像中心，标准差 (超参数) $\sigma_x, \sigma_y$控制分布宽度（图16）。作者的工作选择了（128,80）。
  - 目的：鼓励智能体将目标物体保持在视野中央，便于后续操作。

**总奖励合成**
$$
r_t = r_t^{\text{env}} + r_t^{\text{MineCLIP}} + \alpha r_t^{\text{intr}}
$$  
- $r_t^{\text{env}}$：环境稀疏奖励（如任务完成时+1）。  
- $r_t^{\text{MineCLIP}}$：基于视频-文本对齐的预训练奖励（评估动作序列与任务相关性）。  
- $\alpha$：超参数（默认1），平衡内在奖励的权重。

### 长短期世界模型

长短期世界模型是 ​**LS-Imagine** 的核心组件，旨在同时建模 ​**短时单步转移** 和 ​**长时跳跃转移**，以平衡即时反馈与长期目标的权衡。其架构基于DreamerV3改进，新增 ​**跳跃标志预测** 和 ​**长时状态间隔预测** 模块。

#### LEARNING JUMPING FLAGS

根据当前状态，决定下一次应当预测长期状态转移还是短期状态转移，并且选择对应的转移分支。引入 jumping flags $j_t$，指出在时间步 $t$，应该使用长期或短期状态转移。当观察图像中，远处出现了任务相关的目标，affordance map 应当出现峰值，jumpy state transition 使得 agent 想象未来靠近目标的状态。定义 affordance map 中的相对峰度和绝对峰度如下：

$$
\begin{align*}
K_r &= \frac{1}{WH} \sum_{w=1}^{W} \sum_{h=1}^{H} \left[ \left( \frac{\mathcal{M}_{o,I}(w,h) - \mathrm{mean}(\mathcal{M}_{o,I})}{\mathrm{std}(\mathcal{M}_{o,I})} \right)^4 \right], \\
K_a &= \max(\mathcal{M}_{o,I}) - \mathrm{mean}(\mathcal{M}_{o,I}).
\end{align*}
$$

为了归一化相对峰度，使用如下：

$$
P_{\mathsf{jump}} = \mathrm{sigmoid}(K_r) \times K_a
$$

跳跃概率衡量了置信度。设置一个阈值，当 $P_{jump}$ 大于它时，设置 $j_t$ 为 True，开始 imagination phase。

#### LEARNING JUMPY STATE TRANSITIONS

![ls_arch](assets/images/rl.LongShortTermImagination/ls_arch.png)

状态转移模型，包含了短期和长期分支。大多数基于 DreamerV3。

- ​**短时分支**：  
  建模单步转移：  
  $$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$$  
  - 输入：历史状态 $h_{t-1}$、随机状态 $z_{t-1}$、动作 $a_{t-1}$  
  - 输出：下一确定性状态 $h_t$，w
- ​**长时分支**：  
  建模跳跃转移：  
  $$h_t' = f_\phi(h_{t-1}, z_{t-1})$$  
  - ​**无需动作输入**：直接预测未来多步后的状态（如从当前位置直接跳到树附近）  
  - ​**附加预测器**：  
    - ​**间隔预测器**：预测跳跃所需环境步数 $\Delta_t$  
    - ​**累积奖励预测器**：预测跳跃期间的累积奖励 $G_t$

下标 $t$ 在真实环境下，不是时间步，而是序列的顺序。

#### 模型组成

**短时分支**（Short-Term Transition）
- ​**输入**：历史状态 $(h_{t-1}, z_{t-1})$ 和动作 $a_{t-1}$  
- ​**动态方程**：  
  $$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$$  
- ​**功能**：  
  模拟单步状态转移（类似传统MBRL），预测下一时刻的确定性状态 $h_t$ 和随机状态 $z_t$。

**长时分支**（Long-Term Imagination）
- ​**输入**：历史状态 $(h_{t-1}, z_{t-1})$  
- ​**动态方程**：  
  $$h_t' = f_\phi(h_{t-1}, z_{t-1})$$  
- ​**功能**：  
  不依赖动作 $a_{t-1}$，直接预测跳跃后的状态 $(h_t', z_t')$，模拟智能体接近目标后的未来状态。

#### 关键机制
**跳跃标志**（Jumping Flag）
- ​**计算方式**：  
  - ​**相对峰度**：衡量affordance map中高响应区域的集中程度  
    $$K_r = \frac{1}{WH}\sum \left(\frac{\mathcal{M}_{o,I}(w,h) - \mu}{\sigma}\right)^4$$  
  - ​**绝对峰度**：衡量目标区域的置信度  
    $$K_a = \max(\mathcal{M}_{o,I}) - \mu$$  
  - ​**跳跃概率**：  
    $$P_{\text{jump}} = \text{sigmoid}(K_r) \times K_a$$  
- ​**触发条件**：动态阈值 $P_{\text{thresh}} = \mathbb{E}[P_{\text{jump}}] + \sigma$，若 $P_{\text{jump}} > P_{\text{thresh}}$ 则置 $j_t=1$。

**跳跃状态预测**
- ​**间隔预测器**​（Interval Predictor）：  
  预测从当前状态到跳跃状态所需的真实环境步数 $\Delta_t$。  
  - 通过匹配后续交互中affordance map的响应值确定真实间隔。  
- ​**累积奖励预测器**​（Reward Predictor）：  
  预测跳跃期间的累积奖励 $G_t = \sum_{i=1}^{\Delta_t} \gamma^{i-1} r_{t+i}$。


Buffer 中的每条轨迹包含相邻相邻时间步和长距离间隔时间步的状态。

- **跳转标志（Jumping Flag）**：通过 affordance map 的峰度和均值动态触发跳转。  
- **双分支架构**：  
  - **短期分支**：单步状态转移（如 DreamerV3）。  
  - **长期分支**：预测跳转后的状态、步长间隔（Δₜ）和累积奖励（Gₜ）。  

### 行为学习
- **混合想象力序列**：交替使用长短期想象，通过改进的λ-return计算累积奖励（式9）。  
- **策略优化**：仅对短期想象步骤（jₜ=0）更新策略，避免跳转状态的动作缺失问题。  

## insight

作者提出了长短期世界模型。也许我们可以不用长短期世界模型，使用长短期想象，再加以验证。世界模型类似 RL 的 Value-Based，我们使用 Rule-based 的方法，毕竟任务只有一个，找到伤员。

可以使用探索，想象来作为启发式的 hint。以 VLM 给出图像哪个方向最有可能存在伤员，VLM 作为“大脑”来思考和决策。小模型则探索。解决世界模型可能难以泛化和训练的问题。

## Tag
#Paper
#RL