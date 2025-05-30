---
id: xpm7k3l0en3aumjgrs2gri7
title: HoST_学习不同姿势下人形机器人站立控制
desc: ''
updated: 1745954382102
created: 1742398503572
---

论文：Learning Humanoid Standing-up Control across Diverse Postures。作者提出 HoST 框架，利用多 critic 架构，高效学习姿势（posture）适应的动作和基于课程来训练学习不同地形平面下的动作。

### 1. 作者和团队信息
- **作者**：Tao Huang, Junli Ren, Huayi Wang, Zirui Wang, Qingwei Ben, Muning Wen, Xiao Chen, Jianan Li, Jiangmiao Pang
- **团队**：上海交通大学、上海人工智能实验室、香港大学、浙江大学、香港中文大学等

### 2. 背景和动机
- **背景**：人形机器人从不同姿势站起来的能力对于其在现实世界中的应用至关重要，如从沙发上站起来、从跌倒中恢复等。
- **动机**：现有的方法要么局限于模拟环境，忽略了硬件限制，要么依赖于特定地面的预定义运动轨迹，无法在真实世界中实现跨姿势的站立。

### 3. 相关研究
- **经典的人形机器人站立控制方法**：依赖于通过基于模型的运动规划或轨迹优化来跟踪手工制作的运动轨迹，但计算量大，对干扰敏感，并且需要精确的执行器建模，限制了它们在现实世界中的应用。
- **基于强化学习的人形机器人控制方法**：通过最小化建模假设来学习控制策略，但现有的方法没有在真实世界中展示出跨多种姿势的站立运动能力。
- **四足机器人的站立控制**：一些基于强化学习的方法已经使四足机器人能够从跌倒中恢复并转换姿势，为本文提供了灵感。

### 4. 核心思路
- 提出一个名为 **HoST (Humanoid Standing-up Control)** 的强化学习框架，通过多阶段任务分解、多批评家架构、探索策略优化、运动平滑性正则化以及从模拟到现实的转移技术，实现人形机器人从多种姿势中站起来的控制。

### 5. 方案与技术

- **强化学习框架**：采用 Proximal Policy Optimization (PPO) 算法，将任务分解为**三个阶段（扶正身体、跪起和站立）**，并为每个阶段设计相应的奖励函数。
- **环境设置**：在模拟环境中设计了四种地形（地面、平台、墙壁和斜坡），以模拟现实世界中可能遇到的不同起始姿势。
- **奖励函数设计**：包括任务奖励、风格奖励、正则化奖励和后任务奖励，通过多批评家架构独立地估计回报并优化策略。
- **探索策略**：在训练初期施加垂直拉力帮助机器人从完全倒下的状态过渡到稳定的跪姿，并通过动作缩放系数 β 逐渐收紧动作输出范围。
- **运动约束**：采用 L2C2 方法进行平滑性正则化，减少运动的振荡，并通过动作范围限制防止剧烈运动。
- **sim-to-real 转移技术**：通过领域随机化减少模拟与现实之间的差距，提高控制策略在现实世界中的适应性。


#### 问题建模

建模为具有时间范围的 MDP，表示为 $\mathcal{M}=⟨\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \mathcal{\gamma}⟩$

每时刻 t 观察到状态 s_t，策略根据状态生成动作 a_t。根据环境状态转移，得到新的状态 s_t+1 和奖励 r_t。为了最大化奖励，使用 RL，目的得到最优的策略 π_θ，最大化 the expected cumulative reward (return) $\mathbb{E}_{\pi_\theta}[\Sigma_{t=0}^{T-1} \gamma^t r_t], \gamma \in [0,1]$。力求 episode length T 内获取最大奖励。期望的返回由值函数（critic）评测。使用 PPO 作为 RL 算法。

- State space：本体感知信息包括 IMU 和关节角 $s_t=[\omega_t,r_t,p_t,\dot{p}_t,a_{t-1},\beta]$。其中，$\omega_t$ 代表机器人 base 的角速度，$r_t$ 和  $p_t$ 代表 roll pitch，$p_t$，$\dot{p}_t$ 代表关节角位置和速度。$a_{t-=1}$ 代表上一次动作，$\beta\in(0,1]$ 代表输出动作的缩放因子。
- Action space：使用 PD 控制器，基于力矩来驱动。动作 a_t 代表当前和下一步关节角位置的差值，于是 PD 目标计算为 $p_t^d=p_t + \beta a_t$，其中，a_t 每一维度的值都限制在 [-1,1]。β 隐式地限制动作的速度，是重要的部分。力矩可以根据刚度系数 $K_p$ 和阻尼系数 $K_d$ 计算。$\tau_t=K_p \cdot (p_t^d - p_t) - K_d \cdot \dot{p}_t$

#### 更多细节

奖励设置和优化方面，站立涉及多个电机相关的技能：比如身体、屈膝、上升。

### 6. 实验与结论
- **模拟实验**：在四种模拟地形上进行了广泛的实验，验证了 HoST 框架的有效性，成功率达到 99.5% 以上，运动平滑性和能量消耗也得到了有效控制。
- **现实世界实验**：将训练好的控制策略直接部署到 Unitree G1 人形机器人上，并在实验室和户外环境中进行了测试，结果表明该控制策略能够在多种场景中实现平滑、稳定和鲁棒的站立动作。
- **结论**：HoST框架成功地解决了人形机器人从多种姿势站起来的控制问题，并在现实世界中展示了其有效性。

### 7. 贡献
- **无需预定义轨迹**：通过强化学习从头开始学习站立控制，无需依赖预定义的运动轨迹。
- **跨姿势适应性**：能够在多种不同的起始姿势下实现站立，包括地面、平台、墙壁和斜坡。
- **现实世界部署**：通过运动平滑性正则化和动作范围限制，确保控制策略在物理硬件上的实际可行性。
- **鲁棒性**：在现实世界中展示了对环境干扰（如外力、障碍物和负载）的强鲁棒性。

### 8. 不足
- **环境感知能力**：当前方法主要依赖于机器人自身的本体感知信息，在复杂环境中可能不足以处理与周围环境的交互。
- **姿势和场景的广泛性**：虽然已经考虑了多种姿势和场景，但训练过程中同时包含仰卧和俯卧姿势可能会导致采样干扰，影响性能。
- **与现有系统的集成**：尚未展示如何将站立控制与现有的人形机器人系统（如导航、抓取等）进行集成。
- **sim-to-real转移效果**：尽管领域随机化在一定程度上减少了模拟与现实之间的差距，但仍然存在一些差距，特别是在关节扭矩方面。
- **运动的自然性和效率**：在某些情况下，动作的自然性和效率仍有提升空间。

## 项目代码

```bash
uv pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

```
   File "/workspace/HoST/isaacgym/python/isaacgym/torch_utils.py", line 135, in <module>
    def get_axis_params(value, axis_idx, x_value=0., dtype=np.float, n_dims=3):
```

torch_utils.py:135 的 np.float 改为 np.float64。

### IsaacGym 渲染出错：需要安装 OpenGL 工具和库

运行 IsaacGym 样例程序时，发生段错误（Segmentation Fault）:

```bash
examples# python 1080_balls_of_solitude.py
Importing module 'gym_38' (/workspace/engineai_legged_gym/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so)
Setting GYM_USD_PLUG_INFO_PATH to /workspace/engineai_legged_gym/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugInfo.json
WARNING: Forcing CPU pipeline.
Not connected to PVD
+++ Using GPU PhysX
Physics Engine: PhysX
Physics Device: cuda:0
GPU Pipeline: disabled
Segmentation fault (core dumped)
```

如果指定训练 legged_gym 为 headless，不会出现错误。如果指定渲染图形界面模式，会出现错误。

解决方案，安装 OpenGL 工具和库：

```bash
apt install -y mesa-utils libglvnd-dev
```

为了在 play 时候正常使用界面，依然需要安装：

```bash
apt install -y libglfw3 libglfw3-dev libgl1-mesa-dev libx11-dev
```


## Ref and Tag

[Arxiv](https://arxiv.org/abs/2502.08378)