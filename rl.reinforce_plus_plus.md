---
id: ylw0hme4i9wsnb5l5uuwun8
title: Reinforce_plus_plus
desc: ''
updated: 1743628719607
created: 1743621094410
---

## 1. 算法背景

Reinforce++ 通过引入多种技术（如基线函数、优势估计、信任域约束等）提升性能，可视为 ​**REINFORCE 与现代策略梯度方法（如PPO）的结合体**。REINFORCE++的特点是 比 GRPO 稳定比PPO快。

PPO有4个模型，actor，critic，reference，reward。其中 actor (policy model，最终的推理模型) 和 critic 都是需要训练并更新参数的模型，而且二者大小差不多，非常占显存，很难scaling（比如deepseek v3 600B，训练一个600B就已经巨难了，同时训练两个600B，会不会疯）。

业界的共识是去掉PPO的critic模型，于是只有一个actor是训练模型，ref和reward是推理模型。推理模型就很好解决了，你甚至可以用一个远程的url模型。比较有代表性的算法就是Reinforce++和GRPO。

PPO 需要求梯度上升，最大化目标函数。

$$
\nabla J(\pi_\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \Psi_t \nabla \log \pi_\theta(a_t | s_t) \right]
$$

$$
\approx \frac{1}{N} \sum_{n=0}^{N-1} \sum_{t=0}^{T_n-1} \Psi_t \nabla \log \pi_\theta(a_t | s_t)
$$

上面的 $\Psi_t$ 是衡量价值的函数。通常是
- 累积奖励 (MC 方法)，有 $G_t=\Sigma^T_{k=t+1}\gamma^(k-t)r_k$。
- TD 方法。使用 critic 估计值函数，偏差大方差小。
- GAE，用超参数控制 MC 和 TD 的比例。超参 λ 趋于 0，则退化为 TD(0)；趋于 1 则类似 MC 方法。

既然我们想删掉critic model，那么自然可以从GAE退化到完全用2来估计价值，就变为累积折扣奖励，也就是reinforce用的估计价值的方法。但是呢，PPO的一些非常重要的trick，重要性采样、clip、归一化等等仍得到了保留，因此效果仍会不错，就叫Reinforce++。（openrlhf中，默认 λ\lambda\lambda 就是0.95，TD误差参与的其实很少。在语言模型中，critic模型很难训练得特别好，因此它在GAE中也不应发挥过多作用，删掉训练的不怎么样的critic，reinforce++效果也还可以。）

reinforce++ 是直接调的 ppo trainer。但是在计算 return 和 advantage 的时候走了不同的分支，别的都一样

---

## 流程

## 2. 核心改进点

### 2.1 基线函数（Baseline）
- ​**目的**：降低梯度估计的方差。
- ​**方法**：从回报 $G_t$ 中减去与状态相关的基线（通常为值函数 $V(s_t)$）：
  $$
  \nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - V(s_t)) \right]
  $$
- ​**效果**：保持无偏性的同时显著降低方差。

### 2.2 优势函数（Advantage Function）
- ​**目的**：更精确评估动作的优劣。
- ​**方法**：用优势函数 $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$ 替代原始回报，常见实现包括：
  - ​**GAE（Generalized Advantage Estimation）​**：多步优势估计（见公式推导）。
  - ​**Actor-Critic 框架**：联合训练策略网络（Actor）和值函数网络（Critic）。

### 2.3 熵正则化（Entropy Regularization）
- ​**目的**：鼓励策略探索，防止过早收敛到局部最优。
- ​**方法**：在损失函数中增加策略熵的负项：
  $$
  J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \log \pi_\theta(a_t|s_t) A_t \right] + \beta \mathbb{E}_\tau \left[ H(\pi_\theta(\cdot|s_t)) \right]
  $$
  - $H$ 为熵，$\beta$ 是权重系数。

### 2.4 信任区域约束（Trust Region）
- ​**目的**：限制策略更新的幅度，避免破坏性更新。
- ​**方法**：
  - ​**PPO 截断（Clipping）​**：限制新旧策略的概率比范围：
    $$
    J(\theta) = \mathbb{E}_\tau \left[ \min\left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A_t, \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\epsilon, 1+\epsilon\right) A_t \right) \right]
    $$
  - ​**自然策略梯度**：利用 Fisher 信息矩阵约束更新方向。

### 2.5 重要性采样（Importance Sampling）
- ​**目的**：复用历史数据，提升样本效率。
- ​**方法**：通过旧策略的概率修正梯度估计：
  $$
  \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \nabla_\theta \log \pi_\theta(a_t|s_t) A_t \right]
  $$

---

## 3. 算法流程
1. ​**数据收集**：使用当前策略 $\pi_\theta$ 与环境交互，生成轨迹数据。
2. ​**优势估计**：通过 Critic 网络计算每个状态动作对的优势值 $A_t$（如使用 GAE）。
3. ​**策略更新**：
   - 计算新旧策略概率比。
   - 应用信任区域约束（如 PPO 截断）。
   - 添加熵正则化项。
4. ​**值函数更新**：优化 Critic 网络以最小化值函数误差（如均方误差）。

---

## 4. 与经典算法对比
| ​**改进项**   | REINFORCE | Actor-Critic | PPO | Reinforce++   |
| ------------- | --------- | ------------ | --- | ------------- |
| 基线/优势函数 | ❌         | ✅            | ✅   | ✅（GAE）      |
| 重要性采样    | ❌         | ❌            | ✅   | ✅             |
| 信任区域约束  | ❌         | ❌            | ✅   | ✅（PPO 截断） |
| 熵正则化      | ❌         | ❌            | ✅   | ✅             |
| 多步回报      | ❌         | ✅（TD）      | ✅   | ✅（GAE）      |

---

## 5. 代码示例（简化版）
```python
import torch

class ReinforcePlusPlus:
    def __init__(self, actor, critic, gamma=0.99, lambda_=0.95, clip_eps=0.2, entropy_coef=0.01):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef

    def compute_gae(self, rewards, values, next_values):
        deltas = rewards + self.gamma * next_values - values
        advantages = torch.zeros_like(rewards)
        advantages[-1] = deltas[-1]
        for t in reversed(range(len(deltas)-1)):
            advantages[t] = deltas[t] + self.gamma * self.lambda_ * advantages[t+1]
        return advantages

    def update(self, states, actions, old_log_probs, rewards, next_states):
        # 计算优势值
        values = self.critic(states)
        next_values = self.critic(next_states)
        advantages = self.compute_gae(rewards, values, next_values)

        # 计算新策略的概率比
        new_log_probs = self.actor.get_log_prob(states, actions)
        ratios = torch.exp(new_log_probs - old_log_probs.detach())

        # 策略损失（带截断）
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.clip_eps, 1+self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 熵正则化
        entropy = self.actor.get_entropy(states).mean()
        policy_loss -= self.entropy_coef * entropy

        # 值函数损失
        value_loss = 0.5 * (values - rewards).pow(2).mean()

        # 整体更新
        total_loss = policy_loss + value_loss
        total_loss.backward()
        optimizer.step()
```

## 6. 核心优势

​- 低方差：通过优势估计（GAE）和基线函数降低梯度方差。
​- 高样本效率：重要性采样和信任区域约束允许复用历史数据。
​- 稳定训练：PPO 截断和熵正则化防止策略崩溃。
​- 灵活探索：熵项动态调整策略的随机性。

## 7. 应用场景
- 连续或离散动作空间（如机器人控制、游戏AI）。
- 需要稳定训练的复杂环境（如自动驾驶模拟）。
- 对样本效率要求较高的在线学习任务。

## Ref and Tag

https://verl.readthedocs.io/en/latest/examples/config.html#algorithm

RLHF 对齐之 REINFORCE++ 算法 - 比 GRPO 稳定比PPO快 - 初七123334的文章 - 知乎
https://zhuanlan.zhihu.com/p/14888098807

从PPO到Reinforce++，再对比GRPO - lym的文章 - 知乎
https://zhuanlan.zhihu.com/p/22023807402