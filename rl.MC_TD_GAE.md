---
id: u6qhoc0n6v4ycnjau5e759d
title: MC_TD_GAE
desc: ''
updated: 1743623965943
created: 1743613518985
---

蒙特卡洛(MC) vs 时序差分(TD) vs 广义优势估计(GAE)

贝尔曼期望方程的目标是找到 Discounted Return。

## 蒙特卡洛 (MC)

MC 方法基于贝尔曼方程。MC 的核心在于每次 episode 结束后计算值函数，寻找满足贝尔曼期望方程的值函数。

状态值函数：

$ V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s \right] $

动作值函数：

$ Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s, a_t = a \right] $

### 贝尔曼期望方程的两种形式

状态值函数的贝尔曼方程：

$ V^\pi(s_t) = \mathbb{E}_\pi \left[ r_t + \gamma V^\pi(s_{t+1}) \mid s_t = s \right] $

动作值函数的贝尔曼方程：

$ Q^\pi(s_t, a_t) = \mathbb{E}_\pi \left[ r_t + \gamma Q^\pi(s_{t+1}, a_{t+1}) \mid s_t = s, a_t = a \right] $

累积折扣回报：

$ G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-t-1} r_{T-1} $

递推形式：

$ G_t = r_t + \gamma G_{t+1} $

### MC 方法

MC 方法基于完整的 episode 数据。在 episode 中，每一时间步都有奖励，但是折扣回报需要得到最后一步的奖励才能计算。每次 episode 结束，得到最后的奖励，可以开始从末尾向起始计算折扣回报，根据折扣回报更新价值函数估计。对于轨迹的每个时间步 t，从 t 时刻计算累积折扣汇报 G_t，T 为终止状态。

存在多条轨迹时，MC 方法对 G_t 取平均。如此，MC 方法从结束时间步往 t0 时刻，便可计算回报，更新值函数。

当估计状态价值函数 $ V^\pi(s) $ 时，即在策略 $\pi$ 下，从状态 $s$ 开始的期望累计折扣回报，其估计过程如下：

1. **生成多个轨迹**：按照策略 $\pi$ 从不同的初始状态开始，与环境交互生成多条轨迹，即多个 episode。
2. **计算每个状态的回报**：对于轨迹中的每一个访问的状态 $s_t = s$，记录对应的回报 $G_t$。
3. **更新价值函数**：

    $$
    V(s) = \frac{1}{n} \sum_{i=1}^{n} G_t^{(i)}
    $$

其中，$n$ 是状态 $s$ 被访问的次数，也就是状态 $s$ 在 episode 中出现的次数。$G_t^{(i)}$ 是第 $i$ 次访问 $s$ 时的回报。为了避免存储所有的回报，常使用递增平均的方法更新 $V(s)$：

$$
V(s) \leftarrow V(s) + \frac{1}{n}(G_t - V(s))
$$

简单推导：

假设 $V(s)$ 是所有回报 $G_t^{(1)}, G_t^{(2)}, \ldots, G_t^{(n-1)}$ 的平均值，那么增加第 $n$ 个回报 $G_t^{(n)}$ 后，新均值为：

$$
V_{\text{new}}(s) = \frac{(n-1)V(s) + G_t^{(n)}}{n}
$$

等价于：

$ V_{\text{new}}(s) = V(s) + \frac{1}{n} \left( G_t^{(n)} - V(s) \right) $

对于状态-动作价值函数 $ Q^\pi(s, a) $，类似地：

$ Q^\pi(s, a) = \frac{1}{n} \sum_{i=1}^n G_t^{(i)} $

其中，$ n $ 是状态-动作对 $(s, a)$ 被访问的次数。

但是在实际使用时，多通过下面方法进行更新：

$ V(s_t) \leftarrow V(s_t) + \alpha [G_t - V(s_t)] $

其中，$\alpha$ 是学习率，用来替代 $\frac{1}{n}$，原因主要有几点：

- 适应非平稳环境 在非平稳环境中，价值函数可能随时间变化，早期的观察可能不再准确反映当前情况。固定 $\alpha$ 给予最近的观察更多权重，使得算法能够跟踪变化的环境。
- 持续学习能力 当使用 $1/N$ 时，随着 $N$ 增大，更新步长会变得非常小，导致学习几乎停止。固定 $\alpha$ 确保了持续的学习能力。

## 时序差分 (TD)

MC 方法需要等到 episode 结束才能更新值函数，而 TD 可以在每步更新。核心也是依赖贝尔曼方程，但是 TD 提供了递归定义价值函数的方法，即当前状态的价值有**即时奖励**和**下一状态**的价值决定。

目标也是寻找满足贝尔曼期望方程的值函数。

### TD(0) 简单的 TD 算法

更新规则如下：

$ V(s_t) \leftarrow V(s_t) + \alpha [r_t + \gamma V(s_{t+1}) - V(s_t)] $

- $\alpha$ 是学习率，控制更新步长。
- $r_t$ 是在时间步 $t$ 获得的即时奖励。
- $\gamma$ 是折扣因子，$0 \leq \gamma < 1$。
- $V(s_t)$ 和 $V(s_{t+1})$ 分别是当前状态和下一个状态的价值估计。

定义 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，即为 TD 误差。公式可以简化为：

$ V(s_t) \leftarrow V(s_t) + \alpha \delta_t $

TD(0) 的目标是通过不断**减少**时序差分误差 $\delta_t$，使得 $V(s_t)$ **,收敛**到 $V^\pi(s_t)$。

具体推导为：

$ \mathbb{E}[V(s_t)] = \mathbb{E}[V(s_t) + \alpha (r_t + \gamma V(s_{t+1}) - V(s_t))] $
$ = V(s_t) + \alpha (\mathbb{E}[r_t + \gamma V(s_{t+1})] - V(s_t)) $

根据贝尔曼方程，$\mathbb{E}[r_t + \gamma V(s_{t+1})] = V^\pi(s_t)$，因此：

$ \mathbb{E}[V(s_t)] = V(s_t) + \alpha (V^\pi(s_t) - V(s_t)) $

随着 $ t \to \infty $:

$ \mathbb{E}[V(s_t)] \to V^\pi(s_t) $

相比于蒙特卡洛方法需要估计整个回合的回报，TD 方法仅依赖当前奖励和下一状态的价值估计。这降低了估计的**方差**，因为回报的方差随着时间步数的增加而累积，特别是在高折扣因子下更为显著。同时，由于不需要等待完整回合便可更新，因此效率更高。TD 方法存在的问题在于高偏差，因为过多使用了估计值。

### TD(λ)

TODO

## 广义优势估计 (GAE)

GAE 用于策略梯度方法中优势函数估计。引入参数 $\lambda$，在偏差和方差之间动态平衡。核心思想是将多步时序差分 (TD) 误差加权和作为优势函数的估计，结合 MC 高方差低偏差 和 TD 底方差高偏差的有点。本质是将 TD(λ) 思想用于优势估计。

状态 $s_t$ 下，优势函数 $A(s_t, a_t)$ 表示动作 $a_t$ 比平均期望多的额外收益：

$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$

Q(s_t, a_t) 表示状态-动作值函数，s_t 采取 a_t 后的期望累积奖励。V(s_t) 表示状态值函数，s_t 下遵循当前策略的期望累积奖励。

GAE 引入可调节的超参数 $\lambda$ 平衡偏差和方差，实现更稳定和高校的优势函数估计。而折扣因子 $\gamma$ 通常接近 1，比如 0.99。

GAE 优势估计公式：

$ \hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} $

其中，$\delta_t$ 是时序差分误差，定义为：

$ \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) $

GAE 可以通过递归的方式表示为：

$ \hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1} $

这意味着当前的优势估计不仅包含当前时刻的 TD 误差 $\delta_t$，还包含未来时刻的优势估计 $\hat{A}_{t+1}$，并通过参数 $\gamma \lambda$ 进行加权。

展开递归公式，可以得到：

$ \hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots + (\gamma \lambda)^{T-t-1} \delta_{T-1} $

假设轨迹长度为有限的 $T$，也就是一个 episode 长度。则优势估计为：

$ \hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l} $

这表明 GAE 是未来所有 TD 误差的加权和，权重随步数呈几何衰减。

GAE 可通过反向递归高效计算：

$ \hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1} $

在轨迹终止时 ($ t = T $)，设 $\hat{A}_T = 0$（因无未来奖励）。

假设轨迹长度为 $ T $，按逆序（从 $ t = T - 1 $ 到 $ t = 0 $）计算：

- **最后一步 ($ t = T - 1 $)**：

  $ \hat{A}_{T-1} = \delta_{T-1} + \gamma \lambda \cdot 0 = \delta_{T-1} $

- **中间步骤 ($ t $)**：

  $ \hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1} $

### 伪代码

```py
def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
    T = len(rewards)
    deltas = [rewards[t] + gamma * values[t+1] - values[t] for t in range(T)]
    advantages = [0] * T
    advantage = 0
    for t in reversed(range(T)):
        advantage = deltas[t] + gamma * lambda_ * advantage
        advantages[t] = advantage
    return advantages
```

### PPO 中的 GAE

在收集轨迹数据时，策略 $\pi_\theta$ 与环境交互，收集轨迹：

$$
(s_0, a_0, r_0, s_1), (s_1, a_1, r_1, s_2), \ldots, (s_{T-1}, a_{T-1}, r_{T-1}, s_T)
$$

使用值函数网络（Critic）估计每个状态的值 $V(s_t)$。

计算TD误差。对每个时间步 $t$，计算单步TD误差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

递归计算GAE。从后向前（反向遍历时间步），利用递归公式高效计算GAE：

$$
A_t = \delta_t + (\gamma \lambda) A_{t+1}
$$

初始条件：$A_T = 0$。

### 超参数 λ 的讨论

$\lambda=0$ 时，GAE 退化为单步 TD 误差，即 $A_t = \delta_t$。此时估计偏差大但方差低。

$\lambda=1$ 时，GAE 变成 $-V_\pi(s_t)+\Sigma^\infty_{l=0}\gamma^l r_{t+l}$，MC 的形式。

## Ref and Tag

蒙特卡洛(MC) vs 时序差分(TD) vs 广义优势估计(GAE) - Sonsii的文章 - 知乎
https://zhuanlan.zhihu.com/p/22431139619