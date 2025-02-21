---
id: wgv9wpy2umwsjvmcmt97jcg
title: GRPO_DeepSeekMath
desc: ''
updated: 1740147604132
created: 1738233120668
---

https://arxiv.org/abs/2402.03300
https://www.youtube.com/watch?v=bAWV_yrqx4w
【有难度但必读的一篇论文《DeepSeekMath》】 https://www.bilibili.com/video/BV1qUFMeGE2q/?share_source=copy_web&vd_source=fe1db0b3f565b5940b244b9c7f7b4fa1

爬取了 120B math-related tokens，提出了 Group Relative Policy Optimization (GRPO) 算法，对比 PPO，优化了内存使用。

## Introduction
构建了高质量数据集，包含 120B 数学 tokens。这些数据集由 the Common Crawl (CC) using a fastText-based classfier 实现。也就是说，网上的数据不够好。

## Math Pre-training
### Data Collection and Decontamination
To explain the DeepSeekMath Corpus from CC, 作者提出了 an iterative pipeline，

## 4 RL
### 4.1 Group Relative Policy Optimization (GRPO)
PPO 计算如下：
$$
\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E} \left[ q \sim P(Q), o \sim \pi_{\theta_{\text{old}}}(O|q) \right] \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[ \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t|q, o_{<t})} A_t, \text{clip} \left( \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t|q, o_{<t})}, 1 - \varepsilon, 1 + \varepsilon \right) A_t \right]
$$

在强化学习的策略优化中，与训练 LLM 不太相同，训练模型一般是求梯度下降，RL 要使得奖励最大化，于是需要求梯度上升。新模型和老模型参数比较，再乘以优势函数，得到 policy gradient，策略更新梯度。$o_t$ 是观察，也就是状态。公式中对所有状态的情况球了平均。

$\mathbb{E}$ 部分代表概率分布，来自于就得概率模型分布。PPO 算法要按照 $\mathcal{J_{\text{PPO}}}$ 最大化来优化参数。

![ppo_grpo](assets/images/llm.DeepSeekMath/ppo_grpo.png)

PPO 中，query q，动作 A，输入到 Policy Model，得到状态 o。Reference Model 和 Reward Model 是人为设置且不训练的，使用 KL 评估两个分布的差异，算出惩罚值 r。结合价值模型，输出 v。PPO 力求尽可能小，逐渐更新模型。优势函数采用 Generalized Advantage Estimation (GAE)。PPO 需要同时训练 Policy Model 与 Value Model，训练量比较大。注意，每个 token 都要计算 KL penalty 如下：

$$
r_t = r_{\varphi}(q, o_{\le t}) - \beta \log \frac{\pi_{\theta}(o_t | q, o_{<t})}{\pi_{\text{ref}}(o_t | q, o_{<t})}
$$

$r_\phi$ 是奖励模型，R1 工作中便是 rule-based 的奖励系统，没有模型了。$\pi_\theta$ 是当前 Policy Model，$\pi_{\theta_\text{old}}$ 是前一个 Policy Model。两个 π 相等时，log 部分为 0，代表不需要惩罚。$\pi_\text{ref}$ 通常是最初的 SFT 模型，比如 R1 中的 V3。$\beta$ 是超参数。

Tips：在 ML，数值或概率之比通常都会加上 log，以便缩小数值到较小范围，保证稳定更新。

GRPO 仅训练 Policy Model，降低了训练量。比如训练推理模型时，query q 是一道数学题，经过 Policy Model 得到一系列的 QA 对样本 （R1-ZERO 中，G=60 多个）。直接输入给两个 Model，这两个 Model 中，在 R1 中，Reference Model 是 V3 模型，Reward Model 是专门训练给与奖励值的模型。分别输出 r1, r2, ...。分组后，与 KL 值一起反馈并更新 Policy Model。更新参数的幅度，不需要通过在线训练的模型拿到，而是根据计算结果拿到。

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q) \right]
$$

$$
\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t}, \mathrm{clip} \left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}, 1-\varepsilon, 1+\varepsilon \right) \hat{A}_{i,t} \right] - \beta \mathbb{D}_{KL} \left[ \pi_\theta || \pi_{\text{ref}} \right] \right\}
$$

在 R1 中，使用 rule-based reward system，主要包含 Accuracy Rewards 和 Format Rewards。每个 Group 的优势函数计算如下：
$$
A_i=\frac{r_i-\textit{mean}(r_1,r_2,\ldots,r_G)}{\textit{std}(\{r_1,r_2,\ldots,r_G\})}
$$

注意，GRPO 也要循环，因为有 G 组输出。
![grpo_alg](assets/images/llm.DeepSeekMath/grpo_alg.png)

KL 惩罚中，计算如下：
$$
\mathbb{D}_{KL} \left[ \pi_{\theta} || \pi_{\text{ref}} \right] = \frac{\pi_{\text{ref}}(o_{i,t} | q, o_{i,<t})}{\pi_{\theta}(o_{i,t} | q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} | q, o_{i,<t})}{\pi_{\theta}(o_{i,t} | q, o_{i,<t})} - 1,
$$

### 4.1.2 Outcome Supervision RL with GRPO
只利用最终的一个奖励模型。

### 4.1.3 Process Supervision RL with GRPO
过程奖励模型，奖励过程。

### 4.1.4. Iterative RL with GRPO
前面训练的方式中，Reward Model 已经预训练完毕，冻结参数。此思路中，认为旧的 reward model 不足以监督当前的 policy model，于是 policy model 与 reward model 两者在迭代训练中不断地互相交换。

## 5.2. Insights of Reinforcement Learning
### 5.2.1. Towards to a Unified Paradigm
RL 范式朝向统一，比如 SFT、RFT、DPO、PPO、GRPO，都是求出最大的梯度。

## ToDO
【手撕LLM-GRPO】你只管给Reward, 剩下的交给RL（附代码） - 小冬瓜AIGC的文章 - 知乎
https://zhuanlan.zhihu.com/p/20812786520

## Tag
#Paper
#GRPO