---
id: 2l5nw6bxqnhp42w12n7vw8d
title: TODO
desc: ''
updated: 1742134926500
created: 1738165674608
---

## LLM Reasoning

rStar 小模型推理
https://arxiv.org/abs/2501.04519
https://github.com/microsoft/rStar

OpenAI 的 O1 模型的原理是什么？ - 猛猿的回答 - 知乎
https://www.zhihu.com/question/666999747/answer/4472268952

自主机器人将强化学习与基础模型相结合：方法与观点 - 黄浴的文章 - 知乎
https://zhuanlan.zhihu.com/p/20365147329

RL 框架：verl，https://arxiv.org/pdf/2409.19256v2

R1-V 突破 2K Star，持续进化中！ - Lei Li的文章 - 知乎
https://zhuanlan.zhihu.com/p/22989750949
https://github.com/Deep-Agent/R1-V

多模态R1复现之旅… - pinkman的文章 - 知乎
https://zhuanlan.zhihu.com/p/22890208704

大模型强化学习面经 - 一蓑烟雨的文章 - 知乎
https://zhuanlan.zhihu.com/p/659551066

人型机器人行走
https://why618188.github.io/beamdojo/

多智能体协作综述
https://arxiv.org/abs/2501.06322

SMAC-R1：在MARL中复现R1时刻 - 赵鉴的文章 - 知乎，星际争霸
https://zhuanlan.zhihu.com/p/24922558098

笔记：MoBA 与 Native Sparse Attention - 刀刀宁的文章 - 知乎
https://zhuanlan.zhihu.com/p/24774848974

DeepSpeed 的 ZeRO 解读。

摸着Logic-RL，复现7B - R1  zero - aaaaammmmm的文章 - 知乎
https://zhuanlan.zhihu.com/p/25982514066

在 Qwen2.5-VL 复现 R1
https://github.com/om-ai-lab/VLM-R1

【论文解读】LLM-Microscope：揭秘 LLM 中不起眼 Token 的隐藏力量 - tomsheep的文章 - 知乎
https://zhuanlan.zhihu.com/p/26492642537

UCB cs294/194-196 Large Language Model Agents 课程笔记 - Perf的文章 - 知乎
- https://zhuanlan.zhihu.com/p/26269945565
- [CS294/194-196 Large Language Model Agents](https://rdi.berkeley.edu/llm-agents/f24)
- [UCB_CS294_LLmAgents](https://github.com/WangYuHang-cmd/UCB_CS294_LLmAgents)

LLamaV-o1: (https://mbzuai-oryx.github.io/LlamaV-o1/)

## Robotics

MapNav 使用了 GPT 标注图像数据。可以借鉴处理。
https://arxiv.org/pdf/2502.13451

https://github.com/Peterwangsicheng/RoboBERT

RoboGrasp：一种用于稳健机器人控制的通用抓取策略 - 黄浴的文章 - 知乎
https://zhuanlan.zhihu.com/p/22946605267

独家专访｜清华TEA实验室负责人：具身智能入门/转行到底学什么？ - 深蓝学院的文章 - 知乎
https://zhuanlan.zhihu.com/p/26333134789

## VLA

通过 Affordance 链改进视觉-语言-动作模型 - 黄浴的文章 - 知乎
https://zhuanlan.zhihu.com/p/21713958996

VLA 等各种工作合集。博士想读具身智能/智能机器人应该怎么规划自己的科研？ - EyeSight1019的回答 - 知乎
https://www.zhihu.com/question/655570660/answer/87040917575

Physical Intelligence 最新的 Hi Robot：基于分层的 VLA 模型的开放式指令遵循。
http://www.pi.website/research/hirobot

基于RoboTwin生成海量数据Finetune RDT-1B等具身大模型保姆级教程 - 穆尧的文章 - 知乎
https://zhuanlan.zhihu.com/p/22754193110

具身智能VLA方向模型fine-tune（单臂）（24.12.26已完结）
https://blog.csdn.net/iamjackjin/article/details/144534904?fromshare=blogdetail&sharetype=blogdetail&sharerId=144534904&sharerefer=PC&sharesource=qq_39422041&sharefrom=from_link
> 作者采集了 150 条数据，每个任务 10-30 条，
## RL
在线强化学习改进VLA模型 - 黄浴的文章 - 知乎
https://zhuanlan.zhihu.com/p/23993973779

面向长范围交互式 LLM 智体的强化学习 - 黄浴的文章 - 知乎
https://zhuanlan.zhihu.com/p/24109661682

TRL 
https://huggingface.co/docs/trl/main/en/index

GPRO, R1
https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb

导航 RL
https://github.com/Zhefan-Xu/NavRL

[大模型 30] SWE-RL 强化学习提高模型软件工程能力 - hzwer 黄哲威的文章 - 知乎
https://zhuanlan.zhihu.com/p/26792881958

九坤联合微软亚洲研究院等成功复现 DeepSeek-R1，具体水平如何？ - 薛定谔的猫的回答 - 知乎
https://www.zhihu.com/question/13238901947/answer/111931939432

九坤联合微软亚洲研究院等成功复现 DeepSeek-R1，具体水平如何？ - 到处挖坑蒋玉成的回答 - 知乎
https://www.zhihu.com/question/13238901947/answer/112109118245
> 项目中的奖励函数设计对其他类似任务有重要的实践意义，建议 RL 做其他任务的学习下。如果 reward 判定写得不够严密，模型学习过程容易钻空子，骗取高 reward。K & K 是合成逻辑谜题 (K & K puzzle) 数据集。
> 参考：https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py

LLM RL入门 - Reku的文章 - 知乎
https://zhuanlan.zhihu.com/p/27172237359

R1 小模型复刻失败经验总结
https://mp.weixin.qq.com/s?__biz=Mzg5MTU1NTE1OQ==&mid=2247492943&idx=1&sn=aed1d150faaebb17ac0b9c3d3347e436&chksm=ce0e65e1751e208b29ad2c6c024d4f54504f1d5c43939e447f805d02b447de619f30fd1fd1a3&mpshare=1&scene=23&srcid=03115atkpiBzet6MMBOt0cSX&sharer_shareinfo=f1c7bb00c78885ad976322928466b740&sharer_shareinfo_first=f1c7bb00c78885ad976322928466b740#rd

【论文解读】Chain Of Draft：LLM 少写多想更高效 - tomsheep的文章 - 知乎
https://zhuanlan.zhihu.com/p/28074420230

## Agent

Manus发布一天后迅速出现OpenManus、OWL 等复刻项目，怎么做到的？ - 锦恢的回答 - 知乎
https://www.zhihu.com/question/14321968965/answer/119732180420
> 学习 OpenManus，做成熟和工程化的 Agent。

李宏毅：从零开始搞懂 AI Agent - tomsheep的文章 - 知乎
https://zhuanlan.zhihu.com/p/29123783155

【论文解读】：START：让大模型学会「借工具思考」 - tomsheep的文章 - 知乎
https://zhuanlan.zhihu.com/p/28933816497

## Hack 和工程能力

长远看算法岗真的比开发岗香吗？ - 要没电了的回答 - 知乎
https://www.zhihu.com/question/409815271/answer/87375346326

抽象，简化和领域驱动设计 - 阿莱克西斯的文章 - 知乎
https://zhuanlan.zhihu.com/p/77026267