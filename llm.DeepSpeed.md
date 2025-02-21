---
id: qg6b9126chp8816hq8rj6h3
title: DeepSpeed
desc: ''
updated: 1740121608254
created: 1740121389360
---

从啥也不会到DeepSpeed————一篇大模型分布式训练的学习过程总结 - elihe的文章 - 知乎
https://zhuanlan.zhihu.com/p/688873027

## 为什么需要分布式训练？
主要有两点： 对小模型而言训练速度更快，对大模型而言，其所需内存太大，单机装不下。

## 分布式训练的加速
对于一些单卡可以装载的模型，我们可以通过多个卡数据并行的方法训练，把一轮数据算出来的梯度求和更新参数进行下一轮的梯度下降。这个范式比较经典的例子就是 Parameter Server，后续的章节会定量的介绍。

## Tag
#MLLM
#LLM
#Train