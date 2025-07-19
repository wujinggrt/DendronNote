---
id: 6lwuuvj2cy0cutd0lpmm84d
title: BT的实践_Nav2
desc: ''
updated: 1752904259324
created: 1750688523139
---

## 实践

### BehaviorTree.CPP

在 BehaviorTree.CPP 项目中，支持 Parallel 节点。Nav2 项目使用来增强实时响应性。

在 ROS2 Nav2 框架中，Parallel 节点常用于增强导航的实时响应性：

## Nav2

最大优势在于多线程和实时性的支持。

行为树本身不具体实现执行内容，只编排执行内容。比如 Navigation2，将执行内容实现放到各个 Server，行为树上的节点与 Server 通信，请求具体的执行内容，获得反馈。节点再根据反馈结果，另外请求。BT 负责制不同执行内容之间的跳转。也就是说，BT 中的节点，应当是 ROS2 中的各内容的客户端。

好的软件架构应当由如下特征：
- 模块化
- 组件有可复用性
- 组合
- 良好的隔离关注点

优势：
- 精妙地层次化
- 图像表达有清晰地含义
- 表达性更强

## 架构

BT 连接规划与实际执行。在 BT 中的节点，应当都是请求服务的子节点。这是因为 BT 会根据业务逻辑不断变化，所以抽象出 XML 部分。要易扩展，有很强的模块化属性，应对不断变化的规划需求。

如何落地到实际执行的模块？借鉴 Nav2 的架构设计，TreeNode 仅仅请求服务，维护状态。所以万变不离其宗，使用一些基础模块节点与实际执行的 Server 对应，将执行与行为解耦。

行为可以使用 XML 描述，应对灵活地扩展和变化。执行模块则根据描述，提供服务。如果将执行写到节点，不作解耦，那么应对灵活变化的业务逻辑（多变的规划结果），不停地写这些控制细节，是不现实的，要复用起来才现实。

## Ref and Tag

一文搞懂ROS2 Nav2：概念解析、流程拆分和源码编译安装的踩坑总结 - Sky Shaw的文章 - 知乎
https://zhuanlan.zhihu.com/p/1905585229109893073

- [Nav2 Docs](https://docs.nav2.org/getting_started/index.html)
- [Nav2 Behavior Trees](https://docs.nav2.org/concepts/index.html#behavior-trees)