---
id: ixly6q8plrbe5xijsmg26uv
title: 解耦依赖混乱_提取到上一层
desc: ''
updated: 1752551613273
created: 1743354325718
---

设计两个类的过程中，发现两个类都依赖同一个类或对象，那么应当以依赖注入的方式，把依赖的部分单独提取处理，避免使用混乱和冗余。

比如，RobotProprioception 和 RobotController 都依赖于一个远程的机器人服务 self.socket，那么封装一个 RobotConnection，提取依赖部分。

## OCP

开发新功能，应当不要修改既有代码。修改意味着可能带来 Bug，需要重新测试功能是否完备，且引入新的耦合。

## Ref and Tag