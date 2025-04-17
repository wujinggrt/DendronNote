---
id: 334c5omhplcsh8qjn9js75a
title: 高性能方式_减少用pow
desc: ''
updated: 1744815548377
created: 1744815092521
---

求高阶幂的时候，直接用 pow(x, n) 会慢很多。但是使用 exp(n * log(x)) 组合起来会快很多。exp 的底层实现常采用泰勒级数展开或二进制分解。

## Ref and Tag

c++写高性能计算有什么心得体会? - 叶飞影的回答 - 知乎
https://www.zhihu.com/question/662526033/answer/3582379983