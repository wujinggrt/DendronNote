---
id: 8kpps2a3lx91jkf2ogqlsrf
title: 设计模式
desc: ''
updated: 1753375022620
created: 1753374831565
---

## 单例模式

CPP 方式，参考 [Avoid singletons](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Ri-singleton)

单例不过是全局对象的复杂形式。如果想要不被修改，可以使用 const 或 constexpr。除非是想要推迟初始化，可以不用全局变量的方式。

## Ref and Tag