---
id: 9bpb9k5cruiab4k5wejao7y
title: 变量和脚本
desc: ''
updated: 1740548248496
created: 1740547398352
---

使用 `:echo <str>` 可以打印信息，帮助调试。但是 VsCode 的 vim 插件没有实现 echom 和 messages。

## 设置变量的值

`:let var="hello"` 随后打印 `:echo var`。作为变量，推荐使用全局作用域，比如 `let g:var = 2`

## Ref and Tag

#Vim