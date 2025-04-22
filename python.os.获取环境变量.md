---
id: 8e76tdns4gf7xhgrl4upggf
title: 获取环境变量
desc: ''
updated: 1745341395888
created: 1745341372635
---

## 获取环境变量

os.environ 是一个字典对象，包含了当前进程的所有环境变量。可以通过 os.environ['变量名'] 来获取特定的环境变量。

```py
import os

print(os.environ['PATH'])
```

## Ref and Tag