---
id: ei5qywn058edm6ypiqrl78b
title: Typehint_类型提示
desc: ''
updated: 1743350861621
created: 1742975485079
---

## typing

优先使用新的语法，而非 typing.List 等。

### Optional 和 |，3.9+

`Optional[int]` 代表int or None。但 3.9 之后可以用 or 操作符：

```py
def get_username() -> str | None:
    return "John Doe" if logged_in else None
```

### Union, |, 3.10+

`Uion[int, str]` 在 3.10+ 可以用 or 操作符：

```py
def display_value(value: int | str):
    print(value)
```

### Literal: py3.8+

指定变量只能为一个或几个值：

```py
from typing import Literal

def toggle_status(status: Literal["on", "off"]):
    print(f"Status set to {status}")

type Fruit = Literal["apple", "pear", "banana"]
def show(f: Fruit):
    print(f)
```

Literal 只能包含明确的字面量值，不能是变量或表达式：

```py
# 错误用法
ALLOWED = ("yes", "no")
param: Literal[ALLOWED]  # 类型检查器无法识别
```

## 返回类型有双引号的情况

双引号包围的类型，通常由于前向引用 (Forward Reference) 和字符串字面量类型 (String Literal Types) 引起的。

### 前向引用

类型提示是尚未定义的类，那么用双引号括起来，即字符串字面量类型，避免运行错误。常用于 `@classmethod` 返回自身类型的实例：

```py
    @classmethod
    def user_message(
        cls, content: str, base64_image: Optional[str] = None
    ) -> "Message":
        """Create a user message"""
        return cls(role=Role.USER, content=content, base64_image=base64_image)
```

### 解决循环依赖

两个类互相引用对方的类型，必须用引号包围其中一个类型：

```py
class A:
    def get_b(self) -> "B":  # 引号包围 B
        return B()

class B:
    def get_a(self) -> A:   # A 已经定义，无需引号
        return A()
```

## Ref and Tag

Python Type Hints 简明教程（基于 Python 3.13） - Snowflyt的文章 - 知乎
https://zhuanlan.zhihu.com/p/464979921