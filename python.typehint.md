---
id: ei5qywn058edm6ypiqrl78b
title: Typehint
desc: ''
updated: 1742976054689
created: 1742975485079
---

优先使用新的语法，而非 typing.List 等。

`Optional[int]` 代表int or None。但 3.9 之后可以用 or 操作符：

```py
def get_username() -> str | None:
    return "John Doe" if logged_in else None
```

`Uion[int, str]` 在 3.10+ 可以用 or 操作符：

```py
def display_value(value: int | str):
    print(value)
```

指定变量只能为一个或几个值：

```py
from typing import Literal

def toggle_status(status: Literal["on", "off"]):
    print(f"Status set to {status}")

type Fruit = Literal["apple", "pear", "banana"]
def show(f: Fruit):
    print(f)
```

## Ref and Tag

Python Type Hints 简明教程（基于 Python 3.13） - Snowflyt的文章 - 知乎
https://zhuanlan.zhihu.com/p/464979921