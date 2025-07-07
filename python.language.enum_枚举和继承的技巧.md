---
id: 9o9p0hcg1o44qsqku94wsh5
title: Enum_枚举和继承的技巧
desc: ''
updated: 1751692069985
created: 1751691822702
---

## IntEnum

### Overview

- `IntEnum` 是一个继承自 `int` 和 `Enum` 的混合类，它的成员既是整数，也是枚举。
- 成员的值必须是整数，并且可以直接与普通整数进行比较或运算。

为什么需要 IntEnum？

- 当需要枚举值既具有明确的语义（如枚举名），又能像整数一样参与计算时（例如与旧代码或需要整数的 API 兼容），`IntEnum` 非常有用。
- 避免直接使用“魔法数字”（Magic Numbers），提高代码可读性和可维护性。

### 例子
```python
from enum import IntEnum

class HttpStatus(IntEnum):
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    NOT_FOUND = 404

# 用法示例
print(HttpStatus.OK) # 输出: HttpStatus.OK
print(HttpStatus.OK == 200) # 输出: True (因为继承了整数特性)
print(HttpStatus.OK + 10) # 输出: 210 (可以直接参与整数运算)
# 反向查找
print(HttpStatus(200)) # 输出: HttpStatus.OK
```

---

#  auto()

- `auto()` 是一个辅助函数，用于自动为枚举成员分配值，无需手动指定。
- 默认情况下，它会生成递增的整数值（从 1 开始），但可以通过重写 `_generate_next_value_` 方法自定义行为。

场景：

- 当枚举成员的具体值不重要，只需确保唯一性时，`auto()`可以简化代码。
- 避免手动赋值错误（如重复值或遗漏值）。

## 例子

```python
from enum import IntEnum, auto

class Priority(IntEnum):
    LOW = auto()    # 1
    MEDIUM = auto() # 2
    HIGH = auto()   # 3

print(Priority.LOW < Priority.HIGH) # 输出: True (因为值是整数)
```

## Ref and Tag