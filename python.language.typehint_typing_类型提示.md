---
id: ei5qywn058edm6ypiqrl78b
title: Typehint_typing_类型提示
desc: ''
updated: 1751689426399
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

## TypeVar 与 Generic：泛型

- Generic：基类，用于定义泛型类
- TypeVar：创建类型变量（占位符）

```py
from typing import Generic, TypeVar, List

# 创建类型变量（T 和 U 是占位符）
T = TypeVar('T')  # 任意类型
U = TypeVar('U')  # 另一个任意类型
```

创建：

```py
class Box(Generic[T]):
    def __init__(self, content: T):
        self.content = content
    
    def get_content(self) -> T:
        return self.content

    def replace(self, new_content: T) -> None:
        self.content = new_content

# 自动推断为 Box[int]
int_box = Box(123)
print(int_box.get_content())  # 123

# 显式指定类型为 Box[str]
str_box: Box[str] = Box("Hello")
str_box.replace("World")  # ✅ 合法
str_box.replace(100)      # ❌ 类型检查会报错
```

### TypeVar: 约束泛型允许的范围

```py
# 只允许 int 或 str 类型
NumOrStr = TypeVar('NumOrStr', int, str)

class Container(Generic[NumOrStr]):
    ...

Container(10)    # ✅ 合法
Container("ok")  # ✅ 合法
Container([])    # ❌ 类型检查报错
```

**参数**：
- covariant 和 contravariant: 协变和逆变，处理泛型子类型关系。
    - 协变(covariant=True)：用于只读容器（生产者）。
    - 逆变(contravariant=True)：用于只写容器（消费者）
- invariant: 不变
- bound: 边界，限制类型必须是指定类的子类

#### covariant

子类型实际保持不变，就像 C++ 的多态中，子类的引用或指针赋给父类。

```py
class Animal: pass
class Dog(Animal): pass

T_co = TypeVar('T_co', covariant=True)

class ReadOnlyBox(Generic[T_co]):
    def get(self) -> T_co: ...

box_dog: ReadOnlyBox[Dog] = ReadOnlyBox()
box_animal: ReadOnlyBox[Animal] = box_dog  # ✅ 协变允许
```

#### contravariant

逆变则允许从父类变为子类。

```py
T_contra = TypeVar('T_contra', contravariant=True)

class WriteOnlyBox(Generic[T_contra]):
    def put(self, item: T_contra) -> None: ...

box_animal: WriteOnlyBox[Animal] = WriteOnlyBox()
box_dog: WriteOnlyBox[Dog] = box_animal  # ✅ 逆变允许
```

#### 类型边界：Bound

要求类型必须是某类的子类：

```py
from typing import Protocol

class Drawable(Protocol):
    # 约束的格式
    def draw(self) -> None: ...

# 可以直接使用，也可以用 TypeVar 定义
DrawableType = TypeVar('DrawableType', bound=Drawable)

class Canvas(Generic[DrawableType]):
    def __init__(self, item: DrawableType):
        self.item = item
    
    def render(self):
        self.item.draw()  # 保证所有类型都有 draw 方法
```

### TypeAlias 和 case：别名与转换

类型别名定义 和类型转换/类型提示。

TypeAlias 用于为复杂类型定义别名，提升代码可读性。但需注意：
- 旧版本（Python ≤ 3.11） ：需通过 typing.TypeAlias 显式声明 。
- 新版本（Python ≥ 3.12） ：直接使用 type 关键字定义类型别名（TypeAlias 已弃用）

```py
# 旧版本写法（Python ≤ 3.11）
from typing import TypeAlias
Factors: TypeAlias = list[int]  # 定义类型别名

# 新版本写法（Python ≥ 3.12）
type Factors = list[int]  # 直接使用 type 关键字
```

cast 来自 typing 模块，用于显式告诉类型检查器 某个变量的类型，但不会实际改变其运行时类型。

```py
from typing import cast

a = "123"
b = cast(int, a)  # 告诉类型检查器 a 是 int 类型
print(b + 1)  # 运行时仍可能报错（因为 a 实际是 str），需开发者自行保证类型安全
```

## SupportsIndex

支持 `__index__()` 方法的类型，用于索引转换。通常用于 Dataset。

## runtime_checkable: 运行时检查的装饰器

支持 isinstance/issubclass 检查。

```py
@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...
```

静态检查时，必须要实现了 create，才能有 isinstance(obj, LRScheduleConfig) 为 True。

## Ref and Tag

Python Type Hints 简明教程（基于 Python 3.13） - Snowflyt的文章 - 知乎
https://zhuanlan.zhihu.com/p/464979921