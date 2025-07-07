---
id: tpvklnoiijzxtl3xhe6wa7u
title: Dataclasses
desc: ''
updated: 1751719637620
created: 1751690051816
---

# Python `@dataclass` 装饰器详解

`@dataclass` 是 Python 3.7+ 引入的一个装饰器，用于自动生成类的特殊方法（如 `__init__` 和 `__repr__`），极大地简化了数据类的编写，减少了样板代码。

## 自动生成的方法

使用 `@dataclass` 装饰的类会自动生成以下方法（根据配置参数不同可能有变化）：

*   `__init__`: 构造函数，根据类属性自动初始化对象。
*   `__repr__`: 提供清晰的字符串表示（如 `ClassName(attr1=value1, ...)`）。
*   `__eq__`: 按属性值比较对象是否相等。
*   `__hash__` (可选): 生成哈希值（需配置）。
*   比较方法 (如 `__lt__`, `__le__` 等): 支持对象排序（需设置 `order=True`）。

## 传统方式 vs. `@dataclass`

下面通过一个简单的 `Point` 类来对比两种写法的差异。

### 传统方式

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
        
    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y
```

### 使用 `@dataclass`

代码更简洁，意图更清晰。

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int = 3

    def show(self):
        print(f"x: {self.x}, y: {self.y}")

# 使用方法完全一样
p0 = Point(32, 24)
print(p0)  # 输出: Point(x=32, y=24)
p1 = Point(32) # 有 p1.x 32, p1.y 3，但是 Point 类没有 x 属性，仅有 y 属性 3
```

注意，传入参数时改变的是实例的 x 和 y 字段，值为 32 和 24；因为提供了默认值，所以类 Point 本身也有 y 字段，值为 3，但是没有 x 属性。

## 常用配置参数

通过在装饰器中传递参数，可以精细控制生成的方法和行为：

*   `init=True`：生成 `__init__` 方法（默认启用）。
*   `repr=True`：生成 `__repr__` 方法（默认启用）。
*   `eq=True`：生成 `__eq__` 方法（默认启用）。
*   `order=False`：若设为 `True`，则生成比较方法（如 `<`、`>=` 等），使对象可排序。
*   `frozen=False`：若设为 `True`，实例属性将不可修改（创建后即“冻结”），使其成为不可变对象。
*   `slots=False`：若设为 `True`，将使用 `__slots__` 优化内存（需要 Python 3.10+）。

## 高级用法

### 1. 默认值与字段配置

`dataclasses` 模块提供了强大的字段配置功能。

*   **默认值**：直接为字段赋值即可设置简单的默认值。
*   **`default_factory`**：用于生成可变类型的默认值（如列表、字典），避免所有实例共享同一个可变对象的陷阱。
*   **`field()` 函数**：用于对字段进行更精细的控制，例如设置其是否参与构造函数、比较或哈希计算。

```python
from dataclasses import dataclass, field

@dataclass
class User:
    name: str
    age: int = 18  # 直接设置默认值
    friends: list[str] = field(default_factory=list)  # 使用工厂函数避免共享可变默认值
    id: int = field(init=False, repr=False, default=0) # 不参与构造函数，也不在 repr 中显示
```

### 2. 后初始化处理 `__post_init__`

如果需要在对象初始化（`__init__` 被调用）之后执行额外的逻辑（例如，计算派生属性），可以定义 `__post_init__` 方法。

```python
from dataclasses import dataclass, field

@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)  # area 不通过构造函数初始化

    def __post_init__(self):
        """在初始化完成后自动调用"""
        if self.width < 0 or self.height < 0:
            raise ValueError("Width and height must be non-negative")
        self.area = self.width * self.height  # 计算派生属性
```

## 适用场景

*   **数据容器**：非常适合用作存储结构化数据的容器，如 API 响应、配置信息、数据库记录（DTO）。
*   **替代命名元组（NamedTuple）**：当需要一个类似元组但又希望其属性可变时，`dataclass` 是更灵活的选择。
*   **减少重复代码**：自动生成常用方法，让开发者专注于核心业务逻辑，提高代码的可读性和可维护性。

## 实例：简化旅客记录类

### 问题背景

定义一个 `VisitRecord` 类来记录旅客信息。当两条记录的姓名和电话号码相同时，我们判定这两条记录相等，以便于在集合中去重。

### 传统实现

```python
class VisitRecord:
    """旅客记录
    - 当两条旅客记录的姓名与电话号码相同时，判定二者相等。
    """
    def __init__(self, first_name, last_name, phone_number, date_visited):
        self.first_name = first_name
        self.last_name = last_name
        self.phone_number = phone_number
        self.date_visited = date_visited

    @property
    def comparable_fields(self):
        """获取用于比较对象的字段元组"""
        return (self.first_name, self.last_name, self.phone_number)

    def __hash__(self):
        return hash(self.comparable_fields)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.comparable_fields == other.comparable_fields
        return False

# 示例用法
# users_visited_puket = [...]  # 字典列表
# users_visited_nz = [...]     # 字典列表
#
# def find_potential_customers_v3():
#     # 转换为 VisitRecord 对象后计算集合差值
#     return set(VisitRecord(**r) for r in users_visited_puket) - set(
#         VisitRecord(**r) for r in users_visited_nz
#     )
```

### 使用 `@dataclass` 简化

使用 `@dataclass` 后，代码变得极为简洁。我们只需声明字段和类型，并用 `field()` 函数微调比较行为。

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class VisitRecordDC:
    first_name: str
    last_name: str
    phone_number: str
    # 旅游日期不参与比较
    date_visited: str = field(compare=False)

# 示例用法
# users_visited_puket = [...]  # 字典列表
# users_visited_nz = [...]     # 字典列表
#
# def find_potential_customers_v4():
#     return set(VisitRecordDC(**r) for r in users_visited_puket) - set(
#         VisitRecordDC(**r) for r in users_visited_nz
#     )
```

**代码解析**：
1.  我们不再需要手动实现 `__init__`, `__eq__`, 和 `__hash__` 方法，`@dataclass` 会自动完成。
2.  通过 `field(compare=False)`，我们告诉 `dataclass` 在生成 `__eq__` 和 `__hash__` 方法时跳过 `date_visited` 字段。
3.  默认情况下，数据类是可变的，因此不支持哈希操作（因为哈希值要求对象不可变）。通过设置 `frozen=True`，我们将实例变为**不可修改的**，`dataclass` 会自动为其生成 `__hash__` 方法，使其可以被添加到集合（`set`）或用作字典的键。

## Ref and Tag