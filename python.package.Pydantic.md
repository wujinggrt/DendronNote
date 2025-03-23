---
id: qtr5ed8hxwnpk8zl3u49l1f
title: Pydantic
desc: ''
updated: 1742752272545
created: 1742729166237
---

Pydantic 提供了数据验证工具，现代项目经常使用它。核心逻辑由 Rust 编写，能快速验证数据。安装使用 `pip install pydantic`。

## BaseModel

具有许多可通过实例方法使用的常用实用程序的 Pydantic 自身的超类。更多用法在实战中，看 OpenManus 等库中学习。

```py
from datetime import datetime
from pydantic import BaseModel, PositiveInt
from pydantic import ValidationError

class User(BaseModel):
    id: int  
    name: str = 'John Doe'  
    signup_ts: datetime | None  
    tastes: dict[str, PositiveInt]  

external_data = {
    'id': 123,
    'signup_ts': '2019-06-01 12:22',  
    'tastes': {
        'wine': 9,
        b'cheese': 7,  
        'cabbage': '1',  
    },
}

user = User(**external_data)  

print(user.id)  
#> 123
print(user.model_dump())  
"""
{
    'id': 123,
    'name': 'John Doe',
    'signup_ts': datetime.datetime(2019, 6, 1, 12, 22),
    'tastes': {'wine': 9, 'cheese': 7, 'cabbage': 1},
}
"""

external_data = {'id': 'not an int', 'tastes': {}}  

try:
    User(**external_data)  
except ValidationError as e:
    print(e.errors())
    """
    [
        {
            'type': 'int_parsing',
            'loc': ('id',),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': 'not an int',
            'url': 'https://pydantic.com.cn/errors/validation_errors#int_parsing',
        },
        {
            'type': 'missing',
            'loc': ('signup_ts',),
            'msg': 'Field required',
            'input': {'id': 'not an int', 'tastes': {}},
            'url': 'https://pydantic.com.cn/errors/validation_errors#missing',
        },
    ]
    """
```

### 使用类型提示限制输入数据

```py
from typing import Annotated, Dict, List, Literal, Tuple
from annotated_types import Gt
from pydantic import BaseModel

class Fruit(BaseModel):
    name: str  
    color: Literal['red', 'green']  
    weight: Annotated[float, Gt(0)]  # Greater than 0
    bazam: Dict[str, List[Tuple[int, bool, float]]]  

print(
    Fruit(
        name='Apple',
        color='red',
        weight=4.2,
        bazam={'foobar': [(1, True, 0.1)]},
    )
)
#> name='Apple' color='red' weight=4.2 bazam={'foobar': [(1, True, 0.1)]}
```

当需要限制的类型不再 typing 库的提示中，比如 Gt(0)，我们可以使用 Annotated 处理，如 Fruit.weight。

类型提示还可以是 typing 的 Optional。

## Field

用于精细化控制字段行为的工具，允许在模型类中为每个字段定义额外的约束、元数据或验证规则。它通过 pydantic.Field 函数实现，常用于配置默认值、数据验证、文档生成等场景。例子如下：

```py
from pydantic import BaseModel, Field

class User(BaseModel):
    # 使用 Field 定义字段
    name: str = Field(..., min_length=1, max_length=50)  # 必填字段，限制长度
    age: int = Field(default=18, gt=0, le=150)           # 默认值 + 范围限制
    email: str | None = Field(
        None,
        example="user@example.com",  # 在文档中展示示例值
        regex=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"  # 正则校验
    )
```

### 常用参数

- default: 传入默认值。如果是 `...`，代表无默认值，必须显式提供。 
- default_factory: 动态生成默认值，比如 `special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])`。

数据验证
- default: 字段的默认值（若字段非必填）。
- gt / ge: 数值大于（gt）或大于等于（ge）。
- lt / le: 数值小于（lt）或小于等于（le）。
- min_length / max_length: 字符串、列表等类型的长度限制。
- regex: 用正则表达式验证字符串格式。
- allow_inf_nan: 是否允许 inf 或 nan（默认为 True，浮点型字段适用）。

元数据与文档
- alias: 字段的别名（用于解析/序列化时映射不同名称，如 JSON 键名）。
- title: 字段的标题（用于生成文档）。
- description: 字段的详细描述。
- example: 在文档中展示的示例值（如 OpenAPI 文档）。
- deprecated: 标记字段已弃用。

其他
- const: 强制字段必须等于某个固定值（类似常量）。
- repr: 是否在模型的 __repr__ 中显示该字段（默认为 True）。
- exclude: 在导出模型（如 .dict()）时排除此字段。


避免在 Field 中定义过于复杂的逻辑，必要时使用 `@validator` 分离校验逻辑。 优先使用 Python 原生语法：比如简单的默认值（如 age: int = 18）无需使用 Field，直接用赋值语法更简洁。

### 序列化

可以使用 .json() 方法序列化。

## Ref and Tag

https://pydantic.com.cn/