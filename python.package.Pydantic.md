---
id: qtr5ed8hxwnpk8zl3u49l1f
title: Pydantic
desc: ''
updated: 1742734026141
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

### 序列化

可以使用 .json() 方法序列化。

## Ref and Tag

https://pydantic.com.cn/