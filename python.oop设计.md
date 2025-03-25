---
id: pxx5142wg1zx96shb8f0wzu
title: Oop设计
desc: ''
updated: 1742919889402
created: 1742834169901
---

## _instance: 命名约定的单例模式

许多编程情况下，_instance 或其他类似的命名约定并不是 Python 语法的一部分，而是开发者的习惯用法，通常用作私有属性，以存储实例化对象。例如，实现单例模式时，_instance用于确保一个类只能有一个实例：

```py
class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]
    
singleton1 = LLM()
singleton2 = LLM()
print(singleton1 is singleton2)  # 输出: True，证明两个变量指向同一个实例
```

_instance 存储了类 Singleton 的唯一实例。无论创建多少个 Singleton 的实例，返回的都将是同一个对象‌。

## typehint

参数优先使用 typing 的 Optional, List, Literal，除了字典，依然用 dict，但是不指定具体类型。

返回类型不用使用 Optional，比如 str | None 即可。

比如：

```py
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
    ...
```

## 多个返回值时，并且可能为 None 时，优先用字典

比如：

```py
def process(...) -> tuple[np.ndarray, str] | tuple[None, None]:
    ...
    if ...:
        return None, None
res1, res2 = process(...)
```

返回 None, None 的形式看起来比较奇怪。但是为了保证外部处理的代码的形式有 res1, res2，需要写成 tuple[None, None]。

应该封装返回值为一个 class XXResult 

## 

## Ref and Tag