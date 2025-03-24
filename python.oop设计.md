---
id: pxx5142wg1zx96shb8f0wzu
title: Oop设计
desc: ''
updated: 1742834340803
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



## Ref and Tag