---
id: y74wnh1oqjy8n1m8yosskg5
title: 内置函数和魔法方法_special_methods_下划线开头结尾的方法
desc: ''
updated: 1744905430682
created: 1743097733728
---

| Category | Method names |
| --- | ---------- |
| String/bytes representation | `__repr__ __str__ __format__ __bytes__ __fspath__` |
| Conversion to number | `__bool__ __complex__ __int__ __float__ __hash__  __index__` |
| Emulating collections | `__len__ __getitem__ __setitem__ __delitem__  __contains__` |
| Iteration | `__iter__ __aiter__ __next__ __anext__ __reversed__ `|
| Callable or coroutine execution | `__call__ __await__ `|
| Context management | `__enter__ __exit__ __aexit__ __aenter__` |
| Instance creation and destruction | `__new__ __init__ __del__` |
| Attribute management | `__getattr__ __getattribute__ __setattr__ __delattr__ __dir__ `|
| Attribute descriptors | `__get__ __set__ __delete__ __set_name__ `|
| Abstract base classes | `__instancecheck__ __subclasscheck__ `|
| Class metaprogramming | `__prepare__ __init_subclass__ __class_getitem__ __mro_entries__` |

## 各种实践

每个实例有 `__dict__` 属性，是一个字典，存储了一个实例的所有属性。

**获取实例对应 class 的名字**

```py
obj.__class__.__name__
```

## 内置函数

### 操作属性：getattr(), hasattr(), setattr()

利用反射机制动态获取对象的属性或方法。签名如下：

```py
getattr(object, name[, default]) # default 可选
```

但是，过度依赖此函数会导致代码难以维护。

## Ref and Tag