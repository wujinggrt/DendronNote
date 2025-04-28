---
id: npte0906jkw89zpf0feniix
title: 装饰器_decorator
desc: ''
updated: 1745781059619
created: 1745779833169
---

## 用法

```py
    def timer(func): 
        def wrapper(*args, **kwargs): 
            st = time.perf_counter() 
            ret = func(*args, **kwargs) 
            if print_args: 
                print(f'"{func.__name__}", args: {args}, kwargs: {kwargs}') 
            print('time cost: {} seconds'.format(time.perf_counter() - st)) 
            return ret 
        return wrapper 
```

用法：

```py
@timer
def random_sleep(): ...
```

### 接收参数的装饰器写法

不接收参数的，套一层即可。如果装饰器需要接收参数，需要再嵌套一层。

```py
def timer(print_args=False):
     """装饰器：打印函数耗时 
    :param print_args: 是否打印方法名和参数，默认为 False 
    """ 
    def decorator(func): 
        def wrapper(*args, **kwargs): 
            st = time.perf_counter() 
            ret = func(*args, **kwargs) 
            if print_args: 
                print(f'"{func.__name__}", args: {args}, kwargs: {kwargs}') 
            print('time cost: {} seconds'.format(time.perf_counter() - st)) 
            return ret 
        return wrapper 
    return decorator 
```

使用如下：

```py
@timer(print_args=True) 
def random_sleep(): ...
```

相比不带参数版本，可以理解为 timer(print_args=True) 获取内部装饰器，内部的装饰器 decorator 捕获参数 print_args 后，再作用于 random_sleep() 函数。归根结底，回到了封装一层函数的情况。

同上：

```py
_decorator = timer(print_args=True) ➊ 
random_sleep = _decorator(random_sleep) ➋ 
```

不仅函数可以当作装饰器，可调用的对象（实现了 `__call__` 的对象）也可。有参数装饰器一共得提供两次函数调用，通过类实现的装饰器，其实就是把原本的两次函数调用替换成了类和类实例的调用。 

```py
class timer: 
    """装饰器：打印函数耗时 
    :param print_args: 是否打印方法名和参数，默认为 False 
    """ 
    def __init__(self, print_args): 
        self.print_args = print_args 
    def __call__(self, func): 
        @wraps(func) 
        def decorated(*args, **kwargs): 
            st = time.perf_counter() 
            ret = func(*args, **kwargs) 
            if self.print_args: 
                print(f'"{func.__name__}", args: {args}, kwargs: {kwargs}') 
            print('time cost: {} seconds'.format(time.perf_counter() - st)) 
            return ret 
    return decorated 
```

1. 第一次调用：_deco = timer(print_args=True) 实际上是在初始化一个 timer 的实例。 
2. 第二次调用：func = _deco(func) 是在调用 timer 实例，触发 `__call__` 方法。

wrapt 能够分辨函数（function）与方法（method），在处理被装饰的对象时，函数和方法有着不同的影响。方法的第一个参数总是self，会被装饰器当作第一个参数处理，从而出现问题。

装饰器如果不指定参数时，使用时不写括号，应当把除了 func 参数以外的参数，全部定义为关键字参数。

```py
def delayed_start(func=None, *, duration=1):
    ...

# 1. 不提供任何参数 
@delayed_start 
def hello(): ... 
# 2. 提供可选的关键字参数 
@delayed_start(duration=2) 
def hello(): ... 
# 3. 提供括号调用，但不提供任何参数 
@delayed_start() 
def hello(): ... 
```

## Ref and Tag