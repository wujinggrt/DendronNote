---
id: le2m986q3xci6qkx9vgqxbl
title: 对比PyTorch
desc: ''
updated: 1753627917203
created: 1751382982900
---

## 从 PyTorch 走向 Jax

Jax 就像函数式版本的 PyTorch，Jax 通常处理无状态的函数。

PyTorch 有六个核心 API：

```py
model = Model(arg_model) # 1. 模型初始化
opt = Optimizer(arg_opt, model.parameters()) # 2. 优化器初始化
y = model(x) # 3. 模型计算
loss = loss_f(y, target) # 4. 损失函数计算
loss.backward() # 5. 反向传播
opt.step() # 6. 优化器更新参数
```

接下来，需要将有状态的函数转化为无状态函数。首先，理清 PyTorch 内部状态有什么。

### 1. 模型初始化

PyTorch 的模型初始化中，model 实例保存了模型的参数值。对应到 Jax 中，应该为 `model, params = Model(arg_model)`，但是通常写作 `model = Model(arg_model); params = model.init()`，而 Model 通常是 Jax 的神经网络模块库 flax。

PyTorch 有全局共享的随机数种子，但是 Jax 没有真正的随机数，通常明确写出，比如 model.init(key)，key 就是种子。

```py
model = Model(arg_model)
key = jax.random.key(0)
params = model.init(key)
```

### 2. 优化器初始化

变量 opt 保存梯度等状态，所以，Jax 写作 

```py
opt = Optimizer(arg_opt)
state = opt.init(params)`。
```

### 3. 模型计算

Jax 中，模型计算写作如下，params 是模型参数。

```py
y = model.apply(params, x)
```

### 4. 损失函数计算

因为损失函数不包含状态，所以与 PyTorch 一样，y 是模型输出，target 是目标值。

```py
loss = loss_f(y, target)
```

### 5. 反向传播

PyTorch 中，反向传播由张量的方法完成，即 `loss.backward()`，根据动态生成的计算图，计算梯度。

Jax 中，贯彻 stateless 风格的情况下，需要输入模型参数 `params` 和 输入 `x`，得到各个 `params` 对应的梯度。由于框架都是自动微分的，所以只需要写损失函数给框架即可，其余交给框架完成。最终如下：

```py
def loss_func(params, x, target):
    y = model.apply(params, x)
    return loss_f(y, target)
loss, grads = jax.value_and_grad(loss_func)(params, x, target)
```

看起来与 PyTorch 的思路差别明显，由 value and grad 直接负责。

### 6. 优化器更新参数

PyTorch 的优化器更新参数由 `opt.step()` 完成。而 Jax 中，考虑 stateless 场景，由两步完成：

```py
updates, opt_state = opt.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

其他的细节，比如函数的 forward 有随机性，比如 Dropout 内容，比如初始化参数的细节，通过阅读 flax 文档了解。

optax 是 Jax 的优化器的库。

## flax 库：对标 torch.nn

Jax 的函数式编程尽可能回避有状态的情况，所以初始基础类会有所不同。flax 中，新建模块不同于 torch.nn.Module 直接继承并在构造函数初始化，我们不能实现 `__init__()` 函数。继承的方式是通过类型标注，增加新的配置参数，比如：

```py
import flax
import typing

class MyMod(flax.linen.Module):
    # 假设需要一个参数 arg_model
    arg_model: typing.Any
```

可以看到，仅能够知道类型。继承的时候，flax 为 flax.linen.Module 注册了 `__init_subclass__` 函数，继承得到新类时，这个新的 class 会发生巨大变化。这些由元编程内容完成。

所有继承 flax.linen.Module 的 class 自动变为 dataclass 类型：

```py
import dataclasses
dataclasses.is_dataclass(MyMod) # True
```

初始化函数 `__init__` 会修改为接受三个参数的函数。开始的参数即自定义的参数，紧跟 parent 和 name。

```py
MyMod.__init__(
    self,
    arg_model: typing.Any,
    parent: Union[Type[flax.linen.module.Module], flax.core.scope.Scope, Type[flax.linen.module._Sentinel], NoneType] = <flax.linen.module._Sentinel object at 0x11a166d40>,
    name: Optional[str] = None,
) -> None:
...
```

## linen: 传统的 Module 库，未来被 nnx 替代

### 定义参数

#### 自动注册参数：装饰器 compact

`@flax.linen.compact` 是核心装饰器，简化模块的定义，自动注册和管理参数、模块等：
- 所有 `self.xxx` 形式的变量，都会自动注册为参数。就像 torch 的 register_buffer 一样。
- 这些参数可以通过 `params = model.init(key, *args)` 初始化。key 是随机数生成器的 key。
使用 compact 自动定义（推荐）：

```py
class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        for _ in range(3):
            # 参数自动共享同一组参数
            x = nn.Dense(128)(x)
    return x
```

#### 在 setup() 中定义参数

**参数共享问题**：比如，3 次循环使用了相同的参数和模块 nn.Dense(128)。如果需要避免，比如 MLP 的场景，应当用 setup 定义。

```py
class Model(nn.Module):
    # 无 compact，需显示注册
    def setup(self):
        self.dense1 = nn.Dense(128)
        self.dense2 = nn.Dense(10)

    def __call__(self, x):
        x = self.dense1(x)
        return self.dense2(x)
```

compact 装饰器与 setup 混用时，特别要注意所有名称唯一，避免冲突。

#### 在 param() 方法定义参数

**self.param()** 定义的参数，在每次循环中，都指向同一个变量。通常用于在 `__call__` 方法中定义，配合 compact 装饰器使用。

```py
self.param(name: str, init_fn: Callable[..., Any], *init_args, **init_kwargs)
```

- `name`: str: 必需参数。这是参数的唯一名称，在当前模块的作用域（scope）内必须是唯一的。例如 'kernel' 或 'bias'。Flax 使用这个名字来组织和识别参数。
- `init_fn`: Callable: 必需参数。这是一个可调用对象（通常是一个函数），用于生成参数的初始值。它会接收参数的 shape 和 dtype 作为输入。Flax 在 flax.linen.initializers 中提供了许多常用的初始化器，例如：
    - nn.initializers.lecun_normal(): 经典的权重初始化器。
    - nn.initializers.zeros: 初始化为全零（常用于偏置）。
    - nn.initializers.ones: 初始化为全一。
    - nn.initializers.xavier_uniform(): Xavier/Glorot 均匀分布初始化。
    - 也可以提供任何自定义的函数，只要它遵循 (key, shape, dtype) -> Array 的签名。
- `init_args` 和 `init_kwargs`: 传递给 init_fn 的额外参数。在绝大多数情况下，你只需要提供参数的 shape 和 dtype。Flax 的设计使得这些参数会自动被捕获和传递，你通常不需要手动处理。最重要的隐式参数是 shape。
    - `shape`: 张量的形状，决定参数的 shape
    - `dtype`: 数据类型。

### jax.random.key

用于生成伪随机数生成器密钥的核心函数。

## nnx: 替代 linen API

与之前的 Flax-Linen 或 Haiku 中的 Module 不同，nnx.Module中一切都是 explicit 的，这意味着：
- Stateful modules: 参数直接保存在 modules，类似 PyTorch，不再拆分到独立的变量
- Simpler mental model: 不再需要 init/apply 来独立地初始化
- JAX compatibility: 在 jit/grad 中，自动处理状态分离
- Mutable during setup, frozen after initialization

例子可以看出很像 torch 了：

```py
class Linear(nnx.Module):

  def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
    self.b = nnx.Param(np.zeros((dout,)))
    self.din, self.dout = din, dout

  def __call__(self, x):
    return x @ self.w + self.b

class Model(nnx.Module):
    def __init__(self, dim):
        self.linear = nnx.Linear(dim, dim)
        self.bn = nnx.BatchNorm()
    
    def __call__(self, x):
        return self.bn(self.linear(x))

model = Model(512)  # Parameters live inside the module
```

动态的状态通常保存在 nnx.Param 中，静态变量则直接保存。jax.Array和 np.ndarray的属性也被视作动态状态，最好将之存储在 nnx.Variable中，如 nnx.Param。

### nnx.Rngs: 新一代随机数生成器

比如，`rngs = nnx.Rngs(42)` 代表设置种子为 42 的随机数。通常传递给 dropout 等部分。nnx.Rngs 能够自动分裂密钥，而 Linen 要手动调用 jax.random.split。

### struct

JAX-compatible dataclasses：

- Core tool: `@struct.dataclass`，创造 immutable classes that work with JAX xformations
- Pytree registration: 与 jax.tree_map 自动管理

例子：

```py
from flax import struct

@struct.dataclass
class TrainingState:
    params: dict
    step: int
    rng: jax.Array

state = TrainingState(params={}, step=0, rng=jax.random.key(0))
new_state = state.replace(step=1)  # Returns new instance
```

### traverse_util

处理嵌套的参数结构。

- Core utilities:
    - flatten_dict: 

### nnx.bridge

迁移 linen 到新的 nnx API 的工具，能够无缝转换。

linen -> nnx:

```py
from flax import linen as nn
from flax.nnx import bridge

# Define linen model
class LinenModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(128)(x)

# Initialize linen
model_linen = LinenModel()
variables = model_linen.init(jax.random.key(0), jnp.ones((1, 64))

# Convert to nnx
model_nnx = bridge.from_linen(
    model_linen, 
    variables
)
```

nnx -> linen:

```py
from flax.nnx import Module, Linear

# Define nnx model
class NNXModel(Module):
    def __init__(self, rngs):
        self.dense = Linear(64, 128, rngs=rngs)

    def __call__(self, x):
        return self.dense(x)

# Initialize nnx
model_nnx = NNXModel(rngs=nnx.Rngs(0))

# Convert to linen
model_linen, variables = bridge.to_linen(model_nnx)
```

## Ref and Tag

一文打通PyTorch与JAX - 游凯超的文章 - 知乎
https://zhuanlan.zhihu.com/p/660342914

https://flax.readthedocs.io/en/latest/index.html