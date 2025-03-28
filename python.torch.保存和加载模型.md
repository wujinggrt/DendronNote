---
id: edll7o8gebsag50rvjmls0i
title: 保存和加载模型
desc: ''
updated: 1743141246767
created: 1742751626701
---

在PyTorch中，当你训练好一个模型后，通常有两种方式来保存你的工作：
1. 保存整个模型（Model）：
   - 这种方式会保存模型的架构和所有参数。
   - 使用 `torch.save()` 函数，例如：`torch.save(model.state_dict(), PATH)`。
   - 这种方法保存的是模型的序列化表示，便于分享和部署。
2. 保存检查点（Checkpoint），仅保存参数，推荐方法，并且仅讨论使用此方法的例子：
   - 检查点不仅包含模型的参数，还可以包含优化器状态、训练轮次、最佳验证指标等训练过程中的额外信息。
   - 保存检查点通常使用 `torch.save` 来保存一个包含多个组件的字典，例如： `torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'best_val_loss': best_val_loss}, PATH)`。
   - 这种方式在进行长时间训练或需要从中断的地方恢复训练时非常有用。


## torch.save()

```py
# torch
def save(
    obj: object,
    f: FILE_LIKE,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
) -> None:
    """...
    Args:
        obj: saved object
        f: a file-like object (has to implement write and flush) or a string or
           os.PathLike object containing a file name
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol
    ...
    """
```

obj 通常是一个字典对象，包含了模型参数，还有其他相关内容，比如：
- "model": 通常是模型参数，调用 model.state_dict() 获取
- "cfg": 配置文件 OmegaConf 实例
- "pickles": 通常用于可视化注意力的图

例如：

```py
torch.save(payload, path.open('wb'), pickle_module=dill)
```

### 参数 pickle_module

pickle_module 参数允许用户自定义序列化模型时使用的序列化库（默认是 Python 标准库的 pickle）。默认使用 Python 的 pickle 模块将模型（state_dict 或整个模型对象）序列化为字节流。通过 pickle_module 参数，可以：
- 替换默认的序列化库（例如使用 dill 或 cloudpickle）。
- 实现自定义的序列化逻辑（例如处理特殊对象或加密序列化数据）。

适用场景：
- 需要序列化 pickle 无法处理的复杂对象（如 lambda 函数、闭包等）。
- 在分布式训练或跨平台部署时，需要更灵活的序列化方案。
- 需要增强序列化的安全性（例如避免 pickle 的安全漏洞）。

比如，替换 pickle 为 dill 库（需要 pip install dill）。

```py
import torch
import dill  # 需要安装：pip install dill

# 保存模型时使用 dill 作为序列化库
torch.save(
    model.state_dict(), 
    "model_dill.pth", 
    pickle_module=dill
)

# 加载时也需要指定 dill
state_dict = torch.load(
    "model_dill.pth", 
    pickle_module=dill
)
model.load_state_dict(state_dict)
```

如果保存时用了 dill，加载时也必须指定 pickle_module=dill，否则会反序列化失败。

## 使用 torch.save 和 load 方法保存和加载任意实例

由于使用了 pickle_module，保存的 payload 可以是任何对象，比如 torch.Tensor 或 np.ndarray 实例，都能够正常存储和解析。

```py
import numpy as np
import torch
import dill

arr1 = np.array([1,2,3])
t1 = torch.zeros(3)
payload = {
    'arr1': arr1,
    't1': t1,
}
torch.save(payload, 'demo_payload.pt', pickle_module=dill)
```

在另一个文件，可以打开并解析：

```py
import numpy as np
import torch
import dill

payload = torch.load('demo_payload.pt', pickle_module=dill)
print(payload)
# {'arr1': array([1, 2, 3]), 't1': tensor([0., 0., 0.])}
```

### nn.Module._modules：保存 nn.Module 实例和字段名的字典

在 torch.nn.Module 及其子类中，_modules: Dict[str, Optional["Module"]] 字段是一个有序字典。字典的 key 对应类和父类以及各个继承链中，nn.Module 子类字段的名字。字典的值，对应实例中能够存储和加载参数的 nn.Module 模块实例。

```py
class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(3, 4))
        self.k = 2
    def forward(self):
        pass
class A(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l = torch.nn.Linear(3, 4)
        self.i = 3
    def forward():
        pass
class B(A):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv2d = torch.nn.Conv2d(3, 4, 1)
        self.my_net = MyNet()
        self.j = 4

    def forward():
        pass
b = B()
for k, v in b.__dict__.items():
    print(f"{k}={v}")
print(type(b._modules))
print(b.conv2d is b._modules["conv2d"]) # True
...
_modules={'l': Linear(in_features=3, out_features=4, bias=True), 'conv2d': Conv2d(3, 4, kernel_size=(1, 1), stride=(1, 1)), 'my_net': MyNet()}
i=3
j=4
<class 'dict'>
True
```

可以看到，普通的字段，比如 i 和 j 都直接在 `__dict__` 中存储，无特殊处理。而继承了 torch.nn.Module 类型的字段不同，比如 self.l 和 self.conv2d 和 self.my_net，存储到 _modules 字段中，一并管理。

### nn.Module 及其子类的直接字段中，并没有 state_dict 和 load_state_dict 属性

nn.Module 类有 state_dict() 和 load_state_dict() 方法。对于 nn.Module 及其子类，state_dict() 和 load_state_dict() 会保存和加载 _modules 中各个子模块的参数部分（nn.Parameter）。比如，"l" 对应 nn.Linear，state_dict() 返回的字典包含 "l.weight" 和 "l.bias" 两个 torch.Tensor 对象。

保存模型参数时，通常遍历 `__dict__` 属性，查看对应的字段是否有 state_dict() 和 load_state_dict() 方法，若有则保存，作为训练的参数。

但是，对于本身继承自 nn.Module 的子类，比如 class B，其 `__dict__` 属性中，只会包含属性，不会包含方法，即 B.state_dict() 和 B.load_state_dict() 不会包含到 `__dict__` 中来。另外，由于 self.conv2d, self.l 字段被保存到了 self._modules 中， `__dict__` 不会直接保存 self.conv2d 和 self.l 等字段，而是保存 self._modules。

```py
assert any(
    hasattr(v, "state_dict") or hasattr(v, "load_state_dict") for _, v in b.__dict__.items()
)  # assert failed
```

最佳实践是使用一个 class Workspace，其字段包含了模型。保存 checkpoint 时，可以通过 `__dict__` 找到了包含 state_dict 和 load_state_dict 属性的字段，即训练的模型。另一方面，可以保存配置文件，在开始时初始化，初始化后再加载模型的训练后的参数。


也可自定义类包含 load_state_dict() 和 state_dict() 获取和加载参数。

### 保存的最佳实践，以 DexGraspVLA 为例

DexGraspVLA 训练扩散策略部分，参考 Workspace，保存检查点如下：

```py
class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=True,
    ):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        # 每个实例的 __dict__ 属性包含了实例的所有属性
        for key, value in self.__dict__.items():
            # if 判断了 nn.Module, nn.Optimizer, nn.Sampler 模块，他们都有这两个方法
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        # 张量确保先放到 CPU 再保存
                        payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
```


加载训练好的模型文件如下：

```py
    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
```

## Ref and Tag