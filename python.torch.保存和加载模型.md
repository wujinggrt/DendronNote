---
id: edll7o8gebsag50rvjmls0i
title: 保存和加载模型
desc: ''
updated: 1743133181680
created: 1742751626701
---

TODO: load_state 等方法

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

### _modules：保存了实例的 key 和对应对象

在 torch.nn.Module 及其子类中，_modules: Dict[str, Optional["Module"] 字段是一个字典。字典保存了实例中能够存储和加载参数的 nn.Module 模块。key 对应字段名，value 对应可保存加载的模块。比如：

```py
class A(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l = torch.nn.Linear(3, 4)
    def forward():
        pass

class B(A):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = torch.nn.Linear(3, 4)

    def forward():
        pass
b = B()
for k, v in b.__dict__.items():
    print(f"{k}={v}")
print(type(b._modules))
...
_modules={'l': Linear(in_features=3, out_features=4, bias=True), 'l1': Linear(in_features=3, out_features=4, bias=True)}
<class 'dict'>
```

可以看到，_

## 保存的最佳实践

### checkpoint 的选择

- 模型：保存 model.state_dict() 到 payload 的 key，加载时使用 model.load_state_dict()
- 优化器：保存 optimizer.state_dict()

{'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'best_val_loss': best_val_loss}

## 例子

### 简单例子


### 以 DexGraspVLA 为例

DexGraspVLA 训练扩散策略部分，参考 Workspace，保存检查点如下：

```py
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
```


加载训练好的模型文件如下：

```py
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

checkponints 文件通常是 

## Ref and Tag