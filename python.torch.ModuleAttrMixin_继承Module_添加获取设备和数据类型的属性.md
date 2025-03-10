---
id: z6ag74k1mrc3on5wazs4m67
title: ModuleAttrMixin_继承Module_添加获取设备和数据类型的属性
desc: ''
updated: 1741583935835
created: 1741583637626
---

nn.Module 不像 torch.Tensor 可以知道模型在 cpu 还是 cuda 上，dtype 同理。但是有 to() 方法将其转移参数到设备。所以，项目一般继承，并引入此属性，方便了解模型参数在哪个设备、是什么类型。

```py
# model/common/module_attr_mixin.py
import torch.nn as nn

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        # 避免模型没有参数的情况下，self.parameters() 没有内容，出现未定义行为。
        self._dummy_variable = nn.Parameter(requires_grad=False)

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
```

## Ref and Tag

#PyTorch