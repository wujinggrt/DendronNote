---
id: dccrazf96gz3grerpuhevx8
title: Module_获取和访问可学习参数
desc: ''
updated: 1744653225404
created: 1744652906614
---

nn.Module子类模型中，parameters()和named_parameters()方法主要用于访问​​可学习的参数​​，即通过nn.Parameter定义的参数或通过register_parameter()注册的参数。

## parameters()和named_parameters()的内容​​

### ​​可学习参数的范围​​

- ​​仅包含可学习参数​​：这两个方法返回的均是模型中需要优化的参数（即requires_grad=True的参数），例如全连接层的权重（weight）和偏置（bias）、卷积层的卷积核参数等。
- ​​不包含非学习参数​​：例如BatchNorm层中的running_mean和running_var（统计量）属于缓冲区（buffer），不会通过这两个方法返回，而是存储在state_dict()中。

### 返回迭代器

两个方法都返回迭代器。后者返回一个元组，包含名称和参数对象。

```py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)  # 可学习参数：fc.weight, fc.bias
        self.register_buffer('buffer', torch.randn(3))  # 非学习参数

model = MyModel()

# 使用parameters()访问
print("parameters():")
for param in model.parameters():
    print(param.shape)  # 输出：torch.Size([3, 4]), torch.Size([3])

# 使用named_parameters()访问
print("\nnamed_parameters():")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
    # 输出：fc.weight: torch.Size([3, 4]), fc.bias: torch.Size([3])
```

保存模型参数时，通常用 state_dict() 来保存，因为不涉及区分是否可更新参数。

传入优化器参数时，通常用 parameters() 来传入，因为只需要可学习参数。

## Ref and Tag