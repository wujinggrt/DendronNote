---
id: jovit5b33arsj0m4y1cj50g
title: 优化器_AdamW_为何有些参数需要decay_有些no_decay？
desc: ''
updated: 1747905629362
created: 1747905617102
---

Q：
```
在 PyTorch 框架，通常设置 AdamW 优化器时，不同参数使用不同的 weight_decay，这是为什么？比如下面代码：
        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
```



## Ref and Tag