---
id: edll7o8gebsag50rvjmls0i
title: 保存和加载模型
desc: ''
updated: 1742818717571
created: 1742751626701
---

TODO: load_state 等方法

## load() 加载模型参数

比如，DexGraspVLA 的 Workspace 中，加载训练好的模型文件如下：

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