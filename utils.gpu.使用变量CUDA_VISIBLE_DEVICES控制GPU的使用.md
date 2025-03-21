---
id: 0ib3dy52yfsqto08lmowh0j
title: 使用变量CUDA_VISIBLE_DEVICES控制GPU的使用
desc: ''
updated: 1742550179401
created: 1742550121970
---

```bash
export CUDA_VISIBLE_DEVICES=1
```

使用上述，接下来运行的模型，只要传入 `device="cuda"`，都会只看到第 1 张（0 作为起始）GPU 核心。

## Ref and Tag