---
id: 1m3kq4rfxs3p5kr73ylmg1b
title: 人形机器人站起_Learning_Getting-Up_Policies
desc: ''
updated: 1741847743668
created: 1741846887278
---

在 legged_gym 中，执行示例代码，报错：
```
  File "/data1/wj_24/projects/legged_gym/isaacgym/python/isaacgym/torch_utils.py", line 135, in <module>
    def get_axis_params(value, axis_idx, x_value=0., dtype=np.float, n_dims=3):
```

这是因为 numpy 现在没有 float，只有 float32 或 float64，应当使用 np.float32 或 np.float64。为了与 torch.float 一致，修改为 np.float32。

## Ref and Tag

[Github](https://github.com/RunpeiDong/HumanUP/tree/master)