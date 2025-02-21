---
id: 4gb9ottxmfh95i6654zy8hq
title: DexVLA复现
desc: ''
updated: 1740055515545
created: 1740053039805
---

## 数据准备
项目使用了 act 工作的数据格式，act 每个 timestep 数据格式如下：

observations
- images
    - each_cam_name     (480, 640, 3) 'uint8'
- qpos                  (14,)         'float64'
- qvel                  (14,)         'float64'

action                  (14,)         'float64'

随后，作者使用 rlds_to_h5py，把数据转换为 h5py 格式。

## VLM
使用 [Qwen2-2B-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) 作为主干网络。