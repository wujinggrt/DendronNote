---
id: k1978v2ocsey0fpiew2fp0r
title: 从源码构建
desc: ''
updated: 1750733012722
created: 1750730676103
---

## 从源码构建 PyTorch

有时候需要支持 Python 3.11，而 Orin Jetson 的 JetPack 版本为 6.2，默认的 Python 版本为 3.10，不能满足项目需求。

参考 [pytorch from source](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source)

构建需要当前的 Python 环境，比如激活 uv 或 conda 环境。

```bash
uv venv -p 3.11.13
. ./.venv/bin/activate
```

如果使用 CUDA，需要对应支持，参考：
-   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
-   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above。注意选择对应平台，比如 Orin Jetson 选择 aarch64-jetson 等。
-   [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

需要确保找得到 nvcc。

最好参考 [CUDA 支持矩阵](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html)

## Ref and Tag