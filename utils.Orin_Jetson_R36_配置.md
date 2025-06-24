---
id: ke6zwxz1njywny6b177zpjo
title: Orin_Jetson_R36_配置
desc: ''
updated: 1750729922033
created: 1750473521003
---

## 配置显卡驱动

出厂通常配置了系统，有 nvidia-smi 命令，nvcc 在 /usr/local/cuda-12.6/bin 下可以查看和使用。

问题（What）：
1. 能使用 nvidia-smi，但看不到显卡信息
2. 能使用 nvcc -V，版本是 12.6.124
3. 核心问题：pip 安装 torch 后，torch.cuda.is_available() 为 False，不能用 GPU 加速推理

### 查看系统状态

但是，nvidia-smi 查看不到显卡信息，比如显存和显卡型号等。最佳实践是使用如下命令查看实时 GPU 状态：

```bash
tegrastats --interval 1000
```

GR3D_FREQ 代表 GPU 利用率，是关键指标；GPU 代表当前频率状态。

查看 JetPack 版本，即 Orin 的系统配置：

```bash
cat /etc/nv_tegra_release
```

输出版本内容。R36 表示大版本号，REVISION 表示小版本号。比如：

```
# R36 (release), REVISION: 4.3 ...
```

代表 JetPack Version 为 36.4.3。最直接办法如下，直接查看安装包的版本：

```bash
dpkg -l | grep nvidia-l4t-core # 会展示版本号
```

36.4.x 版本属于 JetPack 6.2。

### 安装对应 torch

因为使用了不同平台和开发板，Orin 要求定制的 torch 版本。参考 [Installing PyTorch for Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)，下载对应版本 whl 文件并安装。

需要查看适合 JetPack 版本的 PyTorch 版本。参考同级目录下的 [Compatibility Matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html#pytorch-jetson-rel)。

可以直接 uv pip 下载：

```bash
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
```

也可以先下载 wheel 文件并设置：

```bash
export TORCH_INSTALL=path/to/torch-2.2.0a0+81ea7a4+nv23.12-cp38-cp38-linux_aarch64.whl
```

其中，`2.2.0a0+81ea7a4` 是 PyTorch Version，`nv23.12` 是 NVIDIA Framework Container 版本，`cp38` 是 Python 版本。

最终，在 [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)，下载了 JetPack 6.0(L4T R36.2/R36.3) + CUDA 12.4 下的 torch 2.3, torchaudio 2.3 和 torchvision 0.18，可以成功使用 CUDA。

## 从源码构建 PyTorch

有时候需要支持 Python 3.11，而 Orin Jetson 的 JetPack 版本为 6.2，默认的 Python 版本为 3.10，不能满足项目需求。

参考 [pytorch from source](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source)

## Ref and Tag