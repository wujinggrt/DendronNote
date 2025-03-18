---
id: vdmg23j454hslejym14pzwt
title: 使用_GPU
desc: ''
updated: 1742307965607
created: 1742306170823
---

首先确保主机正确安装驱动，还有 NVIDIA Container Toolkit。可以使用 nvidia 下配置好 cuda 环境的镜像。也可以更简单，使用 pytorch 官网提供的镜像。

对于 12.4.1，可以下载 docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 镜像。

```bash
docker run --gpus all --rm nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

## Ref and Tag

Nvidia 显卡 Docker 配置指引 - MitsuhaYuki的文章 - 知乎
https://zhuanlan.zhihu.com/p/29973252141

https://github.com/QwenLM/Qwen2.5-VL/blob/main/docker/Dockerfile-2.5-cu121

https://hub.docker.com/r/nvidia/cuda/tags

https://hub.docker.com/r/pytorch/pytorch/tags