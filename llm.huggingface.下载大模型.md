---
id: jxe475zlbffvuu1cmcxe31u
title: 下载大模型
desc: ''
updated: 1741703233794
created: 1740743369137
---

下载工具：`pip install -U huggingface_hub`。

配置国内镜像源

```bash
export HF_ENDPOINT=https://hf-mirror.com 
```

在本地要先创建目录，保证目录存在，随后再下载。

```bash
huggingface-cli download --resume-download MODEL_NAME --local-dir DIR -o1-7B --local-dir-use-symlinks False --token TOKEN
```

例如：

```bash
huggingface-cli download --resume-download lesjie/scale_dp_l --local-dir /data1/wj_24/huggingface/lesjie/scale_dp_l --local-dir-use-symlinks False
```

如果不指定 --local-dir，会下载到 ~/.cache/huggingface/hub/models。

## Ref and Tag