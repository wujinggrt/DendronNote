---
id: cy0z1e3avly9p2sitrkajl4
title: Vllm
desc: ''
updated: 1740558985616
created: 1740385950788
---

推理

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5
```

## vllm server 参数
- --dtype auto
- --api-key token-abc123
- --max_model_len=8000
- --gpu_memory_utilization=0.98
- --cpu-offload-gb 64 # 单位是 GB


## 单机双卡参与推理

配置 `tensor_parallel_size` 参数分配模型到多个 GPU，实现张量并行。

配置环境变量: 
```bash
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1 # 不像 Hugging Face Hub 发起 HTTP 调用，加快加载时间
```

如果不需要 export，可以启动服务如下：
```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 vllm serve Qwen/Qwen2.5-14B-Instruct \
  --served-model-name qwen2.5-14b-instruct \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max_model_len=32768 \
  --tensor-parallel-size 2 \
  --port 8000
```

参数 `--tensor-parallel-size 2` 表示使用 Tensor Parallelism 技术来分配模型跨两个GPU

## Ref and Tag

[]()
[vllm多机多卡部署deepseek-vl2大模型 - 鹿小臻的文章 - 知乎](https://zhuanlan.zhihu.com/p/23184220175)