---
id: cy0z1e3avly9p2sitrkajl4
title: Vllm
desc: ''
updated: 1742643579412
created: 1740385950788
---

推理

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5
```

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --limit-mm-per-prompt image=5,video=5 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8000
```

## vllm serve 参数

```bash
--dtype {auto, bfloat16, ...}
    通常使用 auto 和 bfloat16
--api-key API_KEY
    如果提供，则需要放置此 key 到 header 中。比如 token-abc123
--max_model_len MAX_MODEL_LEN
    比如 8000，节省显存
--gpu_memory_utilization 0.98 # 避免 OOM
--cpu-offload-gb 64
    单位是 GB
--tensor-parallel-size TENSOR_PARALLEL_SIZE
    比如 2，使用张量并行来跨 GPU 加载大模型
--port PORT
--limit-mm-per-prompt LIMIT_MM_PERJ_PROMPT
  对于每个多模态插件，规定输入实例数量。用逗号分隔，比如 `image=16,video=2`，代表每个提示词最多 16 帐图片，2 个视频。每个模态默认为 1。
```


## 单机双卡参与推理

配置 `tensor_parallel_size` 参数分配模型到多个 GPU，实现张量并行。

配置环境变量: 
```bash
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1 # 禁止 Hugging Face Hub 发起 HTTP 调用，加快加载时间
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

## 使用本地图片向 Qwen2.5-VL 提问

```py
import base64
from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
image_path = "/data1/wj_24/llm/LLaMA-Factory/data/mllm_demo_data/1.jpg"
with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read())
encoded_image_text = encoded_image.decode("utf-8")
base64_qwen = f"data:image;base64,{encoded_image_text}"
chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": base64_qwen},
                },
                {"type": "text", "text": "Who are they?"},
            ],
        },
    ],
)
print("Chat response:", chat_response)
```

## 单节点多卡部署

[参考](https://blog.frognew.com/2024/10/multi-gpu-distributed-serving-qwen-2.5-14b-instruct.html)


## 使用 systemd 配置为系统服务

编辑 `/etc/systemd/system/qwen2.5-14b-instruct.service`：

```bash
[Unit]
Description=qwen2.5-14b-instruct
After=network.target

[Service]
Type=simple
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="HF_HUB_OFFLINE=1"
WorkingDirectory=/home/<thuser>/vllm
User=<theuser>
ExecStart=/bin/bash -c 'source .venv/bin/activate && \
    vllm serve Qwen/Qwen2.5-14B-Instruct \
        --served-model-name qwen2.5-14b-instruct \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --max-model-len=32768 \
        --tensor-parallel-size 2 \
        --port 8000'

Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

由于有了 WorkingDirectory，默认在此目录。在 ExecStart 处，可以看到 .ven/bin/activate 也在此工作目录下。随后启动服务：

```bash
systemctl enable qwen2.5-14b-instruct
systemctl start qwen2.5-14b-instruct
# 查看启动日志
journalctl -u qwen2.5-14b-instruct -f
```

## 提示安装 flash-attn

```bash
WARNING 03-22 19:38:03 vision.py:94] Current `vllm-flash-attn` has a bug inside vision module, so we use xformers backend instead. You can run `pip install flash-attn` to use flash-attention backend.
```

## Ref and Tag

[]()
[vllm多机多卡部署deepseek-vl2大模型 - 鹿小臻的文章 - 知乎](https://zhuanlan.zhihu.com/p/23184220175)