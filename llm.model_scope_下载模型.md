---
id: hnbm11ibxucge7ny52f6rwl
title: Model_scope_下载模型
desc: ''
updated: 1740229900658
created: 1740229598734
---

安装：`pip install modelscope`。官方建议安装 Git 和 Git LFS，方便上传和管理。

## 模型下载

使用命令行：
```bash
modelscope download --model="Qwen/Qwen2.5-0.5B-Instruct" --local_dir ./model-dir
```

或者使用 ModelScope Python SDK 下载模型，该方法支持断点续传和模型高速下载：
```py
from modelscope import snapshot_download
model_dir = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
```

注意，使用 ms-swift 框架微调时，应当使用 modelscope 库的 `AutoTokenizer`，以及对应的模型的类来处理。用法与 huggingface 相似，把 transformers 替换为 modelscope 即可。

## Ref and Tag

#ModelScope
#LLM