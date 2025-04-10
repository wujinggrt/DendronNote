---
id: jxe475zlbffvuu1cmcxe31u
title: 下载大模型
desc: ''
updated: 1744266662125
created: 1740743369137
---

下载工具：`pip install -U huggingface_hub`。

## 配置国内镜像源

```bash
export HF_ENDPOINT=https://hf-mirror.com 
```

## 样例

```bash
huggingface-cli download --resume-download MODEL_NAME --local-dir DIR -o1-7B --local-dir-use-symlinks False --token TOKEN
```

例如：

```bash
huggingface-cli download --resume-download lesjie/scale_dp_l --local-dir /data1/wj_24/huggingface/lesjie/scale_dp_l --local-dir-use-symlinks False
```

如果不指定 --local-dir，会下载到 ~/.cache/huggingface/hub/models。

## 使用脚本下载

### 单次指定下载目录

```py
from huggingface_hub import snapshot_download

# 下载模型到指定目录（不通过缓存）
model_dir = snapshot_download(
    repo_id="模型名称（如 meta-llama/Meta-Llama-3-8B）",
    local_dir="/path/to/your/target_directory",  # 指定目标目录
    local_dir_use_symlinks=False,  # 关闭符号链接，直接复制文件到目录
)
```

### 设置默认下载目录

如果希望所有下载任务存储到指定目录，通过如下方式设置。

#### 设置环境变量 HF_HOME 来指定全局缓存目录

修改环境变量，指定全局缓存目录（适用于所有 Hugging Face 库），等价于 ~/.cache/huggingface 目录。具体如下：

```bash
# 临时生效
export HF_HOME="/path/to/dir"
# 长期
echo 'export HF_HOME=/path/to/dir' >> ~/.zshrc
source ~/.zshrc
```

模型最终下载到 `$HF_HOME/hub` 目录下，此目录下会看到各种模型文件目录。

#### 设置 ~/.cache/huggingface 为软链接

引导到想要的目录下。

#### 在代码指定 cache_dir

```py
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    "模型名称",
    cache_dir="/path/to/your/default_directory",  # 指定缓存目录
)
tokenizer = AutoTokenizer.from_pretrained(
    "模型名称",
    cache_dir="/path/to/your/default_directory",
)
```

#### local_dir vs cache_dir/HF_HOME：
- local_dir: 直接下载到目标目录，文件独立于缓存，适合项目直接管理模型。
- cache_dir/HF_HOME: 文件存储在缓存系统目录（如 default_directory/models--repo_id），适合重复利用已下载的模型。

验证：

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("模型名称")
print(model.name_or_path)  # 输出模型路径
```

## 使用命令行

示例：

### huggingface-cli download

```bash
huggingface-cli download \
  meta-llama/Meta-Llama-3-8B \
  --resume-download \
  --local-dir ./my_models/llama3-8b \
  --local-dir-use-symlinks False
```

### git-lfs

```bash
git lfs install
git clone https://huggingface.co/模型名称 /path/to/your/target_directory
```

## Ref and Tag