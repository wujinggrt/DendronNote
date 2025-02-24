---
id: lvhxgyuvwyw5soqj59gc045
title: LLaMA-Factory
desc: ''
updated: 1740406633800
created: 1740375991093
---

## 教程目标

功能包括：
1. 原始模型直接推理
2. 自定义数据集构建
3. 基于LoRA的sft指令微调
4. 动态合并LoRA的推理
5. 批量预测和训练效果评估
6. LoRA模型合并导出
7. 一站式webui board的使用
8. API Server的启动与调用
9. 大模型主流评测 benchmark
10. 导出GGUF格式，使用Ollama推理

## 检查 CUDA 环境

```py
>>> import torch
>>> torch.cuda.current_device()
0
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce RTX 4090'
>>> torch.__version__
'2.6.0+cu124'
```

## 下载模型

参考 Hugginface 教程。下载后，运行官方原始的推理 demo，验证模型文件的正确性和 transformers 库是否可用。

## 原始模型直接推理

开始工作之前，先试用推理模式，验证 LLaMA-Factory 推理部分是否正常。LLaMA-Factory 带了基于 gradio 开发的 ChatBot 推理页面，帮助测试，需要执行：

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat --model_name_or_path Qwen/Qwen2-VL-2B-Instruct --template qwen2_vl
```

### template

关于 Qwen2.5-VL-3B-Instruct 模型，也可以沿用 qwen2_vl 模板。类似地，LLaMa 3.X 也是用的 llama3 的模板。于是只用运行如下命令便可以 Qwen2.5-VL-3B-Instruct 启动。

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct --template qwen2_vl
```

模板支持情况参考 [Supported Models](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#supported-models)。

### 从 yaml 加载

也可保存在 yaml 中，按照官方的样例 [examples/inference/qwen2_vl.yaml](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/inference/qwen2_vl.yaml)

```yaml
model_name_or_path: Qwen/Qwen2-VL-7B-Instruct
template: qwen2_vl
infer_backend: huggingface  # choices: [huggingface, vllm]
trust_remote_code: true
```

随后直接执行：`llamafactory-cli webchat examples/inference/qwen2_vl.yaml`

可以通过本机的 IP，比如 http://localhost:7860 访问。对于服务器，可以使用 http://<server addr>:7860 访问。

### 打开防火墙，本地浏览器访问服务器的 webchat 页面

在服务器上查看端口是否正在监听：

```bash
ss -tuln | grep 7860
```

查看防火墙开放的端口端口：

```bash
sudo ufw status
❯ sudo ufw status
Status: active

# 可以看到，允许了比如 5901:5920 的 VNC server 端口等。
To                         Action      From
--                         ------      ----
21/tcp                     ALLOW       Anywhere
80/tcp                     ALLOW       Anywhere
5908/tcp                   ALLOW       Anywhere
5905/tcp                   ALLOW       Anywhere
5901:5920/tcp              ALLOW       Anywhere
22/tcp                     ALLOW       Anywhere
Anywhere                   ALLOW       192.168.123.129
5916                       ALLOW       Anywhere
21/tcp (v6)                ALLOW       Anywhere (v6)
80/tcp (v6)                ALLOW       Anywhere (v6)
5908/tcp (v6)              ALLOW       Anywhere (v6)
5905/tcp (v6)              ALLOW       Anywhere (v6)
5901:5920/tcp (v6)         ALLOW       Anywhere (v6)
22/tcp (v6)                ALLOW       Anywhere (v6)
5916 (v6)                  ALLOW       Anywhere (v6)

# 开放
sudo ufw allow 7860/tcp
# 阻止
sudo ufw deny 7860/tcp
```

## 可选参数

|动作参数枚举|参数说明|
|---|---|
|version|显示版本信息|
|train|命令行版本训练|
|chat|命令行版本推理chat|
|export|模型合并和导出|
|api|启动API server，供接口调用|
|eval|使用mmlu等标准数据集做评测|
|webchat|前端版本纯推理的chat页面|
|webui|启动LlamaBoard前端页面，包含可视化训练，预测，chat，模型合并多个子页面|

另外两个常用参数包含:
- model_name_or_path 参数名称为 huggingface 或 modelscope 的标准定义，或者是绝对路径。
- template 模型 QA 的模板。



## 自定义数据集构建

数据集格式在不同训练阶段表现不同。更多细节参考 [官方数据集解释](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md)。

系统目前支持 alpaca 和 sharegpt 两种数据格式。以 alpaca 为例，数据集是一个 json 对象的 list，格式如下：

```json
[
  {
    "instruction": "用户指令（必填）",
    "input": "用户输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
  }
]
```

以 alpaca_data_zh_51k.json 为例，在 data/dataset_info.json 注册为 具体如下：

单轮对话场景：

```json
{
  "instruction": "写一个有效的比较语句",
  "input": "篮球和足球",
  "output": "篮球和足球都是受欢迎的运动。"
}
```

多轮对话场景：

```json
{
  "instruction": "谢谢",
  "input": "",
  "output": "不用谢! 很高兴我提供的信息能够帮助到你! 如果还有什么其他问题也可以向我提问。",
  "history": [
    [
      "请你给我写一个面试准备计划，我想要去面试微软的程序员岗位",
      "首先，你可以去微软官网寻找招聘信息并申请面试。\n其次，您可以在社交媒体平台寻找微软公司对程序员的面试问题，并做好准备。\n最后，您可以自己对面试过程进行模拟，熟悉话题并减少紧张感。\n我希望你能面试成功。"
    ]
  ]
}
```

训练的数据最好遵循这种格式，随后在 data/dataset_info.json 中注册。接下来，使用两个具体例子学习。

关于数据集的描述文件，则为：

```json
"数据集名称": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

### alpaca 和 sharegpt

## Ref and Tag
[Github](https://github.com/hiyouga/LLaMA-Factory)
[知乎教程](https://zhuanlan.zhihu.com/p/695287607)
[与 DeepSpeed 训练 Qwen](https://zhuanlan.zhihu.com/p/714707824)

[[llm.huggingface.DeepSpeed集成]]
[[llm.DeepSpeed_核心概念]]
[[llm.Megatron]]

#LLM
 