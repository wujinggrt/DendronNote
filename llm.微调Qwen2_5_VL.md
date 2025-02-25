---
id: tfd9jjc8w7feftqzbyc0pud
title: 微调Qwen2_5_VL
desc: ''
updated: 1740474499924
created: 1740209908837
---

使用了 Qwen2-VL-7B-Instruct，SFT 框架为 LlaMA-Factory。

全量微调的参数：
```yaml
model_name_or_path: Qwen2-VL-7B-Instruct

### method
stage: sft
do_train: true
finetuning_type: full

### dataset
dataset: train_nlp
template: qwen2_vl
cutoff_len: 3000
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/qwen2_vl-7b/
logging_steps: 5
save_steps: 300
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 100

### save
save_total_limit: 1
load_best_model_at_end: True

flash_attn: fa2
# offload 指将forward中间结果保存到内存、硬盘（NVMe）等缓存中，然后在需要时进行加载或重计算，进一步降低显存占用
deepspeed: examples/deepspeed/ds_z3_offload_config.json
```

## 使用 LLaMA-Factory 微调 Qwen 模型

官方提供了文档 [使用LLaMA-Factory微调Qwen模型](https://github.com/QwenLM/Qwen2.5/blob/main/examples/llama-factory/finetune-zh.md)。

### 准备训练数据

自定义的训练数据保存为 json 文件，每行格式为：

```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Tell me something about large language models."
        },
        {
            "role": "assistant",
            "content": "Large language models are a type of language model that is trained on a large corpus of text data. They are capable of generating human-like text and are used in a variety of natural language processing tasks..."
        },
        {
            "role": "user",
            "content": "How about Qwen2?"
        },
        {
            "role": "assistant",
            "content": "Qwen2 is a large language model developed by Alibaba Cloud..."
        }
      
    ]
}
```

LLaMA-Factory 目录下的 `data/dataset_info.json` 注册自定义训练数据时，在文件尾部添加：

```json
"qwen_train_data": {
    "file_name": "PATH-TO-YOUR-TRAIN-DATA",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
}
```

### 配置训练参数

官方提供了，

## 以下为作者比赛笔记
全量微调需要 offload 模式，作者使用 4 卡 A30，内存开销大致 120GB。

## Prompt

SFT 时，Prompt 优化比较重要。比如，要求只提取图像分类任务做说明，意图识别要与任务操作一致。

baseline 原始 Prompt：

```
<image>\n你是一个电商领域识图专家,可以理解消费者上传的软件截图或实物拍摄图。现在,请你对消费者上传的图片进行分类。你只需要回答图片分类结果,不需要其他多余的话。以下是可以参考的分类标签,分类标签:[\"实物拍摄(含售后)\",\"商品分类选项\",\"商品头图\",\"商品详情页截图\",\"下单过程中出现异常（显示购买失败浮窗）\",\"订单详情页面\",\"支付页面\",\"消费者与客服聊天页面\",\"评论区截图页面\",\"物流页面-物流列表页面\",\"物流页面-物流跟踪页面\",\"物流页面-物流异常页面\",\"退款页面\",\"退货页面\",\"换货页面\",\"购物车页面\",\"店铺页面\",\"活动页面\",\"优惠券领取页面\",\"账单/账户页面\",\"个人信息页面\",\"投诉举报页面\",\"平台介入页面\",\"外部APP截图\",\"其他类别图片\"]。
```


|Prompt|分数变化|个人想法|
|---------|---|----|
|加入更细致的类别说明: label:{实物拍摄(含售后)}, description: {用户用相机实拍的照片，包括用户售后的照片（损坏、缺失、与描述不符），或者其他用相机实拍的图片。}|分数有1个点左右的提升	|在做分类任务时，一个清晰的类别描述很重要。但需要限制Prompt的长度。prompt超过2500字之后模型对Prompt的理解力有降低，会出现重复输出的问题。 这里使用json格式对标签与描述做区分。|
|Prompt中加入分类原因:在模型训练时，加入人工标注的标签，让模型输出原因，示例:现在请你根据消费者上传的图片及分类标签，并依据该图片的标签:'退货页面', 给出分为该类的原因	|分数有1个点左右的提升|label在前，reason在后会出现标签和最终结果不一致的问题。|
|Prompt中加入Cot，输出更详细的原因||有轻微的提升|

### 数据集优化
比赛给了1000条训练集，作者做了两个工作：
1. K-Fold 训练模型，把预测结果不同的修正，得到轻微提升。
2. 使用模型标注测试集，把测试集结果加入训练集，继续训练，有一个点的提升。

## 数据准备

### 优化细节
1. Prompt 中先对图片做描述。可以使用 GPT 描述图片，参考 [数据集生成工作](https://arxiv.org/pdf/2311.12751)。然后再根据描述分类，构造一个两步走的 Prompt。
2. 数据平衡。由于分析最终的结果，训练集中类别较少的数据在测试集能正常标注，没有此调整，可能会影响。
3. 更细致的标签类别分析。

### 训练反而效果不好

1. 优化模型时，可能会遇到效果反而变差的情况。需要细致分析。
2. 提分最大的三个点：(1) 将模型预测的测试集加入模型训练；(2) 不止输出类别标签，同时输出原因；(3) 加入 OCR 的结果。
3. 微调的 Prompt 中，原因部分不能过分相似，否则分数骤降。
4. Prompt 长度要始终，太长会降低模型对提示词的理解。

## 医学模型微调经验

微调参数：`--learning_rate 1e-4 --warmup_ratio 0.03 --max_length 2048 --batch_size 4 --weight_decay 0.1 --gradient_accumulation_steps 16`。

见解在于，为了保证通识能力不退化，引入了通识数据集 alpaca。alpaca-zh 的中文版本数据集，

## Link and Ref
[LLaMA-Factory微调多模态大语言模型教程 - BrownSearch的文章 - 知乎](https://zhuanlan.zhihu.com/p/699777943)


[](https://zhuanlan.zhihu.com/p/17193156687)
[医疗大模型微调银奖，Qwen-VL-Chat](https://zhuanlan.zhihu.com/p/839580322)
[llama_factory 流程篇——以Qwen为例（阿里云+llama_factory+lora微调大模型 ） - 小王子的文章 - 知乎](https://zhuanlan.zhihu.com/p/714707824)

[[llm.DeepSpeed_核心概念]]
[[llm.LLaMA_Factory]]

#MLLM