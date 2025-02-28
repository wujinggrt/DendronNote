---
id: 17qvnqry352fhhwxcp3imco
title: Transformers库用法
desc: ''
updated: 1740752220962
created: 1740205377695
---

## PreTrainedModel

PreTrainedModel 是所有模型的基类。需要指定类属性 (class attributes)，子类要覆盖后面的内容。

### 优势

继承了 PreTrainedModel 的模型，可以与 Transformers 生态无缝集成。比如使用 AutoModel 注册，后续可以用此框架的 AutoConfig.from_pretrained 来加载参数，XX.from_pretrained 来加载模型。具体来说：
1. 预训练模型的便捷加载与保存。
    - 可以使用 from_pretrained() 加载 Hugging Face Hub 或本地保存的权重。
    - 可以使用 save_pretrained() 保存到指定目录。
    - 兼容多框架，比如 torch.save 或 tf.saved_model。
2. 配置管理 (PretrainedConfig)。
    - 解耦模型结构与超参数：独立保存在 config 对象，便于修改和重新建模。
    - 动态调整配置：修改 config，快速创建不同模型。
3. 与 Transformers 生态无缝集成。
    - 兼容内置工具：直接使用 Trainer 类训练、Pipeline 推理，AutoTokenizer 自动匹配分词器。
    - 支持混合精度训练：与库中的 training_args 结合，轻松启用 FP16/BP16 训练。
    - 梯度检查点 (Gradient Checkpointing)：启用 config.use_cache = False，节省显存。
4. 上传社区。

PreTrainedModel 继承了 nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin。

实例化时，接收参数 ( config: PretrainedConfig*inputs**kwargs )。

### 实现继承细节

必须要绑定配置类，正确实现构造函数。指定关联的配置类，要求为 PretrainedConfig 或子类，用于自动解析配置参数。在构造函数中，接收对应 config 对象。还需要实现 forward 方法，保证前向传播，对应 nn.Module 要求。

```py
class ScaleDP(PreTrainedModel):
    """
    Diffusion models with a Transformer backbone.
    """
    config_class = ScaleDPPolicyConfig
    def __init__(
            self,
            config: ScaleDPPolicyConfig,
    ):
        ...
```

可选但有影响的部分包括处理预训练权重，保存和加载模型。

集成 Transformers 方面：
- 兼容 Trainer 类。确保模型在 forward() 中，输出包含 loss。
- 支持 Pipeline。如果需要注册模型到自动类，则要用 AutoModel.register 或自定义 AutoConfig。
- 启动优化特性。config.use_cache = False 节省显存。

## PretrainedConfig

问 DeepSeek：如果我要继承 PretrainedConfig，那么应该注意哪些地方

```py
class CustomConfig(PretrainedConfig):
    model_type = "custom"  # 必须定义，用于自动识别模型类型

# 必须注册模型类型
AutoConfig.register("custom", CustomConfig)
# 之后可通过 AutoConfig.from_pretrained("custom") 加载
```

## Tokenizer

一般从预训练模型中获取分词器，比如：

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
```

多个句子需要预处理，那么传递列表给 `tokenizer`。可以看到，对其到了相同的大小。
```python
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences)
print(encoded_inputs)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1]]}
encoded_input = tokenizer(batch_sentences, padding=True)
print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

## GenerationMixin
transformers 的库中，自回归文本生成模型大多数继承了 `GenerationMixin`，包含一系列工具，比如 `generate()` 方法。

```py
class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    ...
```

`GenerationMixin` 要求子类自行实现各类方法，以便它的 `generate()` 调用。

```py
# transformers/generation/utils.py
```

参考 [how-to-generate](https://huggingface.co/blog/zh/how-to-generate)。

## 补充材料
[Quicktour](https://huggingface.co/docs/transformers/main/zh/quicktour)

## Ref and Tag

#阅读代码
#Code