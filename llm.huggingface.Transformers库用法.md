---
id: 17qvnqry352fhhwxcp3imco
title: Transformers库用法
desc: ''
updated: 1741334316742
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

## 预处理数据 AutoProcessor

对于文本，使用分词器（Tokenizer）转换为一系列 tokens；图像输入使用图像处理器（ImageProcessor）转换为张量；多模态输入，使用处理器（Processor）结合 Tokenizer 和 ImageProcessor 或 Processor。

AutoProcessor 始终有效的自动选择适用于模型的正确 class，无论是哪一类。

### Tokenizer：编码文本

一般从预训练模型中获取分词器，比如：

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
```

多个句子需要预处理，那么传递列表给 `tokenizer`。可以看到，对其到了相同的大小。

#### 编码

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

- input_ids 是与句子中每个 token 对应的索引。
- attention_mask 指示是否应该关注一个 token。
- token_type_ids 在存在多个序列时标识一个 token 属于哪个序列。

如果需要返回的形式是张量，而非 list，那么在 tokenizer 调用时传入参数 return_tensors="pt"。

#### 解码

```py
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```

### ImageProcessor：计算机视觉任务

### AutoProcessor：处理多模态场景

示例如下：

```py
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

#### 聊天模板

聊天场景由一条或多条消息组成的对话组成，每条消息都有一个“用户”或“助手”等 角色，还包括消息文本。聊天模板是Tokenizer的一部分。用来把问答的对话内容转换为模型的输入prompt。

比如，上面代码中：

```py
>>> text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
>>> print(text)
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>
<|im_start|>assistant
```

参考 [聊天模型的模板](https://huggingface.co/docs/transformers/main/zh/chat_templating#%E8%81%8A%E5%A4%A9%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%A8%A1%E6%9D%BF)。

## AutoConfig

加载模型的配置参数。通常和模型保存在一起，在同目录下的 config.json。

## GenerationMixin

transformers 的库中，自回归文本生成模型大多数继承了 `GenerationMixin`，包含一系列工具，比如 `generate()` 方法。

```py
class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    ...
```

由于 LLM 一次只预测一个 token，因此除了调用模型，还需要执行更复杂的操作生成新的句子，即自回归生成。自回归生成是给定一些初始输入，通过迭代调用模型及其自身的生成输出来生成文本的推理过程。由 transformers 中的 generate() 完成。

在自回归生成时，迭代重复地生成 token，直到遇见模型决定的停止条件。模型应该学会在何时输出结束序列（EOS）标记。如果没有遇到，则在最大长度时停止生成。相关配置参考 generation.GenerationConfig 文件。

输出时，如果指定 return_dict_in_generate=True 或 config.return_dict_in_generate=True，返回 ModelOutput，否则返回 torch.LongTensor，一般形状是 (batch_size, num_generated_tokens)，这是默认的选项。注意，调用 generate() 方法时，可以传递参数来覆盖默认行为。

`GenerationMixin` 要求子类自行实现各类方法，以便它的 `generate()` 调用。当调用 generate 时，有哪些方法会参与？

```py
# transformers/generation/utils.py
```

参考 [how-to-generate](https://huggingface.co/blog/zh/how-to-generate)，了解 generate() 输出的方式。

参考 [GenerationMixin](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate)

## 补充材料
[Quicktour](https://huggingface.co/docs/transformers/main/zh/quicktour)

## Ref and Tag

#阅读代码
#Code