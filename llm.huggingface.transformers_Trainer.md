---
id: 0da424ysmswufl4406wj1dt
title: transformers_Trainer
desc: ''
updated: 1744903856647
created: 1740301523116
---

`Trainer` 提供类似 PyTorch 的 API，用于处理大多数标准用例的全功能训练。例化你的 Trainer 之前，创建一个 `TrainingArguments` 来定制训练。Trainer 支持在多个 GPU/TPU 分布式训练，支持混合精度。

## API

### 构造函数

构造函数的参数，参考文档或代码提出的文件。关心的参数如下：

```py
class Trainer:
    """
    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. 比如 lora configs，fp16/bf16 settings 等。
            如果不提供，则使用默认的 TrainingArguments。
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. 
        train_dataset (Union[`torch.utils.data.Dataset`, `torch.utils.data.IterableDataset`, `datasets.Dataset`], *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`]
        eval_dataset (Union[`torch.utils.data.Dataset`, Dict[str, `torch.utils.data.Dataset`, `datasets.Dataset`]), *optional*):
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
            This supercedes the `tokenizer` argument, which is now deprecated.
        compute_loss_func (`Callable`, *optional*):
            A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
            batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default [loss function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618) used by [`Trainer`].
    ...
    Important attributes:
        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to `False` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return `False` .
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)
    """
```

到有些项目传入 tokenizer，在源码已经有装饰器标注为 deprecate，推荐使用 processing_class。

#### train_dataset 和 eval_dataset

Dataset 的 `__getitem__` 中，每个样本格式要求通常由训练的模型决定。通常是一个字典，包含 input_ids, labels 等字段：

```py
{
    "input_ids": List[int],      # 输入文本的 token ID
    "decoder_input_ids": List[int], # 解码器输入（可选，部分模型需要）
    "labels": List[int]          # 目标文本的 token ID
}
```

#### 参数 data_collator

类似 DataLoader 的 collate_fn，将 list 重新组织为 batch。不指定则默认使用 transformers.data.data_collator.default_data_collator()，最后路由到

```py
def torch_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    ...
```

### 方法

`Trainer` 包含基本训练循环。如果想要自定义训练，可以继承 `Trainer`，并覆盖如下方法：
- `get_train_dataloader()` — 创建训练 DataLoader。
- `get_eval_dataloader()` — 创建评估 DataLoader。
- `get_test_dataloader()` — 创建测试 DataLoader。
- `log()` — 记录观察训练的各种对象的信息。
- `create_optimizer_and_scheduler()` — 如果它们没有在初始化时传递，请设置优化器和学习率调度器。请注意，还可以单独继承或覆盖 create_optimizer 和 create_scheduler 方法。
- `create_optimizer()` — 如果在初始化时没有传递，则设置优化器。
- `create_scheduler()` — 如果在初始化时没有传递，则设置学习率调度器。
- `compute_loss()` - 计算单批训练输入的损失。默认实现中，返回的元组中，第一个元素应当是 loss。如果在构造函数传入了 compute_loss_func 则使用它。但是，默认使用 outputs = model(**inputs) 的 outputs[0] 作为 loss。
- `training_step()` — 执行一步训练。返回 loss
- `prediction_step()` — 执行一步评估/测试。
- `evaluate()` — 运行评估循环并返回指标。
- `predict()` — 返回在测试集上的预测（如果有标签，则包括指标）。

## 示例

```py
from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

### data_collator 参数

data_collator 的输入参数 batch 的数据类型通常是 ​​由多个样本组成的列表（List）​​，其中每个样本是一个字典（dict），键值对表示模型需要的输入字段（如 input_ids、attention_mask、labels 等）。

```py
def custom_collator(batch):
    # 假设每个样本包含文本和图像像素
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    
    # 文本动态填充
    text_inputs = tokenizer(texts, padding=True, return_tensors="pt")
    
    # 图像转换为张量
    image_inputs = tensor(images)
    
    return {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "pixel_values": image_inputs
    }
```

## Trainer.add_callback()
在自定义训练循环的另一个方式是，使用 [callback](https://huggingface.co/docs/transformers/v4.49.0/zh/main_classes/callback)。

```py
    def add_callback(self, callback):
        """
        Add a callback to the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback]`):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        ...
```

Trainer 维护了一个关于回调的列表。

## Trainer.compute_loss()

```py
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
```

## evalute

用于保存最佳模型。

## 保存 Checkpoints

Trainer 会将所有checkpoints保存在你使用的 TrainingArguments 中设置的 output_dir 中。这些checkpoints将位于名为 checkpoint-xxx 的子文件夹中，xxx 是训练的步骤。

从checkpoints恢复训练可以通过调用 Trainer.train() 时使用以下任一方式进行：
- `resume_from_checkpoint=True`，这将从最新的 checkpoint 恢复训练。
- `resume_from_checkpoint=checkpoint_dir`，这将从指定目录中的特定 checkpoint 恢复训练。

## Logging

默认对主进程使用 logging.INFO。可以在 `TrainingArguments` 覆盖。

## 特定GPU选择

如果使用了 DeeSpeed，或 accelerate，可以用：

```bash
deepspeed --num_gpus 2 trainer-program.py ...
```

```bash
accelerate launch --num_processes 2 trainer-program.py ...
```

### 多 GPU
使用其中一个或几个，需要指定：

```bash
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch trainer-program.py ...
```

与任何环境变量一样，你当然可以将其export到环境变量而不是将其添加到命令：

```bash
export CUDA_VISIBLE_DEVICES=0,2
python -m torch.distributed.launch trainer-program.py ...
```

## TrainingArguments

### 常见属性和参数

- `output_dir`：必选，指定模型checkpoint和最终结果的输出目录，目录不存在会自动创建。checkpoint 保存在子目录 checkpoint-000 下。后面的数字代表训练步数。重启训练时，可以指定 `resume_from_checkpoint` 参数，指定从 checkpoint 恢复训练。
- `learning_rate`: float, 学习率
- `per_device_train_batch_size=8`: 训练批次大小
- `per_device_eval_batch_size=8`: 评估批次大小
- `lr_scheduler_type`: str, 默认 linear，可选 linear, cosine, constant, polynomial, piecewise, exponential
- `warmup_ratio`: float, 默认 0.0，warmup 比例
- `warmup_steps`: int, 默认 0，会覆盖 warmup_ration
- `bf16`: bool，默认 False，指定是否使用 bf16 训练

[transformers.TrainingArguments](https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/trainer#transformers.TrainingArguments) 是一个 `@dataclass`，通常使用 `transformers.HfArgumentParser` 把此 class 转换为 [argparse](https://docs.python.org/3/library/argparse#module-argparse) 参数，从而可以在命令行中覆盖。

### transformers.HfArgumentParser

它能够与 Python 原生的 argparser 无缝连接，将 dataclass 解析为参数。构造函数接受一个可迭代的 dataclass 的参数，其它则是 kwargs。

```py
class HfArgumentParser(ArgumentParser):
    """ 是 argparser.ArgumentParser 子类。 """
    def __init__(
        self, 
        dataclass_types: Optional[Union[DataClassType, Iterable[DataClassType]]] = None, **kwargs):
        ...

    def parse_args_into_dataclasses(
        self,
        args=None,
        return_remaining_strings=False,
        look_for_args_file=True,
        args_filename=None,
        args_file_flag=None,
    ) -> Tuple[DataClass, ...]:
        ...
```

方法 `parse_args_into_dataclasses()` 会解析命令行参数到指定 dataclass 实例的字段。一般传入的 dataclass 都会指定默认值，但是命令行参数会**覆盖**。返回的元组包含对应传入参数 DataClassType 顺序的 dataclass 类型实例。如果指定，元组还包含其他相关内容，具体参考源码或文档。

用法比如：

```py
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    ...
@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    ...
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    using_ema: bool = field(default=False) # whether to use ema update whole module, default to false
    ...
@dataclass
class ActionHeadArguments:
    policy_head_type: str = field(default="scale_dp_policy") # or unet_diffusion_policy
    ...

def parse_param():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ActionHeadArguments))
    # 随后便可像数据类一样访问 model_args, data_args, ... 的各个 field，
    # 就像 argparse 解析出来返回的参数一样
    model_args, data_args, training_args, action_head_args = parser.parse_args_into_dataclasses()
    ...
    return (
        model_args,
        data_args,
        training_args,
        action_head_args,
        config,
        bnb_model_from_pretrained_args,
    )

if __name__ == "__main__":
    (
        model_args,
        data_args,
        training_args,
        action_head_args,
        model_config,
        bnb_model_from_pretrained_args,
    ) = parse_param()
    config = {
        "model_args": model_args,
        "data_args": data_args,
        "training_args": training_args,
        "action_head_args": action_head_args,
        "bnb_model_from_pretrained_args": bnb_model_from_pretrained_args,

```

## 使用 Huggingface 的工具微调和训练

TODO: 使用 Trainer 和 Accelerate 来实验。

## 例子

### DexVLA 的 Qwen2VLATrainer

```mermaid
classDiagram
    class QWen2VLATrainer {
        +sampler_params: dict
        +prefetch_factor: int
        +lora_module: str
        +lang_type: str
        +using_ema: bool
        +local_rank: int
        +resume_from_checkpoint: bool
        +ema: EMAModel

        +__init__(sampler_params, prefetch_factor, *args, **kwargs)
        +get_train_dataloader() -> DataLoader
        +get_eval_dataloader(eval_dataset: Optional[Dataset]) -> DataLoader
        +_get_train_sampler() -> Optional[Sampler]
        +create_optimizer()
        +training_step(model: nn.Module, inputs: dict) -> torch.Tensor
        +_inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
        +_maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, all_loss=None)
        +_save_checkpoint(model, trial, metrics=None, using_ema=False)
        +_save(output_dir: Optional[str] = None, state_dict=None)
        +_load_from_checkpoint(resume_from_checkpoint, model=None)
    }

    class Trainer {
        <<Abstract>>
        #model: nn.Module
        #args: TrainingArguments
        #optimizer: torch.optim.Optimizer
        #lr_scheduler: torch.optim.lr_scheduler
        #state: TrainerState
        
        +train()
        +evaluate()
        +predict()
        +save_model()
        +_prepare_inputs()
        +compute_loss()
    }

    QWen2VLATrainer --|> Trainer : Inheritance

    class EMAModel {
        +averaged_model: nn.Module
        +step(model: nn.Module)
        +__init__(model, power)
    }

    QWen2VLATrainer --> EMAModel : Composition
```

更多参考 [[训练器：Qwen2VLATrainer|robotics.DexVLA_code_阅读代码和复现#训练器qwen2vlatrainer]]。


### 修改 Trainer 实现 LoRA++

以微调deepseek为例，基于transformers改写实现lora+ - KaiH的文章 - 知乎
https://zhuanlan.zhihu.com/p/688157210

## Ref and Tag
[[llm.huggingface.Transformers库用法]]
[[llm.huggingface.DeepSpeed集成]]

[Hugginface: Trainer](https://huggingface.co/docs/transformers/v4.49.0/zh/main_classes/trainer)

LLM大模型之Trainer以及训练参数 - Glan格蓝的文章 - 知乎
https://zhuanlan.zhihu.com/p/662619853

https://huggingface.co/docs/transformers/v4.51.1/en/trainer?ckpt=specific+checkpoint