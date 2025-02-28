---
id: 0da424ysmswufl4406wj1dt
title: transformers_Trainer
desc: ''
updated: 1740730558418
created: 1740301523116
---

`Trainer` 提供类似 PyTorch 的 API，用于处理大多数标准用例的全功能训练。例化你的 Trainer 之前，创建一个 `TrainingArguments` 来定制训练。Trainer 支持在多个 GPU/TPU 分布式训练，支持混合精度。

构造函数的参数，参考文档或代码提出的文件。常用的如下：
```py
class Trainer:
    """
    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions.
        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. 比如 lora configs，fp16/bf16 settings 等。
            如果不提供，则使用默认的 TrainingArguments。
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. 类似与传给 DataLoader 的 collate_fn，重新组织为 batch。
        train_dataset (Union[`torch.utils.data.Dataset`, `torch.utils.data.IterableDataset`, `datasets.Dataset`], *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`]
        eval_dataset (Union[`torch.utils.data.Dataset`, Dict[str, `torch.utils.data.Dataset`, `datasets.Dataset`]), *optional*):
            The dataset to use for evaluation. If it is a [`~datasets.Dataset`]
    ...
    """
```

`Trainer` 包含基本训练循环。如果想要自定义训练，可以继承 `Trainer`，并覆盖如下方法：
- `get_train_dataloader` — 创建训练 DataLoader。
- `get_eval_dataloader` — 创建评估 DataLoader。
- `get_test_dataloader` — 创建测试 DataLoader。
- `log` — 记录观察训练的各种对象的信息。
- `create_optimizer_and_scheduler` — 如果它们没有在初始化时传递，请设置优化器和学习率调度器。请注意，你还可以单独继承或覆盖 create_optimizer 和 create_scheduler 方法。
- `create_optimizer` — 如果在初始化时没有传递，则设置优化器。
- `create_scheduler` — 如果在初始化时没有传递，则设置学习率调度器。
- `compute_loss` - 计算单批训练输入的损失。
- `training_step` — 执行一步训练。
- `prediction_step` — 执行一步评估/测试。
- `evaluate` — 运行评估循环并返回指标。
- `predict` — 返回在测试集上的预测（如果有标签，则包括指标）。

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

方法 `def parse_args_into_dataclasses()` 解析命令行参数到指定 dataclass 类型的实例。一般传入的 dataclass 都会指定默认值，且会被传入的命令行参数覆盖。返回的元组包含对应传入参数 DataClassType 顺序的 dataclass 类型实例。如果指定，元组还包含其他相关内容，具体参考源码或文档。

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


## Ref and Tag
[[llm.huggingface.Transformers库用法]]
[[llm.huggingface.DeepSpeed集成]]

[Hugginface: Trainer](https://huggingface.co/docs/transformers/v4.49.0/zh/main_classes/trainer)