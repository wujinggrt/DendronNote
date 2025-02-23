---
id: p6pfuzn9vki5iiiseciacrh
title: DeepSpeed集成
desc: ''
updated: 1740323635850
created: 1740323635850
---

DeepSpeed 由 Zero Redundancy Optimizer 驱动，能够在一个 GPU 上装上大型模型。ZeRO 工作包含几个阶段：
- ZeRO-1, 跨越 GPUs 做优化器状态 partitioning。
- ZeRO-2, 跨越 GPUs 做梯度 partitioning。
- ZeRO-3, 跨越 GPUs 做参数 partitioning。

GPU 首先环境，ZeRO 可以从 GPU 到 CPU 做 offloading 优化器内存和计算，以适应和训练大型模型。DeepSpped 集成所有 到了 `Trainer`。只需要提供 config 文件或模板。关于推理，提供 ZeRO-3 和 offloading，但是更多会选择 vllm。

安装只需 `pip install deepspeed`。

## 内存要求

检查 GPU 和 CPU 内存是否足够，十分重要。DeepSpeed 提供了如此工具。用法如：

```bash
# 模型在 ./bigscience/T0_3B
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 2783M total params, 65M largest layer params.
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.37GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=1
   15.56GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=0
```

代表在没有 CPU offload 时，需要单个 80GB GPU，或者有一个约 60GB CPU 来 offload，那么需要 8GB GPU。我们需要在开销和速度之间考虑 tradeoff。GPU 内存如果足够，取消 CPU/NVM offload 会更快。

## 选择一个 ZeRO 阶段

在明确内存需求后，比如是否需要 offload 等，接下来要选择 ZeRO 阶段。

|Fastest|Memory efficient|
|---|---|
|ZeRO-1|ZeRO-3 + offload|
|ZeRO-2|ZeRO-3|
|ZeRO-2 + offload|ZeRO-2 + offload|
|ZeRO-3|ZeRO-2|
|ZeRO-3 + offload|ZeRO-1|

从上表可以看到速度与内存开销的 tradeoff。可以从最快的开始尝试，如果发现内存耗尽，那么选择下一阶段，速度降低但内存更高效。Best practice：
1. 激活梯度 checkpointing
2. 尝试 ZeRO-2
3. 尝试 ZeRO-2 + offload 优化器
4. 尝试 ZeRO-3
5. 尝试 ZeRO-3 + offload 参数 到 CPU
6. 尝试 ZeRO-3 + offload 参数和优化器到 CPU
7. 尝试降低默认值，比如 `generate()` 方法中的 narrower search beam
8. 尝试混合精度 (在较老的 GPU 架构使用 fp16，在 Ampere 使用 bf16)
9. 添加更多硬件，或启用无限 (Infinity) 来 offload 参数和优化器到 NVMe
10. 一旦没有耗尽内存，测试高效的吞吐量，随后增加 batch size，直到逼近 GPU 效率极限
11. 最后，尝试优化训练设置，关闭某些 offload 特性，或使用更快的 ZeRO stage，增加 batch size 找到最佳 tradeoff

## DeepSpped 配置文件

DeepSpeed 可以与 `Trainer` 类一起定义训练，需要配置文件。当执行训练脚本时，DeepSpeed 记录从 Trainer 收到的配置，并打印到控制台，用户可以看到具体使用到的配置。

完整的 DeepSpeed 配置选项参考 [DeepSpeed Configuration JSON reference](https://www.deepspeed.ai/docs/config-json/)。可以查看关于实践例子，参考仓库 [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples)。

在训练时，在命令行场景，把 DeepSpeed 配置文件传递为对应 JSON 文件的路径。如果在 `Trainer` 中设置，可以传递为一个字典对象。

## DeepSpeed 和 Trainer 参数

配置参数包含三种类型：
1. `Trainer` 和 DeepSpeed 共享的配置参数，当它们定义冲突时，难以定义错误。所以，共享参数只在 `Trainer` 的命令行参数中定义和传入。
2. 一些配置参数自动继承自模型参数，不再手动调整。`Trainer` 使用配置值 `auto` 来确定正确或有效的值。也可以显示设置自己的配置参数，但要确定，`Trainer` 参数和 DeepSpeed 配置参数一直。不匹配会导致训练失败，且难以定位。
3. 一些 DeepSpeed 专有的配置参数，只用手动设置以满足训练需求。

修改 DeepSpeed 配置和编辑 `TrainingArguments` 方式如下：
1. 创建或加载 DeepSpeed 配置，作为主配置。
2. 基于 DeepSpeed 配置值来创建 `TrainingArguments` 对象。

一些值，比如 `scheduler.params.total_num_steps` 由 `Training` 在训练时计算。

## ZeRO 配置

ZeRO 的三个 stage 对应三个配置。Stage 1 尽管最快，但在规模化上不考虑它。配置简单。主要关注 stages 2 和 3。`zero_optimization` 配置包含所有选项，决定启动什么和如何配置。更多配置参考官方的 JSON 参考。
- Stage 1: 优化器状态分片。
- Stage 2: 梯度分片（推荐显存不足时使用）。
- Stage 3: 参数分片（适合极大模型，需配合offload）。

### ZeRO-1：普遍不感兴趣

跨越 GPUs 对优化器状态分片 (shareds)，获取一点点速度提升。配置文件类似：

```json
{
    "zero_optimization": {
        "stage": 1
    }
}
```

### ZeRO-2
跨越 GPUs 切片优化器和梯度。主要用于训练，其特性不适合推理。性能相关的重要配置有：
- `offload_optimizer` 需要启动，可以减少 GPU 内存使用。
- `overlap_comm` 设为 `true` 时，增加 GPU 内存消耗，以减少所有 allreduce 的延迟。使用 4.5x 的 `allgather_bucket_size` 和 `reduce_bucket_size` 值。例子中，设为 `5e8` 代表需要 9GB 的 GPU 内存。如果我们的显存低于 8GB，则减少 `overlap_comm` 来避免 OOM 错误。
- `allgather_bucket_size` 和 `reduce_bucket_size` 以显存作为代价，提高通信速度。此值越小，通信越慢，显存越节省。需要我们做平衡，比如更大的 batch size 比慢一些更重要。
- `round_roboin_gradients` 在 DeepSpeed 0.4.4 版本可以用来做 CPU offloading。它并行地执行梯度拷贝到 CPU 内存，此行为发生在细粒度梯度分区 (fine-grained gradient partitioning) 的 ranks 之间。性能因梯度累积步 (拷贝了更多的优化器步) 或 GPU 数量 (增加了并行) 而提升。

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true,
        "round_robin_gradients": true
    }
}
```

### ZeRO-3

跨越 GPUs 切片优化器、梯度和参数。不仅可训练，也可用于推理，因为把模型加载到了多个 GPU 中。重要参数：
- `device: "cpu"` 在显存耗尽情况下，CPU 内存闲余时有用，将模型参数 offload 到 CPU。
- `pin_memory: true` 提升吞吐量，但其他进程获取的内存更少，因为 pinned memory 给特定过程专门使用，并且比普通 CPU 内存访问快得更多 (猜测是因为缓存问题，置换更少)。
- `stage3_max_live_parameters` 设置了上限，决定多少参数需要保持在 GPU。在 OOM 时减小它。
- `stage3_max_reuse_distance` 决定一个参数在未来什么时候再次使用，帮助决定是否把参数置换出内存或保存。如果参数再次使用 (如果值小于 `stage3_max_reuse_distance`，还没被置换出去)，则保留它来减少通信开销。它十分有用，当 activation checkpointing 启用时，我们需要在 forward 重计算 (recompute) 直到 backward 传递时保留参数。在 OOM 时减小它。
- `stage3_gather_16bit_weights_on_model_save` 在保存模型时 consolidates fp16 权重。对大模型和多 GPUs 情况，内存和速度方面开销较大。在 resuming 训练时启用它为 `true`。
- `sub_group_size` 控制优化器迭代时哪些参数需要更新。参数分组到 buckets 中，每个大小即 `sub_group_size`，每个 bucket 只会在一次更新。当与 NVMe offload 一起使用时，它决定优化迭代时模型何时移动和移出 CPU 内存。这避免 CPU 内存用尽。当没有使用 NVMe offload 时设置为默认值，而我们可以修改，只要：
    1. 在优化迭代时遇见 OOM，减少它，从而减少临时 buffer 的内存使用。
    2. 优化迭代需要很长时间，增加它，从而增加临时 buffer，提升带宽使用。
- `reduce_bucket_size`, `stage3_prefetch_bucket_size`, `stage3_param_persistence_threshold` 取决于模型 hidden size。推荐设置为 `auto`，由 `Trainer` 自动决定。

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

#### 加载大模型
可以使用 `deepspeed.zero.Init` 上下文管理器来快速初始化模型：

```py
from transformers import T5ForConditionalGeneration, T5Config
import deepspeed

with deepspeed.zero.Init():
    config = T5Config.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration(config)
```

对于预训练的模型，DeepSpeed 配置文件需要在 `TrainingArguments` 中设置 `is_deepspeed_zero3_enabled: true`，以启用 ZeRO 配置。在调用模型的方法 `from_pretrained()` 前，`TrainingArguments` 对象必须创建完成。

```py
from transformers import AutoModel, Trainer, TrainingArguments

training_args = TrainingArguments(..., deepspeed=ds_config)
model = AutoModel.from_pretrained("google-t5/t5-small")
trainer = Trainer(model=model, args=training_args, ...)
```

#### 需要 ZeRO-3 的场景
就算使用 fp16 权重也不能加载模型到一张 GPU，那么我们需要 ZeRO-3。加载后，需要在 `from_pretrained()` 中指定 `torch_dtype=torch.float16`。

有多张 GPU 时，但是单张 GPU 并不包含所有参数，除法它的参数当前在执行层。为了马上访问所有层的所有参数，比如加载模型时的 `from_pretrained()`，一次加载一层，并且在所有 GPU 中分区 (partitioning)。如此做，是因为模型非常大，不可能在单张 GPU 加载，只能分布式地加载。

如果遇到模型参数比如 `tensor([1.])`，或参数大小是 1，而非多维，这代表着参数被切片了，它是 ZeRO-3 placeholder。

```py
tensor([1.0], device="cuda:0", dtype=torch.float16, requires_grad=True)
```

更多关于使用 ZeRO-3 初始化初始化大模型和访问参数的内容，参考 [Constructing Massive Models](https://deepspeed.readthedocs.io/en/latest/zero3.html#constructing-massive-models) 和 [Gathering Parameters](https://deepspeed.readthedocs.io/en/latest/zero3.html#gathering-parameters) 指南。

## NVMe 配置
[ZeRO-Infinity](https://hf.co/papers/2104.07857) 允许 offloading 模型状态到 CPU 或 NVMe 来节省显存。高明的分区 (partitioning) 和 tiling 算法允许每张 GPU 在 offloading 时发送和接受小数量数据，使得现代 NVMe 容下大模型。ZeRO-Infinity 需要 ZeRO-3。

取决于 CPU 和 NVMe 内存可获取情况，可以 offload [优化器状态](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading) 和 [参数](https://www.deepspeed.ai/docs/config-json/#parameter-offloading)。需要保证 `nvme_path` 指定一个 NVMe 设备，因为与硬盘和 SSD 一起工作时会极大减缓速度。最后，[运行 benchmark](https://github.com/deepspeedai/DeepSpeed/issues/998) 决定最优的 `aio` 配置。

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

## DeepSpeed 特性

还有更多的特性。

### Activation/gradient checkpointing

Activation and gradient checkpointing 对速度和显存做了权衡，以解决 OOM 问题或增加 batch size 来获取更好表现。启用方法如下：
1. 对于 Hugging Face 模型，调用 `model.gradient_checkpointing_enable()` 或在 `Trainer` 传入 `--gradient_checkpointing`。
2. 对于非 Hugging Face 模型，使用 DeepSpeed 的 [Activation Checkpointing API](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html)。

### 精度

DeepSpeed 支持 fp32, fp16, bf16 混合精度。

#### fp32

如果模型在混合精度上表现不佳，比如未在混合精度下预训练，可能遇到 overflow/underflow，导致 NaN loss。于是只能使用完全的 fp32 精度。

```json
{
    "fp16": {
        "enabled": false
    }
}
```

#### fp16

配置 PyTorch AMP-like fp16 混合精度以减少内存使用和加速训练。根据 `args.fp16_backend`，`Trainer` 自动启用或禁用 fp16。可以通过命令行参数: `--fp16`, `--fp16_backend amp` or `--fp16_full_eval`。

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

对于额外选项，参考 [FP16 Training Options](https://www.deepspeed.ai/docs/config-json/#fp16-training-options)。

如果需要支持 Apex-like fp16 混合精度，设置一下内容为 `auto` 即可。或者由 `Trainer` 自动配置 `amp`，根据 `args.fp16_backend` and `args.fp16_opt_level` 的值。也可从命令行传入: `--fp16`, `--fp16_backend apex` or `--fp16_opt_level 01`。

```json
{
    "amp": {
        "enabled": "auto",
        "opt_level": "auto"
    }
}
```

#### bf16

使用 bf16，需要至少 DeepSpeed==0.6.0.bf16 有着和 fp32 一样的 dynamic range，且不需要 loss scaling。然而，使用 bf16 且 [gradient accumulation](https://huggingface.co/docs/transformers/v4.49.0/en/deepspeed?precision=bf16&zero-config=ZeRO-3&opt-sched=scheduler#gradient-accumulation) 时，梯度以 bf16 方式累积，可能并非期望，因为此格式的低精度可能造成带有损失的累积。

bf16 可在配置文件设置，也可在命令行传入为: `--bf16` or `--bf16_full_eval`。

```json
{
    "bf16": {
        "enabled": "auto"
    }
}
```

### Optimizer and scheduler

DeepSpeed 和 Transformers 优化器和 scheduler 可以混用和匹配，只要别启用 `offload_optimizer`。否则，可以使用非 DeepSpeed 优化器 (排除 LAMB)，只要包含 CPU 和 GPU 实现版本。

优化器和调度器参数的配置文件可以从命令行设置，以避免难以发现的误差。

#### optimizer

DeepSpeed 提供一些 [优化器](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters) (Adam, AdamW, OneBitAdam, and LAMB)，但我们可以使用 PyTorch import 而来的其他优化器。如果不配置到文件中，`Trainer` 自动选择 AdamW，并使用提供的参数值或默认值，根据命令行提供的如下参数：`lr`, `adam_beta1`, `adam_beta2`, `adam_epsilon`, `weight_decay`。

可以设为 `auto` 或手动输入：

```json
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": "auto",
         "betas": "auto",
         "eps": "auto",
         "weight_decay": "auto"
       }
   }
}
```

#### scheduler

DeepSpeed 支持 LRRangeTest, OneCycle, WarmupLR 和 WarmupDecayLR 学习率 [schedulers](https://www.deepspeed.ai/docs/config-json/#scheduler-parameters)。

Transformers 和 DeepSpeed 提供两个相同的 schedulers：
- WarmupLR 与 Transformers 中的 `--lr_scheduler_type constant_with_warmup` 相同。
- WarmupDecayLR 与 Transformer 中 `--lr_scheduler_type linear` 相同，这是 Transformers 默认的。

如果不在 json 中配置，`Trainer` 自动选择 WarmupDecayLR，并且使用提供的值或默认值，根据命令行参数来决定: `warmup_min_lr`, `warmup_max_lr`, `warmup_num_steps`, `total_num_steps` (如果 `max_steps` 未提供，则自动计算)。

可以设为 `auto` 或手动输入提供：

```json
{
   "scheduler": {
         "type": "WarmupDecayLR",
         "params": {
             "total_num_steps": "auto",
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     }
}
```

### Batch size

可自动配置或显示配置。设置为 `"auto"` 时，`Trainer` 设置 `train_micro_batch_size_per_gpu` 的值为 `args.per_device_train_batch_size`，设置 `train_batch_size` 为 `args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps`。

```json
{
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto"
}
```

### Gradient accumulation

自动配置或显示配置。设置为 `"auto"` 时，`Trainer` 设置它为 `args.gradient_accumulation_steps`。

```json
{
    "gradient_accumulation_steps": "auto"
}
```

### Gradient clipping

自动配置或显示配置。设置为 `"auto"` 时，`Trainer` 设置它为 `args.max_grad_norm`。

```json
{
    "gradient_clipping": "auto"
}
```

### Communication data type

对于通信集合，比如 reduction, gathering 和 scattering 操作，使用了单独的数据类型。

聚集（gather）和分散（scatter）操作所有聚集和分散操作使用与原始数据相同的数据类型执行。例如：使用 bf16 训练时，数据也会以 bf16 格式聚集，因为这是无损操作。

归约（reduce）操作归约操作是有损的。如多 GPU 梯度平均。使用 fp16 或 bf16 通信时更易出现精度损失，因为低精度数值累加不精确。bf16 的精度比 fp16 更低。因此默认使用 fp16 进行归约操作，梯度平均的损失较小。

配置通信数据类型时，由配置文件的 `communication_data_type` 参数指定。比如，选择 fp32 会引入较小的开销，但确保 recution 操作以 fp32 累积，后续再降级到对应的半精度类型上：

```json
{
    "communication_data_type": "fp32"
}
```


### Universal checkpointing

[Universal Checkpointing](https://www.deepspeed.ai/tutorials/universal-checkpointing) 是高效且灵活的特性，用于保存和加载模型检查点。使得模型训练继续启动和微调适用于不同模型架构，并行技术和训练配置变得流畅好用。

使用带有万能检查点的重启训练，只需要设置 [load_universal](https://www.deepspeed.ai/docs/config-json/#checkpoint-options) 为 `true`：

```json
{
    "checkpoint": {
        "load_universal": true
    }
}
```

### Save model weights

保存主要的全精度 fp32 权重到自定义的检查点优化器文件，模式通常为 `global_step*/*optim_states.pt`，保存与普通检查点。

#### fp16

ZeRO-2 训练的模型以 fp16 保存了 pytorch_model.bin 权重。为了以权重为 fp16 保存 ZeRO-3 训练的模型，需要设置 `"stage3_gather_16bit_weights_on_model_save":true`，因为模型权重被 partitioned 到多张 GPUs。此外，`Trainer` 不会保存权重为 fp16，不会创建 pytorch_model.bin 文件。因为 DeepSpeed 的 state_dict 包含了 placeholder，而非真实的权重，所以不会加载它们。

```json
{
    "zero_optimization": {
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

#### fp32

训练时，一般不该保存，因为太消耗内存。通常在训练完成后保存 fp32 权重。如果有空余的 CPU 内存，也可保存。

TODO。

## Ref and Tag

[Huggingface: DeepSpeed 集成](https://huggingface.co/docs/transformers/v4.49.0/en/deepspeed)
[Huggingface: DeepSpeed](https://huggingface.co/docs/transformers/v4.49.0/en/deepspeed)
[DeepSpeed: Training Overview and Features](https://www.deepspeed.ai/training/)

[[llm.huggingface.transformers_Trainer]]
[[llm.huggingface.Transformers库用法]]

#Transformer
#VLA