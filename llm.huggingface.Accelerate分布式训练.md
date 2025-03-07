---
id: a0gljobkcyqtryg0l7dppjn
title: Accelerate分布式训练
desc: ''
updated: 1741347127831
created: 1741345074459
---

accelerate 加速简单易用。

安装 pip install accelerate。

准备加速，只需要将相关的训练对象传递给 prepare() 方法。这包括训练和评估的 DataLoader，模型和优化器：

```py
accelerator = Accelerator(
    log_with="wandb",
    mixed_precision="bf16",  # Enable BF16 mixed precision training
    device_placement=True,
)
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)
```

使用 accelerate 的 backward() 方法替换训练循环中的 loss.backward()：

```py
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

加速训练：

```bash
accelerate launch train.py
```

accelerate launch 还接受参数，比如：
- num_processes NUM_PROCESSES，代表并行运行的进程。通常每个进程对应一个 GPU，一般采用与 GPU 相同数量的 NUM_PROCESSES。
- multi_gpu 代表是否启用分布式 GPU 训练。
- mixed_precision {no,fp16,bp16,fp8}

注意，长参数的 hyphens (--num-processes) 和下划线 (--num_processes) 都可以处理为相同内容。

为了保存模型，需要使用如下方法，后续才方便保存模型。

```py
policy = accelerator.unwrap_model(self.model)
```

## 与 wandb 联合

在 Accelerator 传入参数并初始化：

```py
accelerator = Accelerator(
    log_with="wandb",
    mixed_precision="bf16",  # Enable BF16 mixed precision training
    device_placement=True,
)
accelerator.init_trackers(
    project_name=cfg.logging.project,
    config=OmegaConf.to_container(cfg, resolve=True), # dict
    init_kwargs={"wandb": wandb_cfg},
)
```

随后使用 accelerator.log() 即可。

## 分布式训练的进程间通信

is_main_process 属性和 wait_for_everyone() 方法用于实现通信。

is_main_process 是布尔属性，用于确定当前进程是否为主进程（通常是第一个启动的进程）。这个属性对于执行那些只需要在一个进程中完成的任务特别有用，比如保存模型、记录日志或评估结果等。这样可以避免所有进程同时尝试写入同一个文件或其他资源，从而导致冲突或数据损坏。

最佳实践是只在主进程打 log。

wait_for_everyone() 方法用于确保所有进程都到达了代码中的某一点之后才能继续执行。这在分布式训练中非常重要，尤其是在某些操作（如保存检查点）之前，我们需要确保所有进程都已经完成了它们的工作并且达到了一致的状态。

### 最佳实践

以加载 normalizer 为例。在主进程写入二进制内容，随后各个子进程加载二进制内容，再初始化为 normalizer。使用序列化工具 pickle。

```py
        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, "normalizer.pkl")
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, "wb"))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, "rb"))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
```

## Ref and Tag