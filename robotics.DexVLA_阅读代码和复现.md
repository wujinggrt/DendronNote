---
id: 4gb9ottxmfh95i6654zy8hq
title: DexVLA_阅读代码和复现
desc: ''
updated: 1740660782718
created: 1740053039805
---

## 数据准备

### 数据格式

#### HDF5 格式

与 act 工作的数据格式一致，转换数据为 HDF5 格式。作者使用 rlds_to_h5py 转换，格式具体如下：
```angular2html
# h5 data structure
root
  |-action (100,10)
  |-language_raw (1,)
  |-substep_reasonings (100,)
  |-observations
      |-images # multi-view
          |-left (100,480,640,3)
          |-right (100,480,640,3)
          |-wrist (100,480,640,3)
      |-joint_positions (100,7)
      |-qpos (100,7)
      |-qvel (100,7)
```

#### 字段解释

内容来自猜测和结合 DeepSeek：
- language_raw (1,) —— 原始语言指令，如折叠衬衫。所以当前是一个任务，有一个语言指令，如下每个时间步对应一个子步骤。即此任务的 horizon 为 100。
- substep_reasonings (100,) —— 子步骤推理，每个时间步对应每个子步骤描述。相比 language_raw，


| 字段 | 形状 | 描述 |
| --- | --- | --- |
| action | (100,10) | 100 表示时间步数，10 表示动作维度。 |
| substep_reasonings | (100,) | 100 | 每个时间步对应一个子步骤推理。 |
| observations |  | 表示时间步数，其他维度表示观测数据。 |
| - images | | 表示时间步数，其他维度表示图像数据。 |
| \|- left | (100,480,640,3) | 100 | 表示时间步数，480x640 表示图像分辨率。 |
| \|- right | (100,480,640,3) | 100 | 表示时间步数，480x640 表示图像分辨率。 |
| \|- wrist | (100,480,640,3) | 100 | 表示时间步数，480x640 表示图像分辨率。 |
| - joint_positions | (100,7) | 100 | 表示时间步数，7 表示关节位置维度。 |
| - qpos | (100,7) | 100 | 表示时间步数，7 表示关节位置维度。 |
| - qvel | (100,7) | 100 | 表示时间步数，7 表示关节速度维度。 |

joint_positions 和 qpos 关系：

| 维度 | joint_positions | qpos |
| ---- | --- | --- |
| 定义     | 关节角度或关节位置。 | 广义坐标位置，可能包含更多自由度信息。 |
| 用途     | 描述机器人的关节状态。  | 描述机器人系统的完整状态。 |
| 数据范围 | 通常仅包含关节角度。  | 可能包含关节角度、末端执行器位置等信息。 |
| 示例     | 7 自由度的机械臂的关节角度。 | 7 自由度的机械臂的关节角度 + 末端执行器位置。 |

### act-plus-plus 数据格式
为了了解数据格式，查看仿真环境下的 mobile aloha 如何数据如何组织：
```py
# record_sim_episodes.py
def main(args):
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']

    for episode_idx in range(num_episodes):
        ...
        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """
        # 由于是双臂，所以是 14 对应 7*2
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

```

在仿真下，收集 50 episodes 例子：
```console
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir <data save dir> --num_episodes 50
```

### 生成 h5 格式数据
```py
def generate_h5(
    obs_replay, action_replay, cfg, total_traj_cnt, act_root_dir_path, edit_flag
):
    data_dict = {
        "/observations/qpos": obs_replay["qpos"],
        "/observations/qvel": obs_replay["qvel"],
        "/action": action_replay,
        "is_edited": np.array(edit_flag),
    }
    for cam_name in cfg["camera_names"]:
        data_dict[f"/observations/images/{cam_name}"] = obs_replay["images"][cam_name]

    max_timesteps = len(data_dict["/observations/qpos"])
    print(f"max_timesteps: {max_timesteps}")
    data_dir = act_root_dir_path

    dataset_path = os.path.join(data_dir, f"episode_{total_traj_cnt}")
    # save the data, 2GB cache，chunks 可以利用这些缓存
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = True
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in cfg["camera_names"]:
            _ = image.create_dataset(
                cam_name,
                (max_timesteps, cfg["cam_height"], cfg["cam_width"], 3),
                dtype="uint8",
                chunks=(1, cfg["cam_height"], cfg["cam_width"], 3),
            )
        qpos = obs.create_dataset("qpos", (max_timesteps, cfg["state_dim"]))
        qvel = obs.create_dataset("qvel", (max_timesteps, cfg["state_dim"]))
        # image = obs.create_dataset("image", (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
        action = root.create_dataset("action", (max_timesteps, cfg["action_dim"]))
        is_edited = root.create_dataset("is_edited", (1))
        # dt = h5py.special_dtype(vlen=str)
        # dt = h5py.string_dtype()
        # lang_intrs = root.create_dataset('lang_intrs', data=cfg['lang_intrs'], dtype=dt)
        # lang_intrs['/lang_intrs'][...] = cfg['lang_intrs']
        raw_lang = cfg["lang_intrs"]
        distill_bert_lang = cfg["distill_bert_lang"]
        # encoded_lang = cfg['lang_intrs_distilbert']
        root.create_dataset("language_raw", data=[raw_lang])
        root.create_dataset(
            "distill_bert_lang", data=distill_bert_lang.cpu().detach().numpy()
        )
        # root.create_dataset("language_distilbert", data=encoded_lang.cpu().detach().numpy())

        print(f"==== generate h5 ======")
        for name, array in data_dict.items():
            print(f"name: {name}")
            print(f"array: {array.shape}")
            # 以切片来保存
            root[name][...] = array
```

## 数据流向和训练调用关系

训练 VLA 的文件参考 train_vla.py，阶段 2 和阶段 3 的训练都用到它。

train_vla.py:main() 是核心入口，负责数据加载到模型训练的整个流程：
- **初始化与配置加载** —— 加载任务配置，设置随机种子。
- **数据加载与预处理** —— 加载数据集，使用 `Qwen2VLAProcess` 进行多模态数据预处理。
- **模型加载** —— 使用 `ml_utils.load_model` 加载预训练的视觉-语言模型和扩散专家。
- **训练器初始化与训练** —— 初始化 `QWen2VLATrainer`，调用 `trainer.train` 开始训练。
- **保存训练结果** —— 保存数据集的统计信息和训练后的模型状态。

### 任务配置加载

训练数据通过 `TASK_CONFIGS` 加载配置。此字典在 aloha_scripts/constants.py 文件定义，通过添加条目来指定数据加载：

```py
TASK_CONFIGS = {
    'example_tasks': { # for local debug
        'dataset_dir': [
            "/media/rl/HDD/data/data/aloha_data/4_cameras_aloha/folding_shirt"
        ],
        'episode_len': 1000,  
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'] # replacing with your real keys in h5py formatted data
    }
}
...
```

`TASK_CONFIGS` 包括数据集路径 (dataset_dir)、任务时间步数 (episode_len)、相机视角 (camera_names) 等。

```py
def main():
    ...
    task_config = TASK_CONFIGS[all_config['data_args'].task_name]
    dataset_dir = task_config['dataset_dir']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    ...
```

`all_config` 根据 `parse_param()` 解析，设置在 train_vla.py 开头的数据类 `ModelArguments`, `DataArguments`, `TrainingArguments`, `ActionHeadArgument`。在设计方面，这里应该单独梳理为一个文件更合适。回到 main()，`all_config['data_args'].task_name` 的 task_name 由 `DataArguments` 决定，可以看到注释，应当对应到 constants.py 文件：

```py
@dataclass
class DataArguments:
    ...
    task_name: str = field(default="stack_cube_2024_6_2") # task name corresponding to aloha_scripts/constants.py
    ...
```

### 数据加载与预处理

#### 数据加载

使用 `load_data` 函数加载训练和验证数据集。

```py
    # load dataset
    train_dataset, val_dataset, stats, sampler_params = load_data(
        dataset_dir,
        name_filter,
        camera_names,
        all_config["training_args"].per_device_train_batch_size,
        all_config["training_args"].per_device_eval_batch_size,
        all_config["data_args"].chunk_size,
        skip_mirrored_data=all_config["data_args"].skip_mirrored_data,
        config=all_config,
        stats_dir_l=stats_dir,
        rank0_print=rank0_print,
        policy_class=all_config["action_head_args"].policy_head_type,
        sample_weights=sample_weights,
        train_ratio=train_ratio,
        llava_pythia_process=vla_process,
    )
```

#### 数据预处理

使用 `Qwen2VLAProcess` 对多模态数据（图像和语言指令）进行预处理。

```py
    vla_process = Qwen2VLAProcess(
        tokenizer=tokenizer,
        multimodal_processor=multimodal_processor,
        data_args=all_config["data_args"],
        camera_names=camera_names,
    )
```

#### 模型加载

使用 `ml_utils.load_model` 加载预训练的视觉-语言模型（VLM）和扩散专家（Diffusion Expert）。

```py
    # load qwen2_vl tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        all_config["model_args"].model_name_or_path,
    )
    # load qwen2_vl input processor
    multimodal_processor = AutoProcessor.from_pretrained(all_config["model_args"].model_name_or_path)
    # load dexvla model
    model, data_args = ml_utils.load_model(
        config=all_config, qwen2_vla_config=model_config, rank0_print=rank0_print, tokenizer=tokenizer
    )
```

#### 训练器初始化与训练

接下来，模型调用 `train_bc()`，开始准备训练。

```py
def main(all_config=None, model_config=None):
    ...
    best_ckpt_info = train_bc(
        train_dataset=train_dataset,
        model=model,
        val_dataset=val_dataset,
        config=all_config,
        sampler_params=sampler_params,
        tokenizer=tokenizer,
        processor=multimodal_processor,
    )
    ...
```

使用 `Qwen2VLADataCollatorForSupervisedDataset` 对数据进行整理，生成模型输入。

```py
def train_bc(
    train_dataset=None, val_dataset=None, model=None, config=None, sampler_params=None, tokenizer=None, processor=None
):
    """
    Train a behavior cloning model using the QWen2VLA architecture.
    """
    ...
    data_collator = Qwen2VLADataCollatorForSupervisedDataset(
        multimodal_processor=processor, computed_type=compute_dtype, tokenizer=tokenizer, video=video
    )
```

使用 `QWen2VLATrainer` 初始化训练器，传入模型、数据整理器、训练参数等，开始训练。

```py
    model.config.use_cache = True
    model.config.save_pretrained(config["training_args"].output_dir)

    data_module = dict(train_dataset=train_dataset, data_collator=data_collator, eval_dataset=val_dataset)
    trainer = QWen2VLATrainer(
        model=model, tokenizer=tokenizer, args=config["training_args"], sampler_params=sampler_params, **data_module
    )

    trainer.train(resume_from_checkpoint=config["training_args"].resume_from_checkpoint)
```

#### 保存训练结果

保存模型状态和检查点：

```py
def train_bc(...):
    ...
    trainer.save_state()

    model.config.use_cache = True
```

保存统计信息：

```py
def main(...):
    ...
    best_ckpt_info = train_bc(...)

    # exit(0)
    stats_path = os.path.join(all_config["training_args"].output_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
```

pickle 是标准库内容其一，用于序列化和反序列化 Python 对象。

## 参数配置

参考 train_vla.py:def parse_param()。根据数据类 ModelArguments, DataArguments, TrainingArguments, ActionHeadArguments 解析参数。

## ScaleDP

## 训练器 QWen2VLATrainer

参考 qwen2_vla/train/qwen2_vla_trainer.py。

## VLM
使用 [Qwen2-2B-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) 作为主干网络。也许可以尝试 [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)。

模型结构，在 VLM 末尾增加一个 policy head；而 Helix 直接输出 token，当作 policy 模型的 latent vector。

## 把扩散专家接到 Qwen2-VL 上
项目文件 qwen2_vla/models/modeling_qwen2_vla.py 和 qwen2_vla/models/configuration_qwen2_vla.py 改造了 Qwen2-VL 的源码和配置。两个文件都是从 huggingface 的 transformers 库中 transformers/models/qwen2_vl/modeling_qwen2_vl.py 和对应 configuration_qwen2_vla.py 复制而来，并根据需求做出修改。


### 扩散专家与 VLM 的连接

- 输入投影层：在 VLM 模型的输出部分，扩散专家通过一个输入投影层（input_action_proj）将 VLM 的隐藏状态（hidden states）映射到扩散专家的输入空间。这个投影层通常由两个线性层（MLP）组成，带有 LayerNorm 归一化。
- FiLM 层：如果启用了 FiLM（Feature-wise Linear Modulation）机制，扩散专家还会通过 FiLM 层将 VLM 的推理信息（reasoning tokens）注入到扩散专家的动作生成过程中。FiLM 层通过缩放和偏移参数来调整扩散专家的输出。

关键代码片段：

在文件末尾的 Qwen2VLForConditionalGenerationForVLA 中，作者做出了修改。原版千问模型的只有 `self.visual, self.model, self.vocab_size, self.lm_head, self.rope_deltas` 等 fields。作者添加了 `self.padding_side, self.using_file, ...`。

#### 结合扩散专家的 VLA 模型初始化

```py
class Qwen2VLForConditionalGenerationForVLA(Qwen2VLPreTrainedModel, GenerationMixin):
    """
    类属性。这是 Huggingface Transformers 库的特殊属性，要求指定模型需要绑定的权重。
    权重绑定是常见的优化技术，特别是语言模型，输入嵌入层和输出层的权重可以共享，减少模型参数，提高训练效率。
    库的权重绑定通过 `tie_weights()` 方法实现，模型初始化时，库自动查找 _tied_weight_keys，将权重绑定一起。
    本例中，"lm_head.weight" 与 "embedJ_tokens.weight" 绑定在一起。尽管 self.lm_head 定义为
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    实际并未只想新的 nn.Linear 模块，而是嵌入层的对应的权重。
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        ...
        # 初始化 policy_head，对应扩散专家
        # policy_head_config 配置参考 train_vla.py:class ActionHeadArguments，会被解析为类 dict 类型
        if isinstance(config.policy_head_config, dict):
            config.policy_head_config = AutoConfig.for_model(**config.policy_head_config)
        self.policy_head = AutoModel.from_config(config=config.policy_head_config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
        if config.policy_head_config.model_type == "scale_dp_policy":
            self.policy_head.init_weights()
        # 输入投影层，来自于 Fusion 模块
        self.input_action_proj = ActionProjector(config.hidden_size, config.hidden_size)

        # 是否使用 film 来 fusion，默认使用
        if self.using_film:
            # Initialize projection layers and condition modulation layers
            # 嵌入 condition，即文本的 embedding。主要是放缩和偏移。
            self.reasoning_action_proj = ActionProjector(config.hidden_size, config.hidden_size)
            self.reasoning_film = FiLM(feature_dim=config.hidden_size, condition_dim=config.hidden_size)
```

#### 在 forward() 中调用扩散专家

`forward()` 在原版本上做出了修改。主要添加了衔接扩散专家部分。包含将 `hidden_states` 等信息传给扩散专家。注意，输入是 input_ids，ids 通常代表 IDs，identifiers 缩写。

```py
    def forward(
        self,
        input_ids: torch.LongTensor = None, 
        ...
        labels: Optional[torch.LongTensor] = None, 
        ...
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        ...
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # VLA 输出的内容，outputs[0] 代表最后一层影藏状态，即 hidden_states
        # 结构是 (batch_size, seq_len, hidden_size)
        hidden_states = outputs[0]
        if tinyvla: # dex-vla supports tinyvla-style VLA
            return hidden_states

        # 把隐藏状态传给嵌入层，得到 logits。到此 Qwen2-VL 已经完成文本生成。只需要 tokenizer decode 即可得到文本。
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        llm_loss = None
        # 如果传入了 labels，那么与 label 求交叉熵，以训练 VLM
        # 未传入 labels 代表仅推理，loss 为 None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            llm_loss = loss_fct(shift_logits, shift_labels)
        ...
        # 使用 FiLM 融合
        if self.using_film:
            action_hidden_states = self.film_forward(
                labels=labels, input_ids=input_ids, hidden_states=hidden_states
            )
        else:
            action_hidden_states = hidden_states

        ret = self.policy_head(
            actions=actions,
            hidden_states=action_hidden_states,
            states=states,
            is_pad=is_pad,
        )
        loss = {'loss': ret['loss'] + self.llm_loss_weight * llm_loss,
                'llm_loss': llm_loss,
                'action_loss': ret['loss']}
        # 以 Tuple 返回
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        ...
        return Qwen2VLCausalLMOutputWithPast(loss=loss, logits=logits, ...)
```

模型的输出中，`output[0]` 代表最后一层的 hidden_states。根据源码，在 `Qwen2VLModel:forward()` 中，参数 `return_dict` 默认为 None，所以返回 `Tuple`，具体如下：

```py
if not return_dict:
    return tuple(
        v
        for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
        if v is not None
    )
```

`all_hidden_states` 包含了最后一层的 `hidden_states`。

总结，扩散专家使用了动作、动作隐藏状态和状态 states 作为输入。其中，动作隐藏状态 (action_hidden_states) 使用了 Qwen2-VL 的 logits，可能进一步与 labels、输入进行 FiLM fusion。

#### 使用 FiLM 集成 LLM 的 logits

如果指定了配置 `using_film`，则使用 `film_forward()` 把输入、labels 和隐藏状态一起编码，最后输出 action_hidden_states。

```py
    def film_forward(self, labels, input_ids, hidden_states):
        inputs_index = labels[:, :] == -100
        inputs_index = inputs_index.int()

        xor_array = torch.bitwise_xor(inputs_index[:, :-1], inputs_index[:, 1:])
        indexs = torch.argmax((xor_array != 0).float(), dim=1)
        input_embeddings = []
        reasoning_embeddings = []
        identity = []
        for i in range(indexs.shape[0]):
            end = indexs[i] + 1
            temp = input_ids[i] == 151643  # pad token id for qwen2_vl
            start = sum(temp.int())
            input_embeddings.append(
                self.input_action_proj(hidden_states[i, start:end, :])
            )
            identity.append(torch.mean(hidden_states[i, start:end, :], dim=0))

            reasoning_embeddings.append(
                self.reasoning_action_proj(hidden_states[i, end:, :])
            )
        input_embeddings = torch.cat(input_embeddings, dim=0)
        reasoning_embeddings = torch.cat(reasoning_embeddings, dim=0)
        identity = torch.stack(identity)

        action_hidden_states = self.reasoning_film(
            input_embeddings, reasoning_embeddings
        ).unsqueeze(1)

        action_hidden_states = action_hidden_states + identity.unsqueeze(1)
        return action_hidden_states
```

#### 对比原版文件，做出了哪些修改



### 梯度是如何反向传播的

#### 交叉熵

如果传入 `labels` 给 `forward()`，说明正在训练，进一步计算交叉熵。否则，模型只需要推理，`loss` 为 `None`。

## 视觉编码器条件化
两个方案：
- FiLM 层 (CNN 架构)：在 EfficientNet 的隐藏层
- 交叉注意力 (Trasnformer 架构)：在自注意力后插入跨注意力。

## TODO
制作 PPT，复现此项目。

Figure 的 Helix 思路与此十分相似。Helix 使用了较大的 VLM (7B) 作为主干，使用较小的策略模型 (80M) 生成动作。解耦了大模型和小模型。大模型

Helix 和 HiRT 并未开源代码和模型，DexVLA 开源了代码，复现可能更大。但是，在动作生成方面，还是使用了 action head 层动作学习的网络接在 VLM 模型，VLM 使用 Qwen2-3B-VL。

HiRT 发表了论文，解决了 VLM 模型与策略模型生成速度不匹配的情况。主要使用异步的方案，把 VLM 当做一个大型的编码器，编码视觉和自然语言指令。

问题分析：大模型生成较慢，动作策略的小模型生成较快，DexVLA 并没有解决此问题，还是使用同步的方案；泛化场景问题。

可行性：DexVLA 开源，有框架遵循，有复现可能。使用的 VLM 模型是 2B，最近，千问发表了 Qwen2.5-VL 系列。可以使用可能更优秀的 Qwen2.5-3B-Instruct，使用 DeepSpeed，两张显卡猜测能够微调。在数据收集方面，有 pny 做过数据收集，使用的数据格式类似。

下一步打算：先复现，后修改，不断逼近 Helix 的方案。


## 借助 DeepSeek 的 QA

上传论文后，Q 如下：

### Q：分析 train_vla.py Q

- 以下代码是训练阶段 2 和阶段 3 的入口，请总结数据是如何加载和传入训练的。<粘贴了文件内容>
- 请总结 main 函数做了哪些工作

#### Q：以下代码是 DexVLA 项目的 VLA 模型相关文件，作者做出了修改，请问扩散专家是如何接到 VLM 模型的。
- 配置文件qwen2_vla/models/configuration_qwen2_vla.py如下：<文件内容>
- 配置文件qwen2_vla/models/modeling_qwen2_vla.py如下：<文件内容>

#### Q：在 class Qwen2VLForConditionalGenerationForVLA 中，扩散专家是怎么调用的

#### Q：forward() 中的 hidden_states 是什么




## Tag and Ref
[[robotics.DexVLA]]
[[robotics.Helix：用于通才人形机器人控制的_VLM]]
[[robotics.HiRT_使用分层机器人Transformer提示机器人控制]]
[[insights.Robotics]]

#复现
