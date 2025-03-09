---
id: us3phg4jcf3ej4lpymsyu6q
title: DexGraspVLA_复现
desc: ''
updated: 1741521477801
created: 1741144146461
---

## 数据

数据集参考 MaskImageDataset。接收参数 zarr_paths 是路径字符串列表，随后根据每个路径，获取 StreamingReplayBuffer 实例，并保存在 self.replay_buffers 列表。

### StreamingReplayBuffer

继承 ReplayBuffer。但是在构造函数没有调用 `super().__init__()`，父类没有执行初始化，父类的属性不会被正确初始化，于是不能访问。但是 StreamingReplayBuffer 自己重新组织了字段，覆盖了属性访问。在创建时，使用 classmethod 的 copy_from_path() 方法，临时赋予新的属性比如 zarr_path。

#### copy_from_path(cls, path, keys=None)

构造一个空的 StreamingReplayBuffer 对象，再从 zarr_path 读取 data 和 meta 数据进来。

首先是 meta 部分，一般保存了 episode_ends，是一个 <class zarr.core.Array>，长度是 num_episodes。

data 部分，则包含了 action, rgbm, right_cam_img, right_state，类型也是 Array。对于相机部分，比如 rgbm (RGB 不分图像和掩码部分) 和 right_cam_img (仅 RGB)，使用 ZarrImageReference 来封装。对于其它的，比如 action 和 right_state，使用原来的数据，使用切片访问便可得到 np.ndarry 对象。于是，从 zarr 读取数据后，全部都转换为了 np 对象，或 ZarrImageReference 对象。

## Sampler

buffer_start_idx, buffer_end_idx 指出了 episode 在 buffer 中的区间。sample_start_idx, sample_end_idx 指出了具体每次训练时，每个时间步 t 对应的 horizon 区间。有可能 start_idx < 0，这在 n_obs_step > 1 时会出现，使用复制和填充第一个观察来处理。末尾部分同理。

sample_sequence() 方法最终返回字典，每个 key 对应的 value 为 shape (horizon_len, *data_shape)。比如图像是 (640, 480, 3)，对应 (horizon_len, 640, 480, 3)。

## MaskImageDataset

相机分辨原来是 (640, 480)，处理 rgbm 时，在 _process_mask_image_batch() 方法中，将图像的 channel 轴转置到第二维，得到形状 (T, 3, H, W)。使用 torchvision.transform.interpolate() 插值，把图像 resize 到 (518,518)。

## ObsEncoder

数据送给 ObsEncoder 模块之前，头部和腕部图像 resize 为 518x518x3。在 grasp.yaml:shape_meta 可看见。

配置参考 controller/config/train_dexgraspvla_controller_workspace.yaml:obs_encoder，shape_meta 来自 controller/config/task/grasp.yaml 中的 shape_meta。在 model_config 中，如果 head 和 wrist 没有指定 local_weights_path，即 null，则使用 torch.hub.load() 加载，使用如下：

```py
self.dino_head = torch.hub.load(
    "facebookresearch/dinov2",
    model_config['head']['model_type'], # 比如 dinov2_vits14
    pretrained=True
)
# 冻结参数。wrist 同理
self.dino_head.eval()
```

下载位置放置到 ~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth 下。

[DINOv2 Github](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-backbones-via-pytorch-hub) 指出了加载的方式。其中DINOv2 的 embedding 如下：
- 384 for ViT-S.
- 768 for ViT-B.
- 1024 for ViT-L.
- 1536 for ViT-g.

处理 mask 时，首先对每个 14x14 的 patch 使用 1-channel 的 2D 卷积，实现时，使用 nn.Conv2d(1, head_feature_dim, kernel_size=14, stride=14)，即每 14x14 代表一个 patch，提取出来 head_feature_dim 频道的特征图。获取 head_feature_dim 的维度后，即 (b, n, c) 的形状，再送入 4 层的 TransformerEncoder。

### forward_head()

参数 rgbm_data 是 (B,T,4,H,W)，其中，对应 (B,T,4,518,518)。取出 mask_data 为 (B,T,1,518,518)。传给 self.mask_process_net 网络时，由于网络的 patch_embed 使用了 Conv2d，所以对 mask_data reshape 为 ("B T ... -> (B T) ...")。mask 提取特征为 (B*T, num_patches, head_feature_dim)。其中，使用了 14 作为 patch_size，于是 num_patches = patch_size^2 = (518 // 14)^2 = 37*37 = 1369，得到 (B*T, 1369, head_feature_dim)。

### forward()

输入：

    Input:
    obs_dict = {
        'rgbm': (B,T,4,H,W),      # Head camera RGBM image
        'right_cam_img': (B,T,3,H,W), # Wrist camera RGB image
        'right_state': (B,T,13)    # Robot arm state
    }
    Output:
    embeddings: (B,T*(num_patches*2+1),feature_dim) # Concatenate all features along sequence length dimension
                                                    # head and wrist each output T*num_patches features
                                                    # state outputs T features

分别对头部、腕部和状态提取特征，形状分别为 (batch_size, T * 1369, dim), (batch_size, T * 1369, dim), (batch_size, T, dim)。最后，在第 1 维拼接，即 rearrange([...], "N B C D -> B (N C) D")。得到 (B, T*(num_patches * 2 + 1), feature_dim)。作为 obs_tokens。可以直接送给 TransformerForActionDiffusion。

output_shape() 方法不止返回特征维度，还返回头部、腕部和状态的特征长度，即每部分分别对应的 [1369, 1369, 1]。用于分辨特征中的位置。在 use_attn_mask 时起作用。

## DexGraspVLAController

### 配置和初始化

关于输入动作和输出动作维度，关注参数 shape_meta（定义在 grasp.yaml），决定 Transformer 的 action_shape 和 action_horizon，分别为 13 和 n_action_steps (64)。

TransformerForActionDiffusion 使用的 n_emb 与 ObsEncoder 使用的 feature_dim 一致。ObsEncoder 编码后，输出的 shape 为 (batch_size, 2739, feature_dim)，feature_dim 根据头部和腕部相机中，使用的 DINOv2 输出特征来决定，取最大者。2739 分别是图像每个 patch 的特征，还有状态的特征。

### model: TransformerForActionDiffusion

传入的 max_cond_tokens 为 T*(num_patches * 2 + 1) + 1，额外的 1 是为了 timestep，初始化 cond_pos_emb，作为可学习的位置编码。

归一化器从训练的数据集加载并获取。并进一步设置给模型。

#### forward()

输入：

- sample: (B,T,input_dim)
- timestep: (B,) or int, diffusion step
- cond: (B,N,n_emb)，其中，N=T*(num_patches * 2 + 1)。
- output: (B,T,input_dim)
- return: 
    - if gen_attn_map:
        output tensor (B,T,input_dim), attention_maps List[(B,num_heads,T,L)]
    - else:
        output tensor (B,T,input_dim), None

第一个参数为 sample，对应不断加噪/去噪的动作，预测加噪的参数。随后，与可学习位置编码相加，得到输入 x。

把时间步与 cond 拼接，得到 (B,N+1,n_emb)，加上可学习位置编码，得到最终的条件编码 cond_emb，传给每个 blocks。在 RDTBlock 中，使用 CrossAttention，x 作为 q，条件作为 kv。

传入 x 与 cond_emb 给 block，计算后预测最终的噪声。

### compute_loss()：训练

传入的 batch 中，obs 部分参考 Dataset，包含 rgbm, right_cam_img 和 right_state。action 则是 13 维。由 ObsEncoder 实例编码，得到 obs_token，形状是 (B, N, n_emb)。

Controller 的 forward() 方法简单的调用 compute_loss()。如果需要预测，应当使用方法 predict_action()。

### predict_action() 和 conditional_sample()：预测动作

condition_data 和 condition_mask 确保指定索引范围的内容为 condition_data 内容，其余部分则由扩散去噪生成。condition_data 和 condition_mask 全部为 torch.zeros()。condition_mask 为 torch.bool 类型，默认全部为 False，全部由扩散的逆向生成动作。具体由 conditional_sample() 调用 TransformerForActionDiffusion 模型预测噪声。

## 双卡 4090 训练注意事项

16 的 batch size，每张卡消耗显约 1600MB。约 2~3 分钟一个 epoch iter。bach size 选择 24，每张显卡消耗约 21800MB，约一分半一个 epoch iter。

## Planner

Planner 只有一个方法，request_trask()。根据 task_name 参数，决定 prompt 是什么。可以选择如下：
- "classify_user_prompt"：
- "decompose_user_prompt"：
- "generate_instruction"：
- ""mark_bounding_box"：
- "check_grasp_success"：
- "check_instruction_complete"：
- "check_user_prompt_complete"：

### Prompts 设计

使用 prompts 提示 planner。作者设计的 prompts 分为几类，对应参数中的 task_name 任务。主要包含功能如下：
- understanding the user prompt
- proposing an object as the current grasping instruction，建议当前指令对应的抓取的物体
- marking the target object bounding box
- checking if the grasp has succeeded
- assessing whether the current instruction is completed
- evaluating whether the entire user prompt is fully fulfilled

#### classify_user_prompt：辨别任务类型

用户提供了 prompt p，比如 "grab the green cup" 等。系统 prompt 需要决定 p 是否指定了具体物体。如果指定具体物体，后续则会操作它。否则，比如清理桌面，则会推理并一个个处理。

```py
if task_name == "classify_user_prompt":
    prompt = f"""
    Analyze the following user prompt: {instruction}

    User prompt types:
    - Type I (return True): User prompts with any specific descriptions
    Examples: 
    * Color-based: "green objects"
    * Position-based: "objects from the right"
    * Property-based: "all cups"
    * Combination: "the red cup on the left"

    - Type II (return False): Abstract prompts without any object descriptions
    Examples: "clear the table", "clean up", "remove everything"

    Please determine:
    - Is this a Type I prompt? (True/False)
    - Provide your reasoning

    Return format:
    True/False: your reasoning

    Examples:
    - "grab the green cup" -> True: Contains specific object (cup) and property (green)
    - "clear the table" -> False: No specific object characteristics mentioned
    """
elif ...
```

VLM 的回答包含：
- "Type I"
- "Type II"
- 两者不包含则报错

#### decompose_user_prompt：为 Type I 分解任务

辨别用户提示词后，如果是 Type I，则进一步生成有序的抓取指令列表。使用如下的系统 prompt 和 head-camera 图像送给模型。

```py
elif task_name == "decompose_user_prompt":
    prompt = f"""
    For user prompt: {instruction}
    Process:
    1. Analyze the user prompt and image together:
    - Match user prompt descriptions with visible objects in the image
    - If a description (e.g., "green objects") matches multiple objects, include all matching objects
    - Verify each mentioned object actually exists in the image

    2. Based on the robot arm's position (right edge of the screen) and table layout
    3. Determine the most efficient grasping sequence
    4. Generate a reordered list of objects to grasp
    
    Requirements:
    - Only include objects mentioned in the original user prompt
    - Keep position information for each object
    - Return as a list, ordered by grasping sequence

    Expected output format:
    ["object with position 1", "object with position 2", ...]
    """
elif ...
```

VLM 的回答包含：
- "[\"<带位置的物体1>\", \"<带位置的物体2>\", ...]"
- 不包含则报错

#### generate_instruction：为 Type II 生成

对于 Typle II，用户需要抓取桌上所有物体，planner 选择最优的目标作为指令。选择要根据当前桌上剩余物体。于是，使用如下系统 prompt 和 head-camera 图像。

```py
elif task_name == "generate_instruction":
    prompt = f"""
    Analyze the current desktop layout and select the most suitable object to grasp, considering the following factors:

    Grasping Strategy:
    1. The robotic arm is positioned on the far right (outside the frame)
    2. Grasping Priority Order:
        - Prioritize objects on the right to avoid knocking over other objects during later operations
        - Then consider objects in the middle
        - Finally, consider objects on the left
    3. Accessibility Analysis:
        - Relative positions between objects
        - Potential obstacles
        - Whether the grasping path might interfere with other objects

    Please provide your response in the following JSON format:
    {{
        "analysis": {{
            "priority_consideration": "Explanation of why this object has priority",
            "accessibility": "Analysis of object's accessibility",
            "risk_assessment": "Potential risks in grasping this object"
        }},
        "target": "A comprehensive description of the target object 
        (e.g., 'the blue cube on the far right of the desktop, next to the red cylinder')"
    }}

    Ensure the output is in valid JSON format.
    Note: The 'target' field should ONLY contain the object's color, shape, and position in a natural, flowing sentence. Do not include any analysis or reasoning in this field.
    """
elif ...
```

注意，两个花括号中，第一个代表 f-format 的输入，第二个代表字典。内部的字典也一样。"analysis" 对应的 {{ 也需要双花括号，否则不能够在字符串中展现出来。

希望 VLM 返回的内容包含 JSON 格式的序列，根据 \`\`\`json 来识别 JSON 内容起始，根据 \`\`\` 识别 JSON 内容终止。

#### mark_bounding_box

对于每条抓取指令 l，planner 标注边框，使用如下系统的 prompt 和头部相机。

```py
elif task_name == "mark_bounding_box":
    prompt = f"""
    Analyze the image and identify the best matching object with the description: {instruction}.
    Instructions for object analysis:
    1. Select ONE object that best matches the description
    2. For the selected object, provide:
    - A concise label, object name (3-4 words max)
    - A detailed description (position, color, shape, context)
    - Accurate bbox coordinates

    Required JSON format with an example:
    ```json
    {{
        "bbox_2d": [x1, y1, x2, y2],
        "label": "green cup",  # Keep this very brief (3-4 words)
        "description": "A cylindrical green ceramic cup located on the right side of the wooden table, next to the laptop"  # Detailed description
    }}
    ```

    Critical requirements:
    - Return EXACTLY ONE object
    - "label": Must be brief (3-4 words) for quick reference
    - "description": Must be detailed and include spatial context
    - Use single JSON object format, not an array
    - Ensure bbox coordinates are within image boundaries
    """
elif ...
```

VLM 返回 JSON 格式，解析类似 generate_instruction。

#### check_grasp_success

执行期间，planner 验证物体会否成功抓取，使用如下系统提示词和头部图像。

```py
elif task_name == "check_grasp_success":
    prompt = f"""
    Analyze the image and determine if the robotic arm has successfully grasped an object:
    1. Observe the spatial relationship between the robotic hand and the object
    2. Output format: explain your reasoning, then conclude with a boolean value (True=grasped, False=not grasped)
    """
elif ...
```

VLM 返回要包含 true 或 false，否则报错。

#### check_instruction_complete

当抓取尝试结束时，机器人会重置初始状态，planner 检查当前指令是否完成，使用如下系统提示词和头部图像。

```py
elif task_name == "check_instruction_complete":
    prompt = f"""
    Please check whether {instruction} exists on the desktop. If it does not exist, output True; otherwise, output False.
    """
elif ...
```

VLM 返回要包含 true 或 false，否则报错。

#### check_user_prompt_complete

对于 Type I 的用户提示词 p，planner 根据是否抓取成功来决定是否完成任务。对于 Type II 用户提示词，planner 在每次抓取尝试后检查提示词是否完全执行。使用如下系统提示词和头部图像。

```py
elif task_name == "check_user_prompt_complete":
    prompt = """
    Please analyze the table in the image:

    Requirements:
    - Only detect physical objects with noticeable height/thickness (3D objects)
    - Exclude from consideration:
    * Flat items (papers, tablecloths, mats)
    * Light projections
    * Shadows
    * Surface patterns or textures

    Return format:
    - True: if the table is empty of 3D objects
    - False: if there are any 3D objects, followed by their names

    Example responses:
    True  (for empty table)
    False: cup, bottle, plate  (for table with objects)
    """
else:
    raise ValueError(f"The task_name {task_name} is not a valid task name.")
```

VLM 返回要包含 true 或 false，否则报错。

## 适配到 UMI 的数据

首先，改造 Dataset 和 ObsEncoder。具体数据格式，可以参考 UMI 的配置文件。

### UMIDataset


## Ref and Tag

[[robotics.DexGraspVLA]]

