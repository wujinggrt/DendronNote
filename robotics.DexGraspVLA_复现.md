---
id: us3phg4jcf3ej4lpymsyu6q
title: DexGraspVLA_复现
desc: ''
updated: 1753000833843
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

### 示例数据及格式

官网给了 zarr 格式的示例。

meta 数据内容和类型如下，注意，制作数据集时，episode_ends 的格式必须是 np.int64，否则 episode_ends 的元素比如 np.uint64 计算时会得到 np.float64，导致 sampler 制作 slice 时，不能以非 int 类型为切片：
```
key is episode_ends, v is <zarr.core.Array '/meta/episode_ends' (51,) int64 read-only>
```

data 数据内容和数据类型如下，其中 uint8 代表 0-255，0-3 代表 rgb，最后一维代表 mask，只取 0 或 1：
```
<class 'zarr.storage.DirectoryStore'>
<zarr.storage.DirectoryStore object at 0x784c73733550>
key is action, v is <zarr.core.Array '/data/action' (3825, 13) float32 read-only>
key is rgbm, v is <zarr.core.Array '/data/rgbm' (3825, 480, 640, 4) uint8 read-only>
key is right_cam_img, v is <zarr.core.Array '/data/right_cam_img' (3825, 480, 640, 3) uint8 read-only>
key is right_state, v is <zarr.core.Array '/data/right_state' (3825, 13) float32 read-only>
```

## Sampler

buffer_start_idx, buffer_end_idx 指出了 episode 在 buffer 中的区间。sample_start_idx, sample_end_idx 指出了具体每次训练时，每个时间步 t 对应的 horizon 区间。有可能 start_idx < 0，这在 n_obs_step > 1 时会出现，使用复制和填充第一个观察来处理。末尾部分同理。

sample_sequence() 方法最终返回字典，每个 key 对应的 value 为 shape (horizon_len, *data_shape)。比如图像是 (640, 480, 3)，对应 (horizon_len, 640, 480, 3)。

## MaskImageDataset

在 replay_buffer 中取出的数，都是由 zarr.Array 转为了 np.Array。

### 归一化图像和插值到 518x518

保存在 zarr 文件中的格式为 (num_steps, H, W, 4)，在 Dataset 中，会转换为重新组织为 (num_steps, 4, H, W)

相机分辨原来是 (640, 480)，处理 rgbm 时，在 _process_mask_image_batch() 方法中，将图像的 channel 轴转置到第二维，得到形状 (T, 3, H, W)。接下来将 rgb 图像除以 255.0，归一化到 [0.0, 1.0]。紧接着，使用 torch.nn.functional.interpolate() 插值，把图像 resize 到 (518,518)。

### 原生数据送给模型需要哪些操作

部署时，相机获取 RGB，SAM 等模块生成 mask。得到 rgbm 后，首先插值，resize 到 (518, 518)。随后经过归一化，便可送给模型。最后再 unnormalize 即可。数据即如此处理。

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

参数 rgbm_data 是 (B,T,4,H,W)，其中，对应 (B,T,4,518,518)。取出 mask_data 为 (B,T,1,518,518)。传给 self.mask_process_net 网络时，由于网络的 patch_embed 使用了 Conv2d，所以对 mask_data reshape 为 ("B T ... -> (B T) ...")。mask 提取特征为 (B*T, num_patches, head_feature_dim)。其中，使用了 14 作为 patch_size，于是 `num_patches = patch_size^2 = (518 // 14)^2 = 37*37 = 1369`，得到 `(B*T, 1369, head_feature_dim)`。

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

output_shape() 方法不止返回特征维度，还返回头部、腕部和状态的特征长度，即每部分分别对应的 [1369, 1369, 1]。用于分辨特征中的位置。在 use_attn_mask 时起作用。此外，硬编码了特征有 2739 个，即 num_patches * 2 + 1，1 对应的是 timestep。

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

## 数据集

首选构造与样例类似的 dataset，可以修改自由度和图像分辨率。但是关键的 key，比如 rgbm, right_cam_img, right_state 和 action 要保持不变。

示教模式中，采集到的数据，只有关节角的状态。可以通过把下一步的关节角作为 action。于是，可以丢弃 episode 中的最后一个时间步。

### UmiDataset

参考 UMI 的配置文件，涉及如下部分：
```yaml
shape_meta: &shape_meta
  obs: 
    camera0_rgb:
      shape: [3, 224, 224]
      horizon: ${task.img_obs_horizon} # int
      latency_steps: 0 # float
      down_sample_steps: ${task.obs_down_sample_steps} # int
      type: rgb
      ignore_by_poliy: False
    robot0_eef_pos:
      shape: [3] #表示末端执行器的位置（x, y, z）。
      horizon: ${task.low_dim_obs_horizon} # int
      latency_steps: ${eval:'(${task.camera_obs_latency} - ${task.robot_obs_latency}) * ${task.dataset_frequeny}'} # float
      down_sample_steps: ${task.obs_down_sample_steps} # float
      type: low_dim
      ignore_by_policy: ${task.ignore_proprioception}
    robot0_eef_rot_axis_angle:
      raw_shape: [3] #原始旋转向量的维度
      shape: [6] #经过旋转表示转换后的维度（6D 表示）。
      horizon: ${task.low_dim_obs_horizon} # int
      latency_steps: ${eval:'(${task.camera_obs_latency} - ${task.robot_obs_latency}) * ${task.dataset_frequeny}'} # float
      down_sample_steps: ${task.obs_down_sample_steps} # float
      type: low_dim
      rotation_rep: rotation_6d #指定旋转的表示方式。
      ignore_by_policy: ${task.ignore_proprioception}
  action: 
    shape: [9] #动作的维度，包括位置（3）、旋转（6)
    horizon: ${task.action_horizon}
    latency_steps: 0 # float
    down_sample_steps: ${task.obs_down_sample_steps} # int
    rotation_rep: rotation_6d
```

输入的图像仅一个 224x224 的 RGB。状态有 robot0_eef_pos。最后，action 为 9 维的。那么，UmiDataset 应该把数据调整为对应这样的。由于是正方形，不用再调整 patch，避免影响效果。

目标掩膜生成和跟踪模块（如 SAM 分割初始掩膜、Cutie 持续跟踪）可能需要与 DINOv2 特征提取器共享相同的输入分辨率，以确保空间对齐和计算一致性。

dinov2_vits14 以 14 为 patch size，恰好将 224 分为 16 个 patch。可以得到编码的特征维度为 384。

设计参考 MaskImageDataset，修改返回的元素即可。

### ObsEncoder

需要调整参数，以适应我们需要的数据集。只要都编码到与 features_dim 相同的内容，便可使用 Transformer 架构。

### TransformerForActionDiffusion

修改参数即可。n_emb 会自动在 policy 中，根据 ObsEncoder 编码的 feature_dim 为多少来设置。隐藏嵌入的维度即 feature_dim。

### SAM

制作数据集的标注思路。可以通过反向播放图像序列。让大模型识别夹住的物体，会更加方便。那么先识别夹住的物体再进一步反向播放图像序列，随后跟踪标边框。

下载 SAM2 的预训练模型后，可以验证：

```py
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image) # numpy.ndarray or PIL image
    masks, _, _ = predictor.predict(...)
```

其中，set_image() 方法接收 numpy.ndarray 或 PIL image 对象。格式为 RGB 或 BGR。

为了选择想要的对象，选择在它上面的一个点即可。一般以 (x,y) 格式指出点，使用 label 为 1 (foreground point) 或 0 (background point)。可以是多个 (x,y) 的点，传二维的 np.ndarray 即可。

predictor.predict() 方法返回三个 masks，还有对应的质量分数和 logits。如果指定 multimaskJ_output=False，则只返回一个。

### 对头部图像标边框

使用 vllm 本地部署 Qwen2.5-VL-3B-Instruct 来标边框。zarr 保存数据位 np.ndarray 形式，cv2 读取也是 np.ndarray，把 np.ndarray 转换为 JPEG 字节流，向 Qwen 询问是重要一步。

```py
import numpy as np
from PIL import Image
from io import BytesIO
import base64

def numpy_to_base64(np_image: np.ndarray) -> str:
    """将numpy数组转换为base64编码的JPEG格式"""
    # 确保数据类型为uint8
    if np_image.dtype != np.uint8:
        np_image = np_image.astype(np.uint8)
    # 转换维度顺序为HWC格式（如果必要），3 通道代表 RGB，4 代表 RGBA
    if np_image.shape[-1] not in [3, 4]:
        np_image = np.transpose(np_image, (1, 2, 0))
    # 转换为PIL图像并编
    pil_img = Image.fromarray(np_image)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
```

有了 str，便可向 Qwen2.5-VL 询问了。

### 标框和制作 mask 思路

Qwen2-VL 标框是为了提供给 sam2 分割，并提供给 cutie 追踪，以得到 mask。

采集数据时，灵巧手从一堆杂物中抓取物体，难以分辨标框的物体，可以使用手动标框。一个 episode 中，

改进和提高的思路：第一次提供作为 context 即可，此操作比较耗时，随后不要再追踪了。

### 多进程制作数据

由于两个项目的环境不同，强行安装 sam2 的环境到 DexGraspVLA，处理冲突是十分繁琐的工作，于是使用多进程协作。

通信内容：io 进程取出图像，为 np.ndarray，向 VLA 请求并标出边框。sam2 根据图像，这个 np.ndarray 和边框，生成相同高和宽的二值掩码。涉及到的通信，是图像、边框和二值掩码。

IPC 方案选择：
- 使用 mutex 和 pickle 序列化
- 共享内存，使用 multiprocessing.shared_memory 的 SharedMemory，使用 named 的共享内存区域即可。同步控制可以用文件锁、信号量来完成。此外，还可以用扩展的 posix_ipc 库，更精准地跨进程同步。实现简单的消费者生产者的模型即可。
- 消息队列，ZeroMQ，性能高，开发复杂度中等。并且，ZeroMQ 也有零拷贝机制。
- REST API (HTTP 通信)，性能低，但通用性强，易开发。

由于小模型需要高频执行，所以对性能有要求。因此，尽可能选择吞吐量的。所以有共享内存由于 ZeroMQ，ZeroMQ 优于 Kafka。但是，有时候单台 PC 可能 GPU 资源不够，需要网络请求，共享内存方案不适合，所以需要消息队列。后期部署，可以使用 ZeroMQ 作为网络通信。

#### ZeroMQ：（REQ-REP 模式） + 零拷贝传输

安装 `pip install pyzmq`，通过 zmq.send/recv 的 copy=False 参数实现零拷贝，适合大数组传输。

实现架构：
- 服务端（进程A）：启动 ZeroMQ 服务端，监听客户端请求，接收数据并处理后返回结果。
- 客户端（进程B）：连接到服务端，发送初始数据，等待响应后继续处理。
- 交替处理：通过 REQ-REP 模式强制请求-响应顺序，确保双方交替操作。

简短示例如下，对于服务端的进程 A，类比实现 sam2 的求掩码：

```py
import zmq
import numpy as np


def process_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP 模式（服务端）
    socket.bind("tcp://*:15555")  # 监听 15555 端口
    print("Server started")
    # 2 bytes for points
    IMAGE_CHUNK_SIZE = 480 * 640 * 4
    BOUNDING_BOX_CHUNK_SIZE = 4 * 2
    chunk_size = IMAGE_CHUNK_SIZE + BOUNDING_BOX_CHUNK_SIZE
    while True:
        # 接收客户端发送的二进制数据（零拷贝）
        msg = socket.recv(copy=False)
        print(f"recerved msg size: {len(msg)}")
        assert len(msg) == chunk_size
        arr = np.frombuffer(
            msg.buffer[:-BOUNDING_BOX_CHUNK_SIZE], dtype=np.uint8
        ).reshape(480, 640, 4)
        bbox_2d = np.frombuffer(
            msg.buffer[-BOUNDING_BOX_CHUNK_SIZE:], dtype=np.uint16
        ).reshape(4,)
        # 处理数据（示例：简单反转数值）
        processed_arr = 255 - arr
        print(f"Received and processed data: {arr.shape} {bbox_2d.shape}\n{bbox_2d}...")

        # 发送处理后的数据（零拷贝）
        socket.send(processed_arr.tobytes(), copy=False)


if __name__ == "__main__":
    process_server()

```

客户端：

```py
import zmq
import time
import numpy as np


def process_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # REQ 模式（客户端）
    socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
    socket.connect("tcp://localhost:15555")  # 连接到服务端

    # 初始化数据（示例：全零数组）
    data = np.zeros((480, 640, 4), dtype=np.uint8)
    bbox_2d = np.array([1, 2, 3, 4], dtype=np.uint16)
    max_try = 5
    tried = 0
    IMAGE_CHUNK_SIZE = 480 * 640 * 4
    BOUNDING_BOX_CHUNK_SIZE = 4 * 2
    chunk_size = IMAGE_CHUNK_SIZE + BOUNDING_BOX_CHUNK_SIZE
    while True:
        # 发送数据（零拷贝）
        socket.send(data.tobytes() + bbox_2d.tobytes(), copy=False)
        time.sleep(1)

        # 接收服务端处理后的数据（零拷贝）
        try:
            msg = socket.recv(copy=False)
            tried = 0
            data = np.frombuffer(msg.buffer, dtype=np.uint8).reshape(480, 640, 4)
            # 客户端进一步处理（示例：加 1）
            # 注意，与整数操作后，可能会从 uint8 变为 int16，会影响 buffer 大小
            data = (data + 1) % 256
            data = data.astype(np.uint8)
        except zmq.Again:
            if tried >= max_try:
                print("max_try count reached, stop trying")
                break
            print("Timeout, retrying...")
            tried += 1
            socket.send(data.tobytes(), copy=False)  # 重发请求


if __name__ == "__main__":
    process_client()
```

因为处理速度问题，客户端一次性可能收到两次请求的数据，即 (2, ...) 的形状，所以需要处理。否则不能 reshape 为 (480, 640, 4)。同理，一次传回客户端，可能数据也会不合适，reshape 也会报错。解决方案如下：
- 一次性读取所有消息队列中的内容，随后处理，并返回。比如，判断数据数量，读取后 reshape(-1, 480, 640, 4)，逐个处理第一维，再返回给客户端。
- 模拟实现一次只读取 `480*640*4`，随后再处理。zeroMQ 没有截断的功能，只能一次性读完，所以可以添加头部元数据的方式，避免“粘包”问题。或者一次只传一个。

### 跟踪对象掩码

SAM2 和 Cutie 都可以跟踪时间序列上的掩码。为了节省显存和资源，使用 SAM2 统一处理。视频可以看做是一系列有序的图像。对于视频，可以用 ffmpeg 分解为多个 jpeg：

```bash
ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'
```

-q:v 代表高质量 jpeg 图像，生成 00000.jpg 起始的图片。

推理视频，需要保存状态。首先，要加载所有 frame 的状态。

#### Cutie

Cutie 环境创建，Python 3.9 比较合适，3.10+ 版本在 cchardet 会出现找不到 longintrepr.h 的问题，新版本对此头文件的使用进行了调整。

mask 值为 0，代表遮盖，其他不同值代表各自对应的物体，比如 1 和 2 可以代表留下的两个物体。对于相同场景，只需要传递一次 mask，随后 processor 会记住。在设计时，进程间通信中，第一次的场景传入图像和 SAM2 提取的 mask，processor 重新清空记忆，使用此上下文信息；随后，不再传 mask，仅传递图像，使用 processor 记录的上下文来追踪 mask。

注意，processor 不像 SAM2 直接接受 [0, 255] 的像素值，而是使用归一化后的值。具体使用 torchvision.transforms.functional 的 to_tensor，将 PIL 图像或在 [0, 255] 值域，形状为 (H, W, C) 且 dtype 为 np.uint8 的 np.ndarray 转换为 torch.FloatTensor，形状为 (C, H, W)，值域为 [0.0, 1.0]。

### 制作数据集后

### 修改 action 和 state 维度，适配我们的机械臂

修改 grasp.yaml 的 action 和 state 下的 shape 为 6

重点修改 obs_encoder.py 下的 forward_state() 和 forward() 方法下的注释提示 action 和 state 最后一维是 13。由于修改了 state 和 action 的维度，首先考虑观测中 state 的维度。图像都插值到了 (518,518)，而 state 的网络 self.state_net 第一个线性层的维度是 13，修改为 6。

### 加载 dinov2 模型

第一次，从网络下载模型，torch.hub 会下载模型和权重文件到 ~/.cache/torch/hub 目录下，可以看到：

```bash
❯ ls -al  ~/.cache/torch/hub
total 16
drwxrwxr-x 4 wj-24 wj-24 4096 Mar  7 18:50 .
drwxrwxr-x 4 wj-24 wj-24 4096 Mar  7 18:44 ..
drwxrwxr-x 2 wj-24 wj-24 4096 Mar 21 18:57 checkpoints
drwxrwxr-x 7 wj-24 wj-24 4096 Mar  7 18:50 facebookresearch_dinov2_main
-rw-rw-r-- 1 wj-24 wj-24    0 Mar  7 18:50 trusted_list
```

checkpoints 则是对应的权重文件：

```bash
❯ ls -al  ~/.cache/torch/hub/checkpoints
total 1759384
drwxrwxr-x 2 wj-24 wj-24       4096 Mar 21 18:57 .
drwxrwxr-x 4 wj-24 wj-24       4096 Mar  7 18:50 ..
-rw-rw-r-- 1 wj-24 wj-24  346378731 Mar  9 17:28 dinov2_vitb14_pretrain.pth
-rw-rw-r-- 1 wj-24 wj-24 1217586395 Mar  9 17:29 dinov2_vitl14_pretrain.pth
-rw-rw-r-- 1 wj-24 wj-24   88283115 Mar  7 18:55 dinov2_vits14_pretrain.pth
-rw-rw-r-- 1 wj-24 wj-24   46827520 Mar 21 18:57 resnet18-5c106cde.pth
-rw-rw-r-- 1 wj-24 wj-24  102502400 Mar 21 18:57 resnet50-19c8e357.pth
```

facebookresearch_dinov2_main 则是仓库的源代码。

接着，可以指定 grasp.yaml 配置如下，避免每次都要下载模型：

```yaml
policy:
  ...
  obs_encoder:
    ...
    model_config:
      head:
        model_type: dinov2_vitb14
        # local weights path, null for online loading
        local_weights_path: /home/wj-24/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth
```

能否考虑 head 和 hand 共用一个模型？

### np.uint64 与任何数计算，都会得到 np.float64，不能作为下标

episode_ends 需要使用类型 np.int64，np.uint64 会解析为 float64

注意大坑，np.ndarray 中，访问数组得到的数字不是原生类型的 int, float 等，而是类似 np.uint64 的类。比如:

```py
arr1 = np.array([123], dtype=np.uint64)
end = arr[0] # type is class of np.uint64
start = 0 # type is native int
float_difference = end - start # 得到的结果是 np.float64，而非 int
```

np.uint64 与 int 类型的四则运算，都不会得到 int，只会得到 np.float64。这会有问题，如果计算的结果用于**索引**或是**切片**，则会导致**异常**。

## 部署

### 加载 Workspace 和 Policy

参考原来的 diffusion policy 项目的部署，文件 eval_real_robot.py 中：

```py
    ...
    # load checkpoint
    # input 是 checkpoint 的路径
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
```

### ObsEncoder

forward() 接收形状为：

```py
    def forward(self, obs_dict, training=True):
        """
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
        """
        ...
```

T 是观察的 n_obs_steps，B 是 DataLoader 组织为的 batchsize。所以测试和部署时，要组织为如此的数据格式。从 (num_steps, H, W, 4) 中取出一张图片是，即 (H, W, 4) 形状时，要组织为 (1, 1, 4, H, W)。

发现 ObsEncoder 的 dinov2 部分加载比较慢。似乎每次都尝试下载。有无方法使用自定义的参数初始化，并加载保存到 checkpoints 模型的参数？

ZeroMQ 的 REQ-REP 必须按照严格的顺序，发送和接收，接受和发送，不能持续的发生

### pymodbus: 通信工具

Modbus 通信支持以下：
- 串行通信：RS-232、RS-422 和 RS-485 等串口通信。RS-485 有长距离和高速度的特点，通常是首选。
- 以太网通信：Modbus TCP 基于 TCP/IP 协议栈的 Modbus 应用协议，采用以太网物理层作为通信媒介，支持点到点或多点到多点通信。

### 历史信息

目前没有训练观察的历史信息到 obs 中，T == 1。
 
### 模型

使用因果掩码时，把视觉部分的掩码都加上？

### pyrealsense2: 获取 RGBD

### 训练

打包当前代码：

```bash
find DexGraspVLA -mindepth 1 -maxdepth 1 | perl -wnle 'm@.*/(Cutie|\.vscode|.*.pkl|__pycache__|data|outputs|local|\.git|wandb|sam2)@ || print;'
```

### 可视化注意力

#### attention map

配置 get_attn_map 之后，在输出检查点的同级目录下，train_sample_attn_maps 目录保存了注意力文件，具体为 pkl 形式。指定 attn_map 路劲后，策略预测时，输出到此位置：

```py
    pred_action = policy.predict_action(batch['obs'], output_path)
```

output_path 是一个 pkl 文件路径，保存了 dict 如下：

```py
    save_dict = {
        "attention_maps": all_timestep_attention_maps,
        "obs_dict": obs_dict_numpy,
    }
```

all_timestep_attention_maps 通义是 dict，保存每个时间步的注意力图。key 是整形数字，values 为 list，对应个 DiT 层的 softmax cross-attention。

具体解析参考文件 attention_map_visualizer.py。

预测噪声时，记录每个 DiT 块的 softmax corss-attention，得到 (DenoiseTimeSteps, num_layers, B, num_heads, T, L)，但是作者仅保留了前两个 sample，即第一个动作 token，得到 (DenoiseTimeSteps, num_layers, :2, num_heads, 0, L)。Decoder 传入的 token 是 q，所以 T 仅取第一个，代表仅参考第一个动作。L 是第一个 token 对于所有条件 tokens 的注意力。

#### 可视化

## Ref and Tag

[[robotics.DexGraspVLA]]

