---
id: gqnz9f63oiug596jem6d3m9
title: ManiSkill-Vitac
desc: ''
updated: 1739372974791
created: 1739260392326
---

使用 Realsense D415 深度相机提供视觉信息。Peg 与 hole 之间距离是增加的。

arguments.py 中，从`solve_argument_confict()`中，优先使用传入命令行的参数，再考虑 yaml 配置文件的参数。

在 evaluation.py 中，offset_list 是测试集数据。用于初始化环境。考察是否能够把测试集数据添加进入训练集。

## Track 2: Tactile-Vision-Fusion Manipulation
In this track, the experimental setup remains unchanged. However, compared to the only tactile task, a Realsense D415 is used to provide visual information, and the distance between the peg and the hole is increased. The gripper needs to determine which hole to insert the peg into based on the shape of the gripped peg. Since the tactile information remains the same during the approach to the hole, **the task primarily relies on visual data for decision-making.**

## Evaluation
```py
    env = PegInsertionSimMarkerFLowEnvV2(**specified_env_args)
    set_random_seed(0)

    offset_list = [
        [2.3, -4.1, -3.2, 8.8, 1],
        [-3.0, 0.5, -6.3, 7.9, 1],
        [3.2, 0.5, 4.2, 3.4, 1],
        ...
    ]
...
                obs, _ = env.reset(offset_list[kk])
```
offset_list 包含测试数据，可以看到，list 中的每条用于重置环境的偏置是 5 个数据的 list，随后根据此环境评估。而 env 是文件 Track_2/envs/peg_insertion_v2.py 下的`class PegInsertionSimEnvV2(gym.Env)`。

```py
class PegInsertionSimEnvV2(gym.Env)
    def reset(self, offset_mm_deg=None, seed=None, options=None) -> Tuple[dict, dict]:
        ...
                offset_mm_deg = self._initialize(offset_mm_deg)
        ...
    def _initialize(self, offset_mm_deg: Union[np.ndarray, None]) -> np.ndarray:
        """
        offset_mm_deg: (x_offset in mm, y_offset in mm, theta_offset in degree, z_offset in mm,  choose_left)
        """
```
可以看到，传入 list 的 5 个数字分别代表如上意义。

## PegInsertionSimMarkerFLowEnvV2
自定义的环境，参考Track_2/envs/peg_insertion_v2.py: `PegInsertionSimMarkerFLowEnvV2`。

```py
class PegInsertionSimMarkerFLowEnvV2(PegInsertionSimEnvV2):
    ...
class PegInsertionSimEnvV2(gym.Env):
    def reset(self, offset_mm_deg=None, seed=None, options=None) -> Tuple[dict, dict]:
        ...
```

## 使用回调函数修改 env，加上测试集数据
关于 TD3 的回调函数的调用时机：回调函数在训练流程的特定阶段被触发，但不会干预环境状态。例如：
* on_step()：每次调用 env.step() 后触发（环境可能处于运行中或已重置）。
* on_rollout_start()：开始收集新的一批数据前触发（此时环境可能已被部分重置）。
* on_rollout_end()：收集完一批数据后触发（环境可能处于任意状态）。

关键点：回调函数仅用于监控或干预训练逻辑（如保存模型、修改超参数），不直接操作环境。

## 查看数据维度
猜测必然在环境的相关配置文件中，自然看向 Track_2/envs/peg_insertion_v2.py，随后搜索 vision，发现：
```py
class PegInsertionSimEnvV2(gym.Env):
    def __init__(
        self,
        ...
        # for vision
        vision_params: dict = None,
        **kwargs,
    ):
```

其参数对应 Track_2/configs/parameters/peg_insertion_v2_points_wj.yaml 文件中的:
```yaml
env:
    ...
    vision_params:
        render_mode: "rast" # "rast" or "rt"
        vision_type : ["rgb", "depth","point_cloud"] # ["rgb", "depth", "point_cloud"], take one, two, or all three.
        # if use point_cloud, the following parameters are needed
        gt_point_cloud: False #  If True is specified, use the ground truth point cloud; otherwise, use the point cloud under render_mode.
        max_points: 128 # sample the points from the point cloud
        # if use ray_trace, the following parameters are needed
        # ray_tracing_denoiser: "optix"
        # ray_tracing_samples_per_pixel: 4
```

查看传入`FeaturesExtractorPointCloud:forward()`的`obs`，以及被拆解的各个数据的维度：
```py
    def forward(self, obs):
        print(f"INFO In FeaturesExtractorPointCloud, obs.keys() is \n{obs.keys()}")
        tactile_left, tactile_right, point_cloud, unsqueezed = self.parse_obs(obs)
        print(f"INFO tactile_left size is {tactile_left.size()}")
        print(f"INFO tactile_right size is {tactile_right.size()}")
        print(f"INFO point_cloud size is {point_cloud.size()}")
        print(f"INFO unsqueezed is {unsqueezed}")

        # the gripper is ignored here.
        print("INFO vision point net vision 1")
        vision_feature_1 = self.point_net_vision1(point_cloud[:, 0])  # object 1
        print("INFO vision point net vision 2")
        vision_feature_2 = self.point_net_vision2(point_cloud[:, 1])  # object 2
        vision_feature = torch.cat([vision_feature_1, vision_feature_2], dim=-1)

        print("INFO point net tac with tac left")
        tactile_left_feature = self.point_net_tac(tactile_left)
        print("INFO point net tac with tac right")
        tactile_right_feature = self.point_net_tac(tactile_right)
        print("INFO got tac right feature")
        tactile_feature = torch.cat([tactile_left_feature, tactile_right_feature], dim=-1)

        features = torch.cat(
            [
                self.layernorm_vision(vision_feature),
                self.layernorm_tac(tactile_feature),
            ],
            dim=-1,
        )
        if unsqueezed:
            features = features.squeeze(0)
        return features
```
```sh
INFO In FeaturesExtractorPointCloud, obs.keys() is
dict_keys(['depth_picture', 'gt_direction', 'gt_offset', 'marker_flow', 'object_point_cloud', 'point_cloud', 'relative_motion', 'rgb_picture'])
INFO key: depth_picture, v.size() is torch.Size([6, 480, 640])
INFO key: gt_direction, v.size() is torch.Size([6, 1])
INFO key: gt_offset, v.size() is torch.Size([6, 4])
INFO key: marker_flow, v.size() is torch.Size([6, 2, 2, 128, 2])
INFO key: object_point_cloud, v.size() is torch.Size([6, 2, 128, 3])
INFO key: point_cloud, v.size() is torch.Size([6, 480, 640, 3])
INFO key: relative_motion, v.size() is torch.Size([6, 4])
INFO key: rgb_picture, v.size() is torch.Size([6, 3, 480, 640])
INFO tactile_left size is torch.Size([6, 128, 4])
INFO tactile_right size is torch.Size([6, 128, 4])
INFO point_cloud size is torch.Size([6, 2, 128, 3])
INFO unsqueezed is False
INFO vision point net vision 1
INFO forward marker_pos shape is torch.Size([6, 128, 3])
INFO forward marker_pos shape is torch.Size([6, 128, 3])
INFO vision point net vision 2
INFO forward marker_pos shape is torch.Size([6, 128, 3])
INFO forward marker_pos shape is torch.Size([6, 128, 3])
INFO point net tac with tac left
INFO forward marker_pos shape is torch.Size([6, 128, 4])
INFO forward marker_pos shape is torch.Size([6, 128, 4])
INFO point net tac with tac right
INFO forward marker_pos shape is torch.Size([6, 128, 4])
INFO forward marker_pos shape is torch.Size([6, 128, 4])
INFO got tac right feature
```
可以看到，传入的包含深度图像，还有触觉等信息。触觉是是 4 个维度的，点云最后维度是 3 的。通过 Debug 模式，考察上述各个 feature 维度：
* vision_feature_1: (6, 128)
* vision_feature_2: (6, 128)
* vision_feature_2: (6, 256)
* tactile_left_feature: (6, 64)
* tactile_right_feature: (6, 64)
* tactile_feature: (6, 128)
* features: (6, 384)，相当于 128+256

6 猜测是指代六个线程。vision 和 tac 的最后一个维度分别是 128 和 64，分别对应配置文件中 vision_kwargs:out_dim 和 tac_kwargs:out_dim。可以看到，每个 torch.cat 拼接了最后维度，把左右的视觉触觉拼接起来。

## PointNetActor
```py
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(False):
            marker_pos = self.extract_features(obs, self.features_extractor)

        if marker_pos.ndim == 3:
            marker_pos = torch.unsqueeze(marker_pos, dim=0)

        batch_num = marker_pos.shape[0]
        l_marker_pos = marker_pos[:, 0, ...]
        r_marker_pos = marker_pos[:, 1, ...]
        marker_pos_input = torch.cat([l_marker_pos, r_marker_pos], dim=0)
        point_flow_fea = self.point_net_feature_extractor(marker_pos_input)
        l_point_flow_fea = point_flow_fea[:batch_num, ...]
        r_point_flow_fea = point_flow_fea[batch_num:, ...]
        point_flow_fea = torch.cat([l_point_flow_fea, r_point_flow_fea], dim=-1)
        pred = self.mlp_policy(point_flow_fea)

        return pred
```
可以看到，要求 marker_pos 至少为四维的 Tensor，或者至少三维。

## Critic
```py
# Track_2/solutions/policies.py
class TD3PolicyForPegInsertionV2(TD3Policy):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(TD3PolicyForPegInsertionV2, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PointNetActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        print(f"actor kw is {actor_kwargs}")
        return PointNetActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None):
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, FeatureExtractorState(self.observation_space)
        )
        return CustomCritic(**critic_kwargs).to(self.device)

# Track_2/solutions/feature_extractors.py
class FeatureExtractorState(BaseFeaturesExtractor):
    ...
```
从`TD3PolicyForPegInsertionV2`的`make_critic`中，我们可以看到，传给`self._update_features_extractor`的包含了自定义的`FeatureExtractorState`，猜测这些部分应当包含了触觉和视觉的融合工作。

## Actor
在 Policy 文件中重写 make_actor，查看传入的参数，即 TD3Policy 的 self.actor_kwargs：
```py
class TD3PolicyForPegInsertionV2(TD3Policy):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        print(f"INFO TD3PolicyForPegInsertionV2 init params:\nargs: {args}\nkwargs{kwargs}")
        super(TD3PolicyForPegInsertionV2, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PointNetActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, FeatureExtractorForPointFlowEnv(self.observation_space)
        )
        print(f"actor kw is {actor_kwargs}")
        return PointNetActor(**actor_kwargs).to(self.device)

# /home/lxt-24/miniforge3/envs/mani_vitac/lib/python3.10/site-packages/stable_baselines3/td3/policies.py
class TD3Policy(BasePolicy):
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = [256, 256]
            else:
                net_arch = [400, 300]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        ...
```
可以看到，`make_actor`用到的参数有传给`TD3PolicyForPegInsertionV2`的`__init__`中的：observation_space, action_space, net_arch, activation_fn, normalize_images。这些内容参考 yaml 配置文件的 policy 部分的 policy_kwargs。考察发现：
```yaml
# Track_2/configs/parameters/peg_insertion_v2_points_wj.yaml
policy:
  policy_name: TD3PolicyForPegInsertionV2
  buffer_size: 10000
  train_freq: 2
  gradient_steps: -1
  learning_starts: 1000
  target_policy_noise: 0.5
  target_noise_clip: 1
  action_noise: 0.6
  batch_size: 128
  learning_rate: 0.0007
  policy_delay: 2

  policy_kwargs:
    features_extractor_class: PointCloud
    share_features_extractor: False
    net_arch:
      pi: [512, 512]
      qf: [512, 512]
      # pi: [256, 256]
      # qf: [256, 256]
    
    features_extractor_kwargs:
      # visual
      vision_kwargs:
        scale: 100
        out_dim: 128
        batchnorm: False
      # tactile
      tac_kwargs:
        out_dim: 64
        batchnorm: False
```

传入 __init__ 的参数：
```
INFO TD3PolicyForPegInsertionV2 init params:
args: (
    Dict(
        'depth_picture': Box(-3.4028235e+38, 3.4028235e+38, (480, 640), float32), 'gt_direction': Box(-3.4028235e+38, 3.4028235e+38, (1,), float32), 'gt_offset': Box(-3.4028235e+38, 3.4028235e+38, (4,), float32), 'marker_flow': Box(-3.4028235e+38, 3.4028235e+38, (2, 2, 128, 2), float32), 'object_point_cloud': Box(-3.4028235e+38, 3.4028235e+38, (2, 128, 3), float32), 'point_cloud': Box(-3.4028235e+38, 3.4028235e+38, (480, 640, 3), float32), 'relative_motion': Box(-3.4028235e+38, 3.4028235e+38, (4,), float32), 'rgb_picture': Box(0, 255, (3, 480, 640), uint8)
    ), 
    Box(-1.0, 1.0, (4,), float32), <function constant_fn.<locals>.func at 0x7e093cb58d30>)
kwargs{
    'share_features_extractor': False, 
    'net_arch': {'pi': [256, 256], 
    'qf': [256, 256]}, 
    'features_extractor_kwargs': {
        'vision_kwargs': {'scale': 100, 'out_dim': 128, 'batchnorm': False}, 
        'tac_kwargs': {'out_dim': 64, 'batchnorm': False}
    }, 
    'features_extractor_class': <class 'solutions.feature_extractors.FeaturesExtractorPointCloud'>
}
```

可以看到有 vision_kwargs 和 tac_kwargs，猜测其必然与视觉和触觉相关。打印的 kw_args 有：
```
actor kw is {
    'observation_space':
        Dict(
            'depth_picture': Box(-3.4028235e+38, 3.4028235e+38, (480, 640), float32), 
            'gt_direction': Box(-3.4028235e+38, 3.4028235e+38, (1,), float32), 
            'gt_offset': Box(-3.4028235e+3[43/1979] 235e+38, (4,), float32), 
            'marker_flow': Box(-3.4028235e+38, 3.4028235e+38, (2, 2, 128, 2), float32), 
            'object_point_cloud': Box(-3.4028235e+38, 3.4028235e+38, (2, 128, 3), float32), 
            'point_cloud': Box(-3.4028235e+38, 3.4 028235e+38, (480, 640, 3), float32), 
            'relative_motion': Box(-3.4028235e+38, 3.4028235e+38, (4,), float32), 
            'rgb_picture': Box(0, 255, (3, 480, 640), uint8)
        ), 
        'action_space': Box(-1.0, 1.0, (4,), float32), 
        'net_arch': [512, 512], 
        'activation_fn': <class 'torch.nn.modules.activation.ReLU'>, 
        'normalize_images': True, 
        'features_extractor': FeaturesExtractorPointCloud(
            (point_net_vision1): PointNetFeatureExtractor(
                (pointnet_local_fea): Sequential(
                    (0): Conv1d(3, 64, kernel_size=(1,), stride=(1,))
                    (1): Identity()
                    (2): ReLU()
                    (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    (4): Identity()
                    (5): ReLU()
                    )
                (pointnet_global_fea): PointNetFeaNew(
                    (conv0): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                    (bn0): Identity()
                    (conv1): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
                    (bn1): Identity()
                    (conv2): Conv1d(128, 512, kernel_size=(1,), stride=(1,))
                    (bn2): Identity()
                )
                (mlp_output): Sequential(
                    (0): Linear(in_features=512, out_features=256, bias=True)
                    (1): ReLU()
                    (2): Linear(in_features=256, out_features=256, bias=True)
                    (3): ReLU()
                    (4): Linear(in_features=256, out_features=128, bias=True)
                )
                )
                (point_net_vision2): PointNetFeatureExtractor(
                (pointnet_local_fea): Sequential(
                (0): Conv1d(3, 64, kernel_size=(1,), stride=(1,))
                (1): Identity()
                (2): ReLU()
                (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                (4): Identity()
                (5): ReLU()
                )
                (pointnet_global_fea): PointNetFeaNew(
                (conv0): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                (bn0): Identity()
                (conv1): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
                (bn1): Identity()
                (conv2): Conv1d(128, 512, kernel_size=(1,), stride=(1,))
                (bn2): Identity()
                )
                (mlp_output): Sequential(
                (0): Linear(in_features=512, out_features=256, bias=True)
                (1): ReLU()
                (2): Linear(in_features=256, out_features=256, bias=True)
                (3): ReLU()
                (4): Linear(in_features=256, out_features=128, bias=True)
                )
                )
                (point_net_tac): PointNetFeatureExtractor(
                (pointnet_local_fea): Sequential(
                (0): Conv1d(4, 64, kernel_size=(1,), stride=(1,))
                (1): Identity()
                (2): ReLU()
                (3): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                (4): Identity()
                (5): ReLU()
                )
                (pointnet_global_fea): PointNetFeaNew(
                (conv0): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
                (bn0): Identity()
                (conv1): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
                (bn1): Identity()
                (conv2): Conv1d(128, 512, kernel_size=(1,), stride=(1,))
                (bn2): Identity()
                )
                (mlp_output): Sequential(
                (0): Linear(in_features=512, out_features=256, bias=True)
                (1): ReLU()
                (2): Linear(in_features=256, out_features=256, bias=True)
                (3): ReLU()
                (4): Linear(in_features=256, out_features=64, bias=True)
                )
                )
                (layernorm_vision): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                (layernorm_tac): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    ),
    'features_dim': 384
}

```


查看文件`Track_2/solutions/feature_extractors.py`，根据注释，可以猜测`FeatureExtractorForPointFlowEnv`是`PointNetActor`的特征提取器：
```py
class FeatureExtractorForPointFlowEnv(BaseFeaturesExtractor):
    """
    feature extractor for point flow env. the input for actor network is the point flow.
    so this 'feature extractor' actually only extracts point flow from the original observation dictionary.
    the actor network contains a pointnet module to extract latent features from point flow.
    """
    ...
```

但是考察`FeaturesExtractorPointCloud`，可以看到包含了视觉和触觉信息，这里应当才是包含了视觉和触觉的特征提取器。
```py
class FeaturesExtractorPointCloud(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        vision_kwargs: dict = None,
        tac_kwargs: dict = None,
    ):
        super().__init__(observation_space, features_dim=1)

        # PointCloud
        vision_dim = vision_kwargs.get("dim", 3)
        vision_out_dim = vision_kwargs.get("out_dim", 64)
        self.vision_scale = vision_kwargs.get("scale", 1.0)
        vision_batchnorm = vision_kwargs.get("batchnorm", False)
        self.point_net_vision1 = PointNetFeatureExtractor(
            dim=vision_dim, out_dim=vision_out_dim, batchnorm=vision_batchnorm
        )
        self.point_net_vision2 = PointNetFeatureExtractor(
            dim=vision_dim, out_dim=vision_out_dim, batchnorm=vision_batchnorm
        )
        self.vision_feature_dim = vision_out_dim * 2
        # Tactile
        tac_dim = tac_kwargs.get("dim", 4)
        tac_out_dim = tac_kwargs.get("out_dim", 32)
        tac_batchnorm = tac_kwargs.get("batchnorm", False)
        self.point_net_tac = PointNetFeatureExtractor(
            dim=tac_dim, out_dim=tac_out_dim, batchnorm=tac_batchnorm
        )
        self.tac_feature_dim = tac_out_dim * 2

        self.layernorm_vision = nn.LayerNorm(self.vision_feature_dim)
        self.layernorm_tac = nn.LayerNorm(self.tac_feature_dim)

        self._features_dim = self.vision_feature_dim + self.tac_feature_dim

    def parse_obs(self, obs: dict):
        obs = obs.copy()
        marker_flow = obs["marker_flow"]
        point_cloud = torch.Tensor(obs["object_point_cloud"])

        unsqueezed = False
        if marker_flow.ndim == 4:
            assert point_cloud.ndim == 3
            marker_flow = torch.unsqueeze(marker_flow, 0)
            point_cloud = point_cloud.unsqueeze(0)
            unsqueezed = True

        fea = torch.cat([marker_flow[:, :, 0, ...], marker_flow[:, :, 1, ...]], dim=-1)
        tactile_left, tactile_right = (
            fea[:, 0],
            fea[:, 1],
        )  # (batch_size, marker_num, 4[u0,v0,u1,v1])

        point_cloud = point_cloud * self.vision_scale

        return tactile_left, tactile_right, point_cloud, unsqueezed

    def forward(self, obs):
        tactile_left, tactile_right, point_cloud, unsqueezed = self.parse_obs(obs)

        # the gripper is ignored here.
        vision_feature_1 = self.point_net_vision1(point_cloud[:, 0])  # object 1
        vision_feature_2 = self.point_net_vision2(point_cloud[:, 1])  # object 2
        vision_feature = torch.cat([vision_feature_1, vision_feature_2], dim=-1)

        tactile_left_feature = self.point_net_tac(tactile_left)
        tactile_right_feature = self.point_net_tac(tactile_right)
        tactile_feature = torch.cat(
            [tactile_left_feature, tactile_right_feature], dim=-1
        )

        features = torch.cat(
            [
                self.layernorm_vision(vision_feature),
                self.layernorm_tac(tactile_feature),
            ],
            dim=-1,
        )
        if unsqueezed:
            features = features.squeeze(0)
        return features
```


## Gymnasium
强化学习环境中，智能体（Agent）通过与环境交互学习策略。每个环境包含：
* 状态（State/Observation）：环境的当前信息（如机器人关节角度）。
* 动作（Action）：智能体可以执行的操作（如向左移动）。
* 奖励（Reward）：执行动作后获得的反馈（如得分增减）。
* 终止条件（Terminated/Truncated）：判断当前回合是否结束（如任务完成或超时）。

Gymnasium的作用
* 提供标准化的环境接口（如env.step()、env.reset()）。
* 包含大量预定义环境（如经典控制、Atari游戏、机器人仿真）。
* 支持自定义环境开发。

简单例子：
```py
import gymnasium as gym

# 创建环境（例如经典的CartPole平衡杆）
env = gym.make("CartPole-v1", render_mode="human")  # "human"表示可视化

# 初始化环境，返回初始状态
obs, info = env.reset()

# 随机交互10步
for _ in range(10):
    action = env.action_space.sample()  # 随机选择一个动作
    # 执行动作，返回：
    # obs：新状态
    # reward：即时奖励
    # terminated：是否达成终止条件（如任务成功/失败）
    # truncated：是否因外部限制终止（如超时）
    # info：调试信息（如日志）
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"状态: {obs}, 奖励: {reward}, 是否结束: {terminated or truncated}")
    if terminated or truncated:
        # 重置环境，返回初始状态和附加信息（如info字典）。
        obs, info = env.reset()

env.close()  # 关闭环境
```
关键属性
* `env.action_space` 动作空间的形状和类型（如Discrete(2)表示二选一动作）。
* `env.observation_space` 状态空间的形状和类型（如Box(4,)表示4维连续向量）。

### 自定义环境
Gymnasium 允许自定义环境，需继承`gymnasium.Env`并实现以下方法：
```py
class MyEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)  # 动作空间
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))  # 状态空间

    def reset(self, seed=None):
        # 重置环境，返回初始状态和info
        return obs, info

    def step(self, action):
        # 执行动作，返回obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info

    def render(self):
        # 可选：实现可视化
        pass

    def close(self):
        # 可选：释放资源
        pass
```

### 与其他库结合使用
```py
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)  # 训练10k步

# 可视化结果版本
obs, info = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
```

### 官方文档
https://gymnasium.farama.org/introduction/basic_usage/

Importantly, `Env.action_space` and `Env.observation_space` are instances of `Space`, a high-level python class that provides the key functions: `Space.contains()` and `Space.sample()`.