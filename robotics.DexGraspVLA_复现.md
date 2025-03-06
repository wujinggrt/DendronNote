---
id: us3phg4jcf3ej4lpymsyu6q
title: DexGraspVLA_复现
desc: ''
updated: 1741240416494
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

buffer_<start|end>_idx 指出了 episode 在 buffer 中的区间。sample_<start|end>_idx 指出了具体每次训练时，每个时间步 t 对应的 horizon 区间。有可能 start_idx < 0，这在 n_obs_step > 1 时会出现，使用复制和填充第一个观察来处理。末尾部分同理。

sample_sequence() 方法最终返回字典，每个 key 对应的 value 为 shape (horizon_len, *data_shape)。比如图像是 (640, 480, 3)，对应 (horizon_len, 640, 480, 3)。

## ObsEncoder




## Ref and Tag

[[robotics.DexGraspVLA]]
