---
id: 4gb9ottxmfh95i6654zy8hq
title: DexVLA复现
desc: ''
updated: 1740206306943
created: 1740053039805
---

## 数据准备
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

## VLM
使用 [Qwen2-2B-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) 作为主干网络。也许可以尝试 [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)。

模型结构，在 VLM 末尾增加一个 policy head；而 Helix 直接输出 token，当作 policy 模型的 latent vector。

### Qwen2-VL
项目文件 qwen2_vla/models/modeling_qwen2_vla.py 是从 huggingface 的 transformers 库中 transformers/models/qwen2_vl/modeling_qwen2_vl.py 复制而来，并根据需求做出修改。

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


## Tag and Ref
[[robotics.DexVLA]]
[[robotics.Helix：用于通才人形机器人控制的_VLM]]
[[robotics.HiRT_使用分层机器人Transformer提示机器人控制]]
[[insights.Robotics]]

#复现
