---
id: 3hpok7a24vlwyzkuzglvtb8
title: ACT
desc: ''
updated: 1740980684009
created: 1740934420521
---


## 从仿真环境收集数据

参考 record_sim_episodes.py。阅读其中代码，弄清格式是如何的。指定 --dataset_dir DIR，如果不存在则创建。

constants.py 定义了数据存放的位置等。一般要有参数模板，指定了 dataset_dir, num_episodes, episode_len, camera_names 等，加载数据和保存数据时有用。

接下来创建仿真环境，主要参考 ee_sim_env.py:make_ee_sim_env() 设置记录轨迹的环境，进一步收集轨迹。环境设置如下：

    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'

sim_env.py:make_sim_env() 配置了最终的运行环境，重新播放轨迹，记录信息。使用绝对关节角角度表示动作，契合 7 维（包含夹爪开合）：

    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'

为什么要分两步？因为难以收集其中某些数据，所以才用先收集轨迹，再播放轨迹来收集剩余数据？

每个时间步保存如下：

    For each timestep:
    observations
    - images
        - each_cam_name     (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'

记录时，先放到字典中，后续再保存到 HDF5 文件。初始化如下：

```py
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []
```

随后，保存观察内容和 action。

## 示例

act-plus-plus 仓库提供了从仿真环境收集的 scripted/human demo，每条 episode 对应一个 HDF5 文件。下载两条 episode 查看 scripted 和 human 的如何：

sim_insertion_scriped/episode_0.hdf5:

```py
def printname(name):
    print(f"f[{name}]:\n\t{f[name]}")
with h5py.File("episode_0_scriped.hdf5", "r") as f:
    f.visit(printname)
```

输出如下，可以看到保存的 keys，而相机只用了 top 的。

    f[action]:
        <HDF5 dataset "action": shape (400, 14), type "<f4">
    f[observations]:
        <HDF5 group "/observations" (3 members)>
    f[observations/images]:
        <HDF5 group "/observations/images" (1 members)>
    f[observations/images/top]:
        <HDF5 dataset "top": shape (400, 480, 640, 3), type "|u1">
    f[observations/qpos]:
        <HDF5 dataset "qpos": shape (400, 14), type "<f4">
    f[observations/qvel]:
        <HDF5 dataset "qvel": shape (400, 14), type "<f4">

再考察 sim_insertion_human/episode_0.hdf5，运行相同代码，查看 keys 也一样。不过是 horizon 长度为 400。

## Ref and Tag

[[robotics.DexVLA_code_阅读代码和复现]]

#Robotics