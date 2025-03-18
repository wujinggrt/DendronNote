---
id: dsrwzp0ob6wn0rz185kgq7w
title: Legged_gym
desc: ''
updated: 1742262091303
created: 1742190380730
---

## Legged Gym 和二次开发

目前，作者已经移植到更新版本的 [IsaacLab](https://github.com/isaac-sim/IsaacLab)。

需要关注的重要文件是：
- legged_gym/legged_gym/envs/base/legged_robot.py，定义了 class LeggedRobot。
- legged_gym/legged_gym/envs/base/legged_robot_config.py

它们描述了机器人的各种细节。

### LeggedRobot

LeggedRobot 中，如果需要添加新的 field，可以在 `_init_buffers()` 方法添加。比如，青龙的 env 目录下，Azure_Long_env.py 中，集成了 LeggedRobot，添加了 DOF pose 的历史值 self.dof_pos_hist 等。

`_parse_cfg()` 解析 legged_robot_config.py 中的配置。

create_sim() 创建仿真环境。可以创建三种常见的 Isaac Gym 中的地形。在 LeggedRobotCfg:terrian:mesh_type 中选择。训练楼梯、上下坡、平地的情况，选择这些配置即可。

set_camera(), `_draw_debug_vis()` 等是 debug 的部分。

`_init_height_points()` 方法，与 LeggedRobotCfg:terrian:measure_points_x 和 measure_points_y 相关，作为存点图，在机器人附近创建 1.6m x 1m 的空间，记录附近高度。

创建环境部分，涉及创建刚体属性，DoF 等：
- `_process_rigid_shape_props()`
- `_process_dof_props()`
- `_process_rigid_body_props()`
- `_create_envs()` 创建环境，加载 URDF 文件等。
- `_create_env_origins()` 在 Isaac Gym 中，一个机器人，或是 Agent，都称之为 env。比如 LeggedRobotCfg:env:num_envs 代表机器人数量。此方法生成机器人，设置其相对原点。把绝对坐标系舍之道 self.env_origins 中。机器人会生成原点，有些时候会需要根据机器人的坐标原点，计算某些空间相关的内容，此时会需要相对原点的信息。
- `_get_heights()` 在不同地形下计算高度，面对地形凹陷和凸起时有用。

#### explore

`_push_robots()` 训练时，从某个方向给推力。以训练时，保证受到推力不摔倒。

`_update_terrain_curriculum()`，`update_command_curriculum()` 是课程，一开始不能让机器人走较难的地形。比如青龙的工作，设置 LeggedRobotCfg:commands:ranges:lin_vel_x 为 [-1.0, 1.0]，设置线性速度在 -1 到 1。如果用了这两个方法，速度会逐渐升高，地形也是。让机器人逐渐学会这些内容。比如，从 5cm 楼梯逐渐走到 30cm 楼梯。

`_get_noise_scale_vec()`，在 LeggedRobotCfg:nosie 下指定噪声，增强健壮。

check_termination() 遇到危险情况时，重置。比如，根据 self.contact_force 判断身体是否受到了力，即是否摔倒，从而选择重置与否。青龙则添加了更多的判断，比如线速度大于 10，角速度大于 5，高度太低小于 0.3 重置，更多参考青龙的源码。这个方法比较重要，希望训练时不要出现的情况，可以写到此方法。比如不要摔倒等。

reset_idx() 在初始化时，还有 check_termination() 之后会调用。还会调用 `_reset_dofs()`, `_reset_root_states()`, `_resample_commands()` 等方法重置关节、状态和重新给与命令。

## TODO: GRU

LSTM 或 GRU 是否可以 batch 训练？即并行训练。如何处理历史信息？一次 rollout 代表一组训练完整的数据吗？

可以参考 LS-Imagination。

## 论文内容

Actively seeking diverse views breaks information cocoons by exposing biases. Varied sources—academic, cultural, adversarial—counter algorithmic echo chambers. Studies show diverse news consumption boosts cognitive flexibility by 37%, improving problem-solving. Confronting conflicting ideas, per Mill’s “marketplace of ideas,” builds resilience. This strengthens democracy through critical engagement with pluralism.

## 其他资源

跟随动捕的动作来学习技能。
- https://arxiv.org/pdf/2302.09450 跳跃
- https://arxiv.org/pdf/2401.16889 跳跃+行走+奔跑


## Ref and Tag

[Github: Legged Gym](https://github.com/leggedrobotics/legged_gym)

【强化学习框架-Legged Gym 训练代码详解】 https://www.bilibili.com/video/BV1sLx6eLEyt/?share_source=copy_web&vd_source=fe1db0b3f565b5940b244b9c7f7b4fa1

青龙
https://atomgit.com/openloong/gymloong


【自学记录：legged_gym】 https://www.bilibili.com/video/BV1fJA2e4E31/?share_source=copy_web&vd_source=fe1db0b3f565b5940b244b9c7f7b4fa1