---
id: dsrwzp0ob6wn0rz185kgq7w
title: Legged_gym
desc: ''
updated: 1742303551225
created: 1742190380730
---

## Legged Gym 和二次开发

目前，作者已经移植到更新版本的 [IsaacLab](https://github.com/isaac-sim/IsaacLab)。

需要关注的重要文件是：
- legged_gym/legged_gym/envs/base/legged_robot.py，定义了 class LeggedRobot。
- legged_gym/legged_gym/envs/base/legged_robot_config.py

它们描述了机器人的各种细节。

### LeggedRobot: reward

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

reset_idx() 在初始化时，还有 check_termination() 之后会调用。还会调用 `_reset_dofs()`, `_reset_root_states()`, `_resample_commands()` 等方法重置关节、状态和重新给与命令。关于命令，在 LeggedRobotCfg:commands:ranges 指定了范围，从中采样一个数使用。可以看到，self.commands[env_ids, 0] 决定了线速度 x 分量，其余参考代码。在最后面，如果线速度随机采样到小于 0.2 的数，则直接设置为 0 (即与 False 相乘)，避免机器人不知道怎么走。正常的行走不会以那么小的速度行走，所以直接不考虑它。

#### progress：推动训练的进行。

step() 方法，根据计算出的 action 来模拟。需要关注 cfg.control.decimation，对应 LeggedRobotCfg.control.decimation，与仿真环境的控制频率相关。仿真可以以 1000Hz 执行，但实际的机器人不可能。计算量大，还有负重、散热等问题需要考虑。所以，不能简单把仿真的频率应用到机器人上。于是，decimation 标识仿真和实际 action 执行 step 的差值。时间步 self.episode_length_buf 自增一个单位，每个机器人执行的 step 数量，查看 self.episode_length_buf 可知。

post_physics_step() 方法在 step() 中调用。计算完 torch 等内容之后，此方法计算观测，奖励，检查 termination 等。

`_post_physics_step_callback()` 根据注释了解，计算角速度、线速度和高度，随机推机器人等在此处设置。

`_compute_torques()` 计算力矩。根据 control_type 计算力控。P 代表位控，比如希望达到关节位置；V 代表速控，T 力控，都是对应关节速度和力矩。

compute_observations() 在观察值方面，有 self.obs_buf 存储观测的值。参考 compute_observations() 的内容。这些值通常用于计算奖励。比如青龙，记录了 dof_pos, dof_vel 等等内容。

#### reward

在最后，设置了一系列的奖励函数。

`_prepare_reward_fuinction()` 在初始化时，导入奖励函数。参考 LeggedRobotCfg:rewards.scales，各个字段的名字及对应值，会解析为字典，保存到 self.reward_scales。根据字典的 key/value 来解析奖励函数名和需要保存的奖励函数。奖励函数名字为 `_reward_<key>`，动态地加载到 self.reward_functions 列表，添加 `_reward_` 的函数名字会放入 self.reward_names 列表。如果注释或者删除 class scales 下的字段，比如注释 torques = -0.00001，奖励函数 `_reward_torques` 则不会被解析和纳入考虑；或设置为 0，也不会考虑。

compute_reward() 计算奖励。class scales 中，字段的值代表奖励的权重，奖励函数返回的结果先与奖励相乘，随后才存储。存储包含整体所有的 reward，还有各个奖励函数对应加权后的奖励。在我们写了很多奖励函数后，如果觉得某些函数不好用，可以在 scales 中注释对应奖励函数的字段即可，不用删除奖励函数。

以 `_reward_lin_vel_z()` 为例，查看如何写奖励函数。此函数惩罚 z 方向线速度，返回 z 方向线速度平方。根据 scales:lin_vel_z 的值 -2.0，可以知道线速度越大，负值越小，惩罚越多。

`_reward_ang_vel_xy()` 的思路类似，惩罚 xy 轴的角速度。

`_reward_tracking_lin_vel()` 时任务型的奖励，根据 self.commands[:, :2]，即命令给与它的速度和实际上的速度，求差值。差值越小，奖励越大，以完成命令。

```py
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
```

`_reward_action_rate()` 希望力矩输出平滑，变动不要太快。变化快，与上一个动作相差越大，惩罚越多。变化太快，容易造成点击过热或损伤。我们希望动作顺滑。

```py
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
```

`_reward_dof_pos_limits()` 希望关节不要过于接近极限位置，接近极限位置时容易扭坏，损坏电机，容易失去平衡。

```py
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (
                torch.abs(self.torques)
                - self.torque_limits * self.cfg.rewards.soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )
```

```py
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
            torch.norm(self.commands[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
```

奖励可以进一步细分:
- 任务类，尽可能完成 self.commands 要求的内容，例如 `_reward_tracking_lin_vel()`。
- smooth 类的任务，期望力矩输出比较平滑，比如 `_reward_action_rate()`
- safety 类任务。比如 `_reward_dof_pos_limits()`
- beauty 类任务，比如 `_reward_feet_air_time()` 希望脚抬高一点，走得美观。

### LeggedRobotCfg

#### env: 机器人的设置

- num_privileged_obs 暂时没有用到。
- num_actions 关节数，自由度
- env_spacing 初始化时，生成机器人时，确保原点之间的间隔
- episode_length 机器人存活的时间。如果没有设置，或者是比较大，机器人可能采取一动不动的状态。设置时间后，会在此时间内重置，确保机器人训练时不摆烂。

#### terrian

- mesh_type 决定地形

#### commands

- num_commands 在 ranges 中命令的数量。需要训练拟合的情况。与 LeggedRobot:_resample_commands() 方法一起使用。
- resampling_time 决定重新给命令的时间。比如训练机器人一会儿走快一会儿慢，一会儿左右。应该设置得不断也不长。10 比较合适。

#### init_state

重置后，希望机器人保持的状态。一般设置为 0，代表不动。

- pos 的 z 一般与机器人高度相当，太高则摔下来，低则卡脚。

#### control

- control_type 决定控制方式，位置、速度和力矩。

#### asset

与 URDF 挂钩。参考青龙。

- keypoints 躯干位置
- foot_name 脚的位置等
- self_collisions 自碰撞。比如甩手碰到身体时，希望穿越还是碰撞。

#### rewards

一般 sacles 之间差距不要太大

- only_positive_rewards 奖励如果设置较好，不需要打开。否则需要启用。

#### noise

噪声，确保可靠性。

#### sim

- dt 仿真的频率。0.005 代表 0.005 秒一步仿真。此值乘以 control.decimation 得到一个 step 的时间。比如默认为 4，那么 0.02s 完成一次 step。如果电脑一般，可以调大 dt，减少仿真步数。

### LeggedRobotCfgPPO

网络的内容。

### 二次开发着手点

一般选择调整奖励，或者是调整课程，对应在 curriculum 部分。比如让机器人跑起来，可以修改奖励。

### 如何使用自己的机器人，使用自己的 URDF

以青龙为例。需要修改 LeggedRobotCfg.asset.file，指出机器人描述文件位置。一般在 resource 目录，例如 resource/robots/cassie/urdf。随后在 asset 下的 keypoints, end_effectors, foot_name 指定机器人的位置。这些东西都是 link，随后是 termination。需要和 urdf 对应起来。其他 ctrol 下的 stiffness 和 damping 等，也要和 urdf 文件描述的保持一致。

比如，青龙有 30 多个 DoF，比原 legged_gym 中的机器人多，那么需要分别设置。对于固定部分，比如头部没有自由度，只能固定。其他自由的，可控制的，在 class init_state 中，指定 default_joint_angles, dof_pos_range 等等内容。另外，可以控制的 DoF 部分，再 class control 部分指出，具体参考 stiffness, damping 等。

### 注册新环境

在 legged_gym/legged_gym/envs/__init__.py 中，进行注册自定义的，继承的 Robot, Cfg, PPO。

### 各个变量对应什么内容

我想要设置某些奖励，需要参考的观察值的具体含义，在哪儿看？

## TODO: GRU

LSTM 或 GRU 是否可以 batch 训练？即并行训练。如何处理历史信息？一次 rollout 代表一组训练完整的数据吗？

可以参考 LS-Imagination。

## 站立的奖励

如果与时序有关，比如先附身，后挺直。那么时序上，如何设置奖励？与 episode_length_buf 中的值，要相关吗？用来判断，是否在几个时间步内，要有这个动作？

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