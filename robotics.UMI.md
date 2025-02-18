---
id: u0ydqn2ohvl9vw86t1e8mt8
title: UMI
desc: ''
updated: 1739854679697
created: 1739810960614
---

提供了一个框架，使用手持夹爪方便地收集数据，训练能够推广到不同的机器人平台。

## Method
### A Demonstration 
* HD1 使用一个 3D 打印的夹爪，手腕部分搭载唯一传感器 GoPro 相机。双臂操作使用另一个相同的 Gripper。
* HD2 155 度的鱼眼相机提供足够广的的视觉信息。
* HD3 旁侧镜子提供隐式立体信息。
* HD4 IMU-感知追踪路径，使用 GoPro 能够与 mp4 视频文件一同记录 IMU 数据 (加速度计和陀螺仪)。作者的 Inertial-monocular SLAM system 基于 ORB-SLAM3，记录了短期的轨迹，在视觉轨迹失败 (比如动作模糊) 后能够起作用。
* HD5 连续地控制夹爪能够明显地扩展完成任务能力。
* HD6 数据收集后，基于运动学来筛选有效数据。

### B Policy Interface Design
收集演示数据后，使用同步后的数据 (RGB images, 6 DoF end-effector pose, and gripper width)，训练基于视觉运动的策略，生成一系列动作 (end-effector pose and gripper width)。作者使用了 Diffusion Policy。

UMI 的目的是保证 interface 在无需知晓底层机器人硬件平台的情况下，任然能够适用，即只用手持夹爪来收集数据并训练。挑战：
* 具体硬件延迟。系统存在各种延迟，比如相机数据流，机器人控制和工业夹爪的控制流。不过 GoPro 采集的各种数据直接没有延迟，包括视频、IMU 数据和基于视觉的夹爪宽度评估。
* 具体具身的自身感知。通常，关节角和 EE pose 等自身感知 (proprioception)，仅仅对具体机械臂有明确定义。因此，UMI 需要解决泛化不同机器人具身的挑战。

![fig5](assets/images/robotics.UMI/fig5.png)

策略接口设计需要处理如下挑战：

PD1. 推理时间的延迟匹配 (Inference-time latency maching)。接口假定同步的观测流和立即执行的动作同步，但实体机器人系统并非如此。不处理延迟会严重影响表现。
1. 观测延迟匹配。实体系统中，不同观测流 (RGB image, EE pose, gripper width) 由不同的控制器获取，存在延迟。分别测量延迟 (附录 A1-A3)，在推理时对齐所有观测到延迟最大者 (通常是相机)。首先，下采样 RGB 相机观察到指定的频率 (通常 10-20Hz)，随后使用获取的每张相片的时间戳 $$t_{obs}$$，线性插值 gripper 和 EE pose。双臂情况下，软同步两个相机，即找到最接近的 frame，通常在六十分之一秒内。最后得到同步后的观测，如 Fig. 5(a) 中绿色菱形部分。
2. 动作延迟匹配。策略假定输出是同步的 EE pose 和 gripper widths，但是，实际上，机械臂和 gripper 只能记录目标 pose 序列到一个执行延迟 (execution latency)，此延迟在不同机器人硬件平台不尽相同。为了保障机器人和 gripper 在指定时间内完成策略生成的动作 (执行后达到指定状态)，需要提前发送命令来抵偿，如 Fig. 5(c)，细节见附录 A4。

### A 延迟测量：
1. 相机延迟测量。记录一个计算机屏幕上的周期 (rolling) QR code，每个视频帧都显示了当前系统的 timestamp $$t_{display}$$。为了避免多次检测 QR codes (主要是计算机显示器上展示 GoPro 拍到的内容包含了 QR code)，遮掩相机录像部分的 QR code。通过减去接收到每个视频帧的 timestamp $$t_{recv}$$，和解码 QR code 时间戳 $$t_{display}$$ 和现实刷新的延迟 $$l_{display}$$，得到了相机延迟：$$l_{camera}=t_{recv}-t_{display}-l_{display}$$
2. 本体感知测量。接收到 policy 输出的时间戳 $$t_{recv}$$ 减去发送到机器人的时间戳 $$t_{robot}$$，得到本体感知延迟：$$l_{obs}=t_{recv}-t_{robot}$$。注意，机器人硬件时间戳难以获取，近似为 1/2 的 ICMP 报文的 round-trip time。
3. Gripper 执行延迟测量。端到端延迟 $$l_{e2e}$$，发送一系列正弦位置信号，记录 gripper 何时感知。于是得到：$$l_{action}=l_{e2e}-l_{obs}$$。
4. 机器人执行延迟测量。最终使用 $$l_{e2e}$$。

### B Data Collection Protocol
在新环境收集数据时，需要遵循以下四个步骤：
1. 时间同步。比如双臂场景，同步两个 GoPro 相机。
2. Gripper 标定。
3. 建图。关于每个新场景，通过缓慢移动 gripper 来扫描环境，得到高质量地图，这对健壮的 SLAM 记录十分重要。
4. 演示 。

PD2 相对 EE pose。为了避免依赖于具体的具身平台，作者提出了所有的 EE pose 都是相对于 gripper 当前的 EE pose。
1. 相对 EE 轨迹作为动作的表达。有工作 (Diffusion Policy) 对比了绝对位置相关的动作与增量动作 (delta actions)。具体来说，对于从时间点 $$t_0$$ 开始的动作序列，将其定义为一系列 SE(3) 变换来表示相对于初始时间点 $$t_0$$ 的末端执行器姿态在时间点 $$t$$ 的期望姿态的相对轨迹表示法，可以使系统在数据收集过程中对跟踪误差以及相机位移更加鲁棒。
2. 相对 EE 轨迹表示作为自我感知。使用相对轨迹表达自我感知的历史 EE poses。比如 obs horizon 设为 2，便能够提供速度信息了，作差即可。结合腕部挂在相机，使得我们不需要标定。并且，执行动作期间，移动机器人的 base 也不会影响任务表现，如 Fig. 10 (a)，使得 UMI 框架对于移动机器人操作也使用。
3. 相对 girpper 间的自我感知。在双臂场景，policy 提供两个 gripper 的相对 pose 对双臂协作的任务成功率至关重要。gripper 间感知通过建地图-再定位的数据，根据 IMU 构建场景级别的坐标系统。每个新场景下，首先收集视频，并用于建图。随后，收集的演示数据会重定位到相同的地图，并分享同一坐标系统。

Insight：是否可以把这种相对融合为部分相对和部分绝对？就像在中间位置，设置锚点，相对则考虑动作增量，绝对考虑初始位置与目标的距离关系。在动作序列中，**随机**采样动作序列中的几个位置，作为绝对位置参考，就像 Long Short-Term 的 Long，而**相对**轨迹则贯穿始终，就像 Short-Term 部分。
