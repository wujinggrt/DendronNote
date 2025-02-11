---
id: gqnz9f63oiug596jem6d3m9
title: ManiSkill-Vitac
desc: ''
updated: 1739285011695
created: 1739260392326
---

使用 Realsense D415 深度相机提供视觉信息。Peg 与 hole 之间距离是增加的。

arguments.py 中，从`solve_argument_confict()`中，优先使用传入命令行的参数，再考虑 yaml 配置文件的参数。

在 evaluation.py 中，offset_list 是测试集数据。用于初始化环境。考察是否能够把测试集数据添加进入训练集。

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