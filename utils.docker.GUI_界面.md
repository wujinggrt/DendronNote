---
id: 4ogju3fm1rt556puboo1zzb
title: GUI_界面
desc: ''
updated: 1747214993982
created: 1743468987345
---

在 Ubuntu 环境下，使用 GUI 需要挂载 X11 套接字目录 /tmp/.X11-unix 和正确设置 DISPLAY 环境变量。QT_X11_NO_MITSHM=1 环境变量，是 Qt 应用程序中一个与 X Window System（X11）图形显示相关的环境变量，禁用 Qt 对 MIT-SHM（MIT Shared Memory Extension） 扩展的支持，用于解决某些环境下 Qt 应用无法正常显示图形界面或出现崩溃的问题。

```bash
docker run -itd --name ros2 -e QT_X11_NO_MITSHM=1 -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix osrf/ros2:nightly
```

如果服务器上运行了 Docker，需要在远程连接服务器时，希望使用如 Xserver 的方式来展示界面，那么还需要暴露 $HOME/.Xauthority 和设置 --net=host，--net=host 使用与主机相同的网络栈，而非虚拟网络栈，这样方便开发和使用界面。

```bash
docker run -dit \
    --name="isaaclab_demo" \
    --privileged \
    --env DISPLAY=$DISPLAY \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
    -v $HOME/.Xauthority:/root/.Xauthority \
    --runtime=nvidia \
    --net=host \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
    bash
```

```bash
docker run -dit \
    --name myros2 \
    --env DISPLAY=$DISPLAY \
    --env QT_X11_NO_MITSHM=1 \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume $HOME/.Xauthority:/root/.Xauthority \
    --net host \
    osrf/ros2:nightly \
    bash
```

最后，执行 xhost + 来允许用户访问 X11 服务。注意，运行此命令时，需要首先设置允许打印的显示器，xhost + 只会作用于此 DISPLAY 环境变量值对应的显示器，其他显示器不会允许显示：

```bash
export DISPLAY=:16
xhost +  # 允许所有客户端连接（生产环境慎用）$DISPLAY 对应的显示器
# 或更安全的做法：
xhost +local:docker  # 仅允许本地 Docker 容器访问
```

如果使用 VNC，那么用需要暴露端口，指定 -p 5900:5900 或映射需要的即可。又比如 Web VNC 使用需要 -p 6080:80 即可。

## 安装 OpenGL 相关库

有时候，使用 IsaacGym 等，需要安装 OpenGL 的库才能显示图像化界面：

```bash
apt install -y mesa-utils libglvnd-dev libglfw3 libglfw3-dev libgl1-mesa-dev libx11-dev
```

以下是几个库的功能解释：

1. ​​mesa-utils​​: 这是Mesa 3D图形库的实用工具包，主要用于测试和诊断OpenGL/Vulkan相关功能。包含以下工具：
    - glxinfo：查询显卡驱动信息、OpenGL版本及硬件支持细节（如网页1提到的检查驱动是否安装成功）。
    - glxgears：通过渲染旋转齿轮测试3D性能，常用于基础性能评估（如网页2提到的测试llvmpipe性能）。
    - 适用于开发者验证图形驱动配置是否正常。
2. ​​libglvnd-dev​​: 这是Vendor-Neutral GL Dispatch Library的开发包，用于管理不同厂商的OpenGL/EGL实现，支持多GPU环境下的驱动切换。功能包括：
    - 允许系统同时安装NVIDIA、Intel、AMD等不同厂商的OpenGL驱动（如网页6提到的驱动切换）。
    - 提供统一的API接口，开发者无需直接处理不同驱动的兼容性问题。
    - 开发时需要此库以链接libglvnd的API头文件（如网页8提到的配置EGL路径问题）。
3. ​​libglfw3​​ 与 ​​libglfw3-dev​​
    - ​​libglfw3​​：跨平台的窗口管理库，提供OpenGL/Vulkan上下文创建、输入事件处理（键盘、鼠标）等功能。适用于开发图形应用程序（如网页12提到的游戏和图形演示程序）。
    - ​​libglfw3-dev​​：开发版本，包含头文件和静态链接库，用于编译基于GLFW的代码（如网页10提到的OpenGL上下文创建示例）。
4. ​​libgl1-mesa-dev​​: Mesa 3D图形库的核心开发包，提供OpenGL/EGL的软件实现及硬件加速支持：
    - 包含OpenGL头文件（如GL/gl.h）和链接库，用于编译需要OpenGL功能的程序。
    - 支持Intel、AMD等开源驱动（如网页3提到的Mesa在Linux图形栈中的核心作用）。
5. ​​libx11-dev​​: X Window系统的开发包，提供X11协议的核心功能：
    - 包含Xlib头文件和库，用于开发基于X11的图形界面程序（如创建窗口、处理事件）。
    - 是开发传统Linux桌面应用的底层依赖（如网页17提到的X11窗口创建示例）。