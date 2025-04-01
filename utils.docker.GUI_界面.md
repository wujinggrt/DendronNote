---
id: 4ogju3fm1rt556puboo1zzb
title: GUI_界面
desc: ''
updated: 1743493631561
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

最后，运行时，允许用户访问 X11 服务：

```bash
xhost +  # 允许所有客户端连接（生产环境慎用）
# 或更安全的做法：
xhost +local:docker  # 仅允许本地 Docker 容器访问
```

如果使用 VNC，那么用需要暴露端口，指定 -p 5900:5900 或映射需要的即可。又比如 Web VNC 使用需要 -p 6080:80 即可。