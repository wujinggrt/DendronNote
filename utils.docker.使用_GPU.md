---
id: vdmg23j454hslejym14pzwt
title: 使用_GPU
desc: ''
updated: 1742873986608
created: 1742306170823
---

首先确保主机正确安装驱动，还有 NVIDIA Container Toolkit。可以使用 nvidia 下配置好 cuda 环境的镜像。也可以更简单，使用 pytorch 官网提供的镜像。

对于 12.4.1，可以下载 docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 镜像。

```bash
docker run --gpus all --rm nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

## 容器

安装英伟达容器工具集参考官网教程：[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
# docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi # 验证安装
sudo systemctl enable docker # 开机自启动
```

随后，使用命令行工具配置 docker 的内容：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

以上命令会修改/etc/docker/daemon.json：

```bash
$ cat /etc/docker/daemon.json 
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

## GPU 设备对镜像可见

### --gpus 选项

```bash
docker run --gpus all 【镜像名】 
# 使用两个GPU 
docker run --gpus 2 【镜像名】 
# 指定GPU运行 
docker run --gpus '"device=1,2"' 【镜像名】
```

### 更精细化的配置

可能会更繁琐，但是控制粒度更细节。普通时候，使用 --gpus all 一把梭即可。

`-e NVIDIA_VISIBLE_DEVICES=0,1` 更通用，不仅适用于 CUDA 应用程序，还适用于其他使用 NVIDIA GPU 的应用程序。而 CUDA_VISIBLE_DEVICES主要针对 CUDA 应用程序。

`-e NVIDIA_DRIVER_CAPABILITIES=compute,utility` NVIDIA_DRIVER_CAPABILITIES 是一个环境变量，用于指定容器内 NVIDIA 驱动程序所需的能力。这些能力包括计算、图形、视频编码和解码等。

注意，`--privileged` 指定后，就算 NVIDIA_VISIBLE_DEVICES=1 也会看到所有显卡。容器会获得主机的几乎所有权限，包括直接访问所有 GPU 设备。这可能会导致 NVIDIA_VISIBLE_DEVICES 环境变量被忽略，因为特权模式下的容器可以直接访问主机的所有设备。就算是 --gpus 1 也一样。

## 例子

```bash
docker run -dit \
    --name="humanup" \
    --privileged \
    --env DISPLAY=$DISPLAY \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --env="XDG_RUNTIME_DIR=/tmp/runtime-root" \
    --runtime=nvidia \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
    bash
```

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

注意配置的卷映射，使用 /tmp.X11-unix 和 $HOME/.Xauthority 在图形化界面比较重要。

`--net=host` 也很重要，让容器使用宿主机的网络栈，而非 Docker 默认创建的虚拟网络。使用场景需为要容器和宿主机网络完全一致的情况。例如，运行 xclock 这类 GUI 应用时，直接使用宿主机的 X11 套接字（DISPLAY=:0）。某些高性能网络应用（如 UDP 广播、多播应用）可能依赖宿主机网络。

默认情况下，Docker 使用 -p 宿主机端口:容器端口 进行端口映射，而 --net=host 让容器直接绑定到宿主机端口。某些特殊网络需求。例如，运行 ping、traceroute 等需要访问宿主机网络设备的命令。

### 服务器的 Docker 使用图形界面

注意，如果 Docker 想用图形界面，需要解决客户端的窗口使用授权问题。使用 `xhost +` 命令，允许所有用户，包括 Docker 访问 X11 显示接口。本地宿主机安装和使用如下：

```bash
sudo apt-get install x11-xserver-utils
xhost +
```

确保你的 X 服务器允许来自容器的连接。`+` 代表允许，`-` 代表禁止。
```bash
xhost +local:root # 允许所有连接
# 或者，对于特定用户，指定：
xhost +si:localuser:your_username
```

## 安装工具

```bash
apt update
apt install -y vim git curl
```

## Ref and Tag

Nvidia 显卡 Docker 配置指引 - MitsuhaYuki的文章 - 知乎
https://zhuanlan.zhihu.com/p/29973252141

https://github.com/QwenLM/Qwen2.5-VL/blob/main/docker/Dockerfile-2.5-cu121

https://hub.docker.com/r/nvidia/cuda/tags

https://hub.docker.com/r/pytorch/pytorch/tags