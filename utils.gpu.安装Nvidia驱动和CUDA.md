---
id: qa3d4ptiu5zlies7n3iriqv
title: 安装Nvidia驱动和CUDA
desc: ''
updated: 1753668508148
created: 1742307053843
---

## 完全删除驱动

```bash
sudo apt purge --autoremove '^nvidia-.*'  # 正则匹配移除所有 NVIDIA 驱动包:cite[2]:cite[5]:cite[7]
# cuda 可以不删除
sudo apt purge --autoremove '*cublas*' '*cuda*'  # 清除 CUDA 相关组件（若安装过）:cite[3]:cite[5]
sudo apt autoremove --purge        # 清理残留依赖:cite[1]:cite[8]
```

删除 cuda tookit 的话，还需要删除 /usr/local/cuda*，但是通可以保留。

## CUDA Toolkit

CUDA 版本要与 nvidia-driver 版本对应起来，具体参考 [Component Versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id5) 的对应关系。找到对应版本后，在 [cuda-downloads](https://developer.nvidia.com/cuda-downloads) 页面，进入 [CUDA Toolkit Archive
](https://developer.nvidia.com/cuda-toolkit-archive)，找到对应版本。选择对应系统、架构等，最后最好选择以 runfile(local) 方式下载，比如：

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
# 安装过驱动了，所以选择 --toolkit
sudo sh cuda_12.2.2_535.104.05_linux.run  --silent --toolkit
```

## nvidia-smi和nvcc_--version版本不一致

需要把 nvcc -V 的 cuda 版本更新到与驱动的一致，以 nvidia-smi 显示的为准。为了方便多版本 cuda 切换，可以建立软链接：

```bash
sudo vim ~/.bashrc 
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.1/bin$PATH

# 保存退出，然后执行
source ~/.bashrc
```

```bash
sudo rm -rf cuda
sudo ln -s /usr/local/cuda-11.1/  /usr/local/cuda
```

## 环境遍历控制

```bash
export CUDA_VISIBLE_DEVICES=1
```

使用上述，接下来运行的模型，只要传入 `device="cuda"`，都会只看到第 1 张（0 作为起始）GPU 核心。

## 用 torch 查看 CUDA 设备数量

```py
import torch

def list_cuda_devices():
    print("="*50)
    print("CUDA Device Information")
    print("="*50)
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        return
    
    # 获取设备数量
    device_count = torch.cuda.device_count()
    print(f"\nAvailable CUDA devices: {device_count}")
    
    # 列出所有设备信息
    print("\nDetailed device info:")
    for i in range(device_count):
        prop = torch.cuda.get_device_properties(i)
        print(f"\nDevice {i}: {prop.name}")
        print(f"  Compute Capability: {prop.major}.{prop.minor}")
        print(f"  Total Memory: {prop.total_memory/1024**3:.2f} GB")
        print(f"  Multiprocessors: {prop.multi_processor_count}")
    
    # 当前设备信息
    current_idx = torch.cuda.current_device()
    print(f"\nCurrent device: Device {current_idx} - {torch.cuda.get_device_name(current_idx)}")

if __name__ == "__main__":
    list_cuda_devices()
```

## Ref and Tag