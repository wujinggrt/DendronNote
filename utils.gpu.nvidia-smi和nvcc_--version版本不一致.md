---
id: qa3d4ptiu5zlies7n3iriqv
title: Nvidia Smi和nvcc_  Version版本不一致
desc: ''
updated: 1742307163404
created: 1742307053843
---

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

## Ref and Tag