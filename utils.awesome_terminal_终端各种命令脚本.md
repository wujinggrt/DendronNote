---
id: ovto6hepvtttctxmnypiebq
title: Awesome_terminal_终端各种命令脚本
desc: ''
updated: 1743398715413
created: 1742868524198
---

## 网桥

```bash
sudo apt install bridge-utils
sudo ip link add name br0 type bridge
sudo ip link set dev eth0 master br0
sudo ip link set dev br0 up
```

注意查看设备，比如 eth0 是我们桥接用到的网卡。

设置网桥的 ip 和 mask，决定了 LAN 中的识别方式。

```bash
sudo ip addr add 192.168.1.1/24 dev br0

sudo ip link show br0
sudo ip addr show br0
```

移除：
```bash
sudo ip link set dev br0 down
sudo ip link set dev eth0 nomaster
sudo ip link del br0 type bridge
```

最后恢复
```bash
sudo ip link set dev eth0 up
```

如果配置了静态地址，需要清楚配置
```bash
sudo ip addr flush dev eth0
```

## Ref and Tag