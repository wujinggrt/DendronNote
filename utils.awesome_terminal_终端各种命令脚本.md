---
id: ovto6hepvtttctxmnypiebq
title: Awesome_terminal_终端各种命令脚本
desc: ''
updated: 1744040734478
created: 1742868524198
---

## ip 工具
### 网桥

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

最后恢复:
```bash
sudo ip link set dev eth0 up
```

如果配置了静态地址，需要清楚配置
```bash
sudo ip addr flush dev eth0
```

### ip route 路由

```bash
ip route # 查看路由表
```

用例：内部内网穿透。

服务器：IP 192.168.19.204，子网掩码（假设为 255.255.255.0），网关（假设为 192.168.19.254）。目标：让服务器能访问 192.168.123.0/24 网段的主机（如 192.168.123.81）。

路由器：
- 接口1：IP 192.168.19.57（与服务器同网段 192.168.19.0/24）。
- 接口2：IP 192.168.123.1（子网 192.168.123.0/24，假设是 LAN 口）。

在服务器上添加静态路由，将目标子网流量指向路由器的接口 192.168.19.54。

```bash
sudo ip route add 192.168.123.0/24 via 192.168.19.57
```

如果删除，则：

```bash
sudo ip route dev 192.168.123.0/24 dev {{网卡}}
```

上述指令临时生效。可以修改路由配置，开机启动即生效。配置 /etc/netplan/01-netcfg.yaml

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    ens33:
      addresses: [192.168.19.204/24]
      routes:
        - to: 192.168.123.0/24
          via: 192.168.19.57
      nameservers:
        addresses: [8.8.8.8]
```

```bash
sudo netplan apply
```

确保路由器允许将 192.168.123.0/24 的流量路由到 192.168.19.0/24，并启用 IP 转发。

如果路由器或服务器启用了防火墙，需允许跨子网通信。

```bash
# 允许来自 192.168.123.0/24 的流量（如使用 ufw）
sudo ufw allow from 192.168.123.0/24
```

设置路由后，发现 ping 不通。查看 arp -a，192.168.19.57 已经有正确的 MAC 地址。使用 traceroute 工具查看，执行 traceroute -m 15 192.168.19.57，发现包在网关 192.168.19.254 后，找不到 192.168.19.57 了。具体来说，

## ufw 端口的防火墙

有时需要打开对应端口，其他主机才能访问。

```bash
sudo ufw status
sudo ufw allow {{Port}}/tcp
sudo ufw allow from 192.168.123.0/24 # 允许来自此网络的流量
```

## sudo 免密码

需要在 root 用户下执行。

```bash
echo "`whoami` ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
```

## Ref and Tag