---
id: ovto6hepvtttctxmnypiebq
title: Awesome_terminal_终端各种命令脚本
desc: ''
updated: 1745865669493
created: 1742868524198
---

## 工作 job 管理

```py
{{any_command}} &             # 在后台运行某命令，也可用 CTRL+Z 将当前进程挂到后台
jobs                      # 查看所有后台进程（jobs）
bg                        # 查看后台进程，并切换过去
fg                        # 切换后台进程到前台
fg {job}                  # 切换特定后台进程到前台
```

bash 终端通过 () & 语法就能开启一个线程，所以对于上面的例子，可以归并到如下一个脚本中：

```bash
(cd renderer && npm run serve | while read -r line; do echo -e "[renderer] $line"; done) &

(cd service && npm run serve | while read -r line; do echo -e "[service] $line"; done) &

wait # 等待所有当前 shell 启动的进程结束
```

## 函数

### 局部变量

```bash
myfunc() {
  local local_var="仅在函数内可见"
  echo $local_var  # 正常输出
}
myfunc
echo $local_var    # 无输出（变量已销毁）
```

## curl

常用参数：
- `-C, --continue-at <offset>`: 断点续传。<offset> 可以是字节数，如果是 -，代表自动找到断点部分，自动续传。大C, $ curl -C - https:// # highlight part is necessary，否则就是解析为-C offset从offset偏移下载
- `-o, --output <filename>`: 小 o，相当于wget
- `-O, --remote-name`: 文件名是URL最后部分，自动命名。不指定 `-o` 或 `-O` 则默认输出到标准输出。
- `-#`: 进度条
- `-L, --location`:  (HTTP) 请求返回3XX code，server迁移了，则 redo 自动使用新的服务器 addr，会自动跳转到新的 URL。
- `-s, --silent`: 禁止 curl 输出进度和错误信息，只显示下载结果。在执行下载的shell内容有用，比如 `$ sudo bash -c $(curl -fsL URL)`
- `-S, --show-error`: 即使禁用了 -s 选项，也会显示错误信息。
- `-f`: fail silently, This is mostly done to better enable scripts etc to better deal with failed attempts. 不报错，特别是 `curl -f ... && CMDS ...` 中，curl 失败不会影响后续命令进行。
- `-l`: 仅仅返回状态码
- `-k`: allow insecure server connections.

请求服务相关参数：
- `-G, --get`
- `-A, --user-agent {{agent-name}}`: 指定客户端的用户代理标头，即User-Agent。curl 的默认用户代理字符串是 curl/[version]。
- `-H, --header {{header}}`: HTTP 场景
- `-d, --data {{data}}`: HTTP MQTT 场景下，在 POST 请求时。如果发送二进制内容，使用 --data-binary；发送不转义的内容，比如 @ 等，使用 --data-raw。例子：
    curl -d "name=curl" https://example.com
- `-D, --dump-header {filename}`:  将响应头保存到文件。
- `-X, --request {method}`: method 可以是 GET，HEAD，POST 和 PUT。默认 GET。
- `-k, --insecure`: 在 SFTP 和 SCP 场景，跳过 known_hosts
- `--interface {name}`: 指定网卡，比如 eth0:1
- `-x, --proxy [protocol://]host[:port]`

curl --proxy http://proxy.example https://example.com

```bash
curl -A 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36' https://google.com
```

上面命令将User-Agent改成 Chrome 浏览器。

```bash
curl -X POST http://host:port/v1/chat/completions \   
 -H "Content-Type: application/json" \    
-H "Authorization: Bearer 在命令行设置的api-key" \    
-d '{    
"model": "qwen2.5-VL-7B-Instruct",    
"messages": [
   {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},        {"type": "text", "text": "图片里的文字是啥?"}    ]}    ]    }'
```

## ip 工具

### 静态 ip 地址

如果使用了 networkd 管理网络，配置 /etc/netplan/01-netcfg.yaml。如果使用了 NetworkManager 管理网络，配置 /etc/netplan/01-network-manager-all.yaml。具体查看 /etc/netplan 目录即可。重点在于设置 dhcp4 为 no，addresses 为具体静态 ip。

```yaml
network:
  version: 2
  # renderer: NetworkManager # 根据本机要求来选择 NetworkManager 或 networkd
  renderer: networkd
  ethernets:
    ens33:
      # dhcp4: no # 静态 IP
      addresses: [192.168.19.204/24]
      # gateway4: 192.168.19.254 # 网关设置
      routes:
        - to: 192.168.123.0/24
          via: 192.168.19.57
      nameservers:
        addresses: [8.8.8.8]
```

```bash
sudo netplan apply
```

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

上述指令临时生效。可以修改路由配置，开机启动即生效。配置 /etc/netplan/01-netcfg.yaml 或 /etc/netplan/01-network-manager-all.yaml，具体根据网络管理工具。

```yaml
network:
  version: 2
  # renderer: NetworkManager # 根据本机要求来选择 NetworkManager 或 networkd
  renderer: networkd
  ethernets:
    ens33:
      # dhcp4: no # 静态 IP
      addresses: [192.168.19.204/24]
      # gateway4: 192.168.19.254 # 网关设置
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
sudo ufw enable # 打开防火墙
sudo ufw status
sudo ufw allow {{Port}}
sudo ufw allow {{Port}}/tcp comment "{{Comment msg}}"
sudo ufw allow from 192.168.123.0/24 # 允许来自此网络的流量
sudo ufw deny {{port}}
sudo ufw delete {{rule_number}} # 删除规则
```

## sudo 免密码

需要在 root 用户下执行。

```bash
echo "`whoami` ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
```

## Ref and Tag