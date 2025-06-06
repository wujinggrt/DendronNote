---
id: ovto6hepvtttctxmnypiebq
title: Awesome_terminal_终端各种命令脚本
desc: ''
updated: 1749197369626
created: 1742868524198
---

## 执行命令

### source 与 .

最佳实践：在交互式命令行中使用 source，在脚本中使用 `.`。

## IO 重定向

### Here Documents

用法：

```bash
cat << EOF
<html> 
  <head> 
    <title>$TITLE</title> 
  </head> 
  <body> 
    <h1>$TITLE</h1> 
    <p>$TIMESTAMP</p>
  </body> 
</html>
EOF
```

有时候，为了美观，不需要前置空白字符（包括制表符）时，可以使用 `<<-` 的形式：

```bash
ftp -n <<- _EOF_ 
  open $FTP_SERVER 
  user anonymous me@linuxbox 
  cd $FTP_PATH 
  hash 
  get $REMOTE_FILE 
  bye 
_EOF_
```

## 工作 job 管理

```py
{{any_command}} &         # 在后台运行某命令，也可用 CTRL+Z 将当前进程挂到后台
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

## 网络工具：ip 工具等

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

安装工具：

```bash
sudo apt install bridge-utils
```

构建虚拟网卡设备，网桥，其中桥接的网卡是 eth0：

```bash
sudo ip link add name br0 type bridge
sudo ip link set dev eth0 master br0
sudo ip link set dev br0 up
```

master 代表网卡 eth0 作为网桥的 br0 的从属，监听 br0 的流量。

设置网桥的 ip 和 mask，决定了 LAN 中的识别方式。

```bash
sudo ip addr add 192.168.1.1/24 dev br0
sudo ip link show br0
sudo ip addr show br0
```

移除需要关闭：

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

### ufw 端口的防火墙

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

### ssh 隧道封装 TCP/UDP 流量

只要能够 ssh 通信，就可以使用隧道封装流量。SSH 隧道在传输层（TCP）之上工作，将原始流量封装在 SSH 的安全连接中。本质上，SSH 隧道通过一条加密的 TCP 连接传输其他协议的数据。

#### 简单的本地端口转发到远程服务器

比如，能够用 ssh 登录服务器:

```bash
ssh root@js2.blockelite.cn -p 10636
```

若服务器监听端口 15558，那么，可以包装如下：

```bash
ssh -N -L 12321:localhost:15558 root@js2.blockelite.cn -p 10636
```

-L 将本地 12321 端口映射到远程服务器的 `localhost:15558`，其中 `localhost` 表示以远程服务器的视角看到的地址，甚至可以写为远程服务器能够访问到的其他地址。在客户端，可以通过访问 `localhost:12321`，访问到服务器的 `localhost:15558` 端口，连接到本地 SSH 隧道端口来转发。

-N 代表不执行任何命令，只转发流量。

比如，在服务端运行：

```bash
python -m http.server 15558
```

在客户端运行如下内容，可以验证访问：

```bash
telnet localhost 12321
```

#### 远程服务器访问本地服务（反向隧道）

在服务端执行：

```bash
ssh -N -R 5556:localhost:5555 用户名@客户端机器
```

服务端的端口 5555 映射到客户端机器的 5556 端口。

#### Tpis：-f, -N 选项

-N 表示不执行远程命令，只作端口转发，连接后不启动 shell 进程，不分配伪终端，节省资源。

-f 表示 SSH 认证后转入后台运行，不占用当前终端。

#### 三种隧道模式

| 隧道类型          | 封装内容                    | 底层传输 | 典型用途                 |
| ----------------- | --------------------------- | -------- | ------------------------ |
| 本地端口转发(-L)  | 任意 TCP 流量               | TCP      | 访问远程数据库、内网 Web |
| 远程端口转发(-R)  | 任意 TCP 流量               | TCP      | 暴露内网服务到公网       |
| 动态 SOCKS(-D)    | 任意 TCP 流量（SOCKS 代理） | TCP      | 浏览器全局代理           |
| UDP 转发(-T 扩展) | UDP over TCP 隧道           | TCP      | DNS 查询、游戏服务器     |


## sudo 免密码

需要在 root 用户下执行。

```bash
echo "`whoami` ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
```

## tr



## Perl

使用 `-n` 或 `-p` 选项时，若没有指定文件名，则从 stdin 读取内容。

在读取开始前和结束时，可以执行准备工作和扫尾工作。这对统计变量十分有用。只需要在代码块 `BEGIN {...}` 和 `END {...}` 部分指出即可。

比如，统计 TCP 端口：

```bash
netstat -an | head | perl -wnla -e 'BEGIN {my %state=();} (/tcp/ or $. == 1) and $state{$F[-1]}++; END {foreach (keys %state){ print "$_ => $state{$_}";}}'
``` 

输出

```
LISTEN => 5
established) => 1
SYN_SENT => 1
ESTABLISHED => 1
```

### 特殊变量

`$_` 所有默认操作内容都会放到此遍历，比如从 stdin 和文件读取的内容，比如 foreach 读取的内容等等。

### 哈希

```perl
my %score = ("barney" => 195, "fred" => 205, "dino" => 30);
```

也可以用-替代引号

```perl
my %score = (-barney=> 195, -fred => 205, -dino => 30);
```

赋值和访问：

```perl
$data{'google'} = 'google.com';
# 或
$data{-google}++； #追加后面的数字为1，内容为：google.com1
```

遍历：

```perl
foreach $key (keys %data){
    print "$data{$key}\n";
}
```

### 类似 Awk 用法

```bash
perl -wnlae '...'
```

默认用 `\s+` 分割，把内容放到数组 `@F` 中。如果需要指定分割的参数，用 `-F {{分割的字符（串）}}` 即可。




## Ref and Tag