---
id: ovto6hepvtttctxmnypiebq
title: Awesome_terminal_终端各种命令脚本
desc: ''
updated: 1754060698491
created: 1742868524198
---

## apt

安装

```bash
sudo dpkg -i {{pkg_name.deb}}
```

查看当前搜到的可安装包：

```bash
sudo apt search {{关键词}}
```


## 终端复用：Tmux, Tmuxp

tmux 终端复用，tmuxp 管理多个 tmux

```bash
sudo apt install -y tmux tmuxp
```

### Tmux

创建会话，和连接：

```bash
tmux new -s session_name
# 创建后 detach
tmux new -d -s {{session_name}}
#  连接
tmux attach -t session_name
tmux a # 简写
# 向 session 发送命令, C-m 必须要，模拟回车
tmux send-keys -t {{session_name}} '{{commands}}' C-m
# Enter 更直观
tmux send-keys -t {{session_name}} '{{commands}}' Enter
# 查看
tmux ls
```

tmux 命令支持简写语法，比如 tmux new-session 写作 tmux new。

一个会话可以包含多个窗口，一个窗口可以包含多个面板。

关闭：

```bash
# 关闭所有会话
tmux kill-server
tmux kill-session -t {{session_name}}
```

常用命令：
- `{{prefix}}"` 水平拆分创建新面板
- `{{prefix}}%` 垂直拆分创建新面板
- `{{prefix}}c` 创建新窗口
- `{{prefix}}x` 关闭面板
- `{{prefix}}&` 关闭窗口
- `{{prefix}}:kill-session` 关闭会话
- `{{prefix}}d` 把会话挂到后台
- `{{prefix}}{{ctrl+方向键}}` 调整面板边界
- `{{prefix}}{ctrl+h|l}` 左右选择窗口
- `{{prefix}}{{数字}}` 选择对应编号窗口
- `{{prefix}}<Tab>` 切换当前和上一个窗口
- `{{prefix}},` 重命名窗口
- `{{prefix}}$` 重命名会话
- `{{prefix}}z` 最大化当前面板

复制到剪切板：按住 shift 选中后，shift 不放，ctrl + c 复制，粘贴也是 shift + ctrl + v。


可以设置进入终端，即进入 tmux：

```bash
# start tmux
if [[ -z "$TMUX"  ]] && [ "$SSH_CONNECTION" != ""  ]; then
   tmux attach || tmux new
fi
```

#### 创建和后台发送命令

方便 ssh 远程登录执命令。

```bash
# 窗口 1，面板 0 发送命令
tmux send-keys -t my_session:1.0 "ls -l" Enter
# 自动选择第一个创库
tmux send-keys -t mysession:2 "vim ~/file.txt" Enter
```

后台会话中创建新窗口：

```bash
tmux new-window -t my_session: -n "monitor"
tmux send-keys -t mysession:monitor "htop" Enter
```

现有窗口中分割窗口：

```bash
# 窗口 1 右侧分割面板
tmux split-window -h -t my_session:1
```

### Tmuxp：加载 tmux 工具

```bash
sudo apt install -y tmuxp
```

注意，首次加载配置，可能需要设置环境变量，可以放到 .bashrc 或 .zshrc：

```bash
export DISABLE_AUTO_TITLE='true'
```

加载：

```bash
cmd="tmuxp load {{config.yaml} -d"
echo "+ ${cmd}"
# 执行
${cmd}
```

```yaml
session_name: 会话名称
start_directory: ${HOME}  # 会话的起始目录
windows:
  - window_name: win1
    layout: main-vertical
    start_directory: ${HOME}/project1
    panes:
      - shell_command1
      - shell_command2
      - shell_command3
  - window_name: win2
    panes:
      - ls
      - top
  - window_name: win3
    panes:
      - shell_command:
        - echo "Did you know"
        - echo "you can inline"
      - shell_command:
        - ...
```

-   **环境变量替换**：tmuxp 会自动替换**花括号包裹的环境变量**，可以直接使用环境变量到配置文件。适用于以下设置：
    -   start_directory
    -   before_script
    -   session_name
    -   window_name
-   **简写语法**：tmuxp 提供了简写/内联样式语法，适合希望保持工作区简洁的用户。
-   **会话初始化**：可以配置复杂的会话初始化过程，不仅仅是简单的命令

windows 下，layout 可以是从左到右的 even-horizontal, 可以是从上到下的 main-vertical。

在 panes 中，可以配置：
- shell_command / cmd：在面板中执行的命令（字符串或命令列表）
- shell_command_before：在主要命令执行前运行的命令
- shell_command_prompt：设置是否显示命令提示符（默认为 true）
- shell_command_delay：命令执行前的延迟时间（毫秒）

环境参数和目录可以配置 start_directory 和 environment。可以配置 focus 为 true，在当前窗口为焦点。

```yaml
session_name: test-demo
start_directory: ~
windows:
  - window_name: advanced
    layout: even-horizontal
    panes:
      - cmd: top
        start_directory: /var/log
        focus: true
        shell_command_before: 
          - cd /var/log
          - echo "Starting log monitoring"
        options:
          remain-on-exit: true
      - 
        shell_command: 
          - ssh user@remote-server
        start_directory: ~/projects
        # 毫秒
        shell_command_delay: 500
```

关于 yaml 语法，如果命令太长，可以使用 >- 或 | 来处理换行。注意，要在命令或对象的一开始就指出，不能在一半指出。比如：

```yaml
panes:
  - shell_command:
      - >-
        if [[ -n ${SHOULD_LAUNCH_CAMERA} ]]; then
          echo "Launching camera...";
          ros2 launch signal_camera_node signal_camera.py;
        else
          echo "Should not launch camera";
        fi

  - shell_command:
      - |
        if [[ -n ${SHOULD_LAUNCH_CAMERA} ]]; then
          echo "Launching camera..."
          ros2 launch signal_camera_node signal_camera.py
        else
          echo "Should not launch camera"
        fi
```

## curl

常用参数：
- `-C, --continue-at <offset>`: 断点续传。<offset> 可以是字节数，如果是 -，代表自动找到断点部分，自动续传。大C, $ curl -C - https:// # highlight part is necessary，否则就是解析为-C offset从offset偏移下载
- `-o, --output <filename>`: 小 o，相当于wget
- `-O, --remote-name`: 文件名是 URL 最后部分，自动命名。不指定 `-o` 或 `-O` 则默认输出到标准输出。
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

### arp

查看 ARP 缓存，列出 IP 地址与 MAC 地址的对应关系。

```bash
arp -a
```

刷新 ARP 缓存

```bash
arp -n # 查看缓存
sudo ip neigh flush all
sudo ip neigh flush dev {{dev0}}
```

### 配置网卡、通信设备

```bash
sudo ip link set dev {{网卡名}} {{选项}}
```

选项可以是 `up|down`。

其中，dev 是显示指定设备名，有时可以省略，但是为了可读性和避免参数冲突，带上 dev 选项是良好的风格。

### 静态 ip 地址

#### 永久生效

如果使用了 networkd 管理网络，配置 /etc/netplan/01-netcfg.yaml。如果使用了 NetworkManager 管理网络，配置 /etc/netplan/01-network-manager-all.yaml。具体查看 /etc/netplan 目录即可。重点在于设置 dhcp4 为 no，addresses 为具体静态 ip。

```yaml
network:
  version: 2
  # renderer: NetworkManager # 根据本机要求来选择 NetworkManager 或 networkd
  renderer: networkd
  ethernets:
    ens33:
      dhcp4: no # 静态 IP
      addresses: [192.168.19.204/24]
      # gateway4: 192.168.19.1 # 网关设置
      routes:
        - to: 192.168.123.0/24
          via: 192.168.19.57
      nameservers:
        addresses: [8.8.8.8]
```

```bash
sudo netplan apply
```

#### ip 命令临时修改，重启失效

```bash
sudo ip addr add 192.168.1.100/24 dev enp0s3  # 设置IP
sudo ip link set enp0s3 up                     # 启用网卡
```

删除

```bash
sudo ip link set enp0s3 down
sudo ip addr del 192.168.1.100/24 dev enp0s3
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

#### 手动添加路由

```bash
sudo ip route add {{目标}} via {{网关}} dev {{接口}} metric {{优先级}}
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

注意，CIDR 中，指定的位数后面要全部为 0，比如 192.168.123.0/24 的后 8 位全部为 0，不匹配则会报错 Invalid prefix。如果需要对单个主机添加静态路由，即点对点路由，可以：

```bash
# 正确写法（/32掩码）
sudo ip route add 203.0.113.5/32 via 192.0.2.1
```

也可以指定网卡：

```bash
sudo ip route add 203.0.113.5/32 dev eth0
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

#### 删除路由

```bash
sudo ip route del {{目标网络}}
```

如果目标网络相同时，可能影响包的转发。比如，使用多张网卡连接路由器时，路由器会给网口下发路由表，通常是 0.0.0.0 的目标地址，在 ip route 中显示为 default。同为 default 的目标网络，在网络的包找不到路由表的转发条目时，会选择 default 的 metric 最小者，即优先值最高者转发。所以，有些网口总是不会被访问，但又需要它来转发，以访问其他主机。

解决方案：通过其他参数，比如 dev 接口，比如 via 的网关，删除指定转发路由。

```bash
# 根据网卡
sudo ip route del default dev eth1
# 或者根据目标网关
sudo ip route del default via 192.168.8.1
```

#### 修改路由优先级（metric）

```bash
sudo ip route change {{目标网络}} via {{网关}} dev {{接口}} metric {{新优先级}}
```

### ip neigh 处理 ARP

```bash
# 清楚 ARP 缓存
sudo ip neigh flush all
sudo ip neigh show
```

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

### 使用 nmap 扫描本地链路子网有哪些 IP

nmap 注意别侵权等问题，不要扫描别人 IP。

```bash
sudo nmap -sn 169.254.0.0/16
```

- -n 表示不进行 DNS 解析，直接扫描 IP 地址。


### 配置 WiFi

无显示器情况下，SSH 连接板子时，可以让其连接 WiFi：

```bash
nmcli dev wifi list
nmcli dev wifi connect {{wifi名称}} password {{wifi密码}}
```

### 检测 TCP 和 UDP 链接

#### telnet: 快速测试 IP 和端口开启的工具

TELNET 连接实际上是标准 TCP 连接，因此可以使用客户端，测试其他依赖 TCP 作为传输协议的服务。不光可以模拟 HTTP，也一样可以模拟SMTP，POP，都可以，只要目标端口开启，并且是 TCP 协议，就都可以使用 TELNET 来连接，这种方式也往往用于测试目标计算机的某个端口是否开启，开启的是什么服务。例如，通过一个简单的请求，可以检查 HTTP 服务器的功能或状态。

```bash
telnet {{ip_addr}} {{port}}
```

#### 使用 nmap 查看 UDP 的端口开放情况

```bash
# 扫描特定 UDP 端口
sudo nmap -sU -p 123 192.168.1.100
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

在本地执行：

```bash
ssh -N -R 12321:localhost:15558 服务器用户名@服务器IP
```

在本地执行：

```bash
python -m http.server 15558
```

服务端的端口 12321 映射到客户端机器的 15558 端口。在服务器上可以 `telnet localhost 12321` 来访问本地端口。

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


### 实践

#### 网线直连的两主机通信

**手动分配**：

由于没有 DHCP 服务器，所以需要手动分配 IP 地址。分别在两主机上配置 IP 地址和子网掩码，为在同一网段的不同 IP 地址。随后可以随意设置网关，因为 ARP 能够直接获取目标 IP 地址的 MAC 地址，通信仅在两主机间的情况下，包不会被转发到网关，所以任意网关都不会有影响。

配置后尝试互相 ping 通。连接正常后，确保主机要安装 openssh-server，配置 PasswordAuthentication yes 或者其他免密登录方法，才能正常登录。

**使用 Link-Local Only 选项**：

设置后，无需手动配置 IP/子网掩码，连接后立即使用。系统通常自动分配经典的 169.254.0.0/16 范围 IP 地址，这是本地链路常用的。

有时候需要手动清除 IP 并重新分配：

```bash
# 手动清除IP重新分配
sudo ip addr flush dev enp0s25
```

**使用与其他主机共享网络选项（推荐）**：

设置后，本主机就像路由器一样，与对方主机形成局域网。本主机扮演 DHCP 服务器，分配 IP 地址给对方。使用命令 `arp -a` 可以查看 IP 和 MAC 地址，进而连接和登录。

#### 使用与其他主机共享

把本主机当做路由器，此主机能够内置 DHCP 服务器，给对应主机分配 IP 地址，要求另一主机是 DHCP 分配 IP 的方式。之后，此主机可以转发其他主机的流量。本主机的 IP 自然的，末尾会是 1，比如 10,42.0.1。

arp -a 可以看到连接的其他主机的 IP 地址和 MAC 地址。


## SSH

### 安装 ssh

```bash
sudo apt update
sudp apt -y install openssh-server
sudo systemctl enable ssh # 开机启动
sudo systemctl start ssh # 启动
sudo ufw allow ssh # 如果防火墙禁止了 ssh，开启
```

### 生成公钥私钥

生成公钥和私钥：

```bash
# -N 是密码
# -f 指定路径，公钥会在后面添加 .pub 后缀
ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ""
```

### 别名和公钥登录

#### 别名

在 `~/.ssh/config` 文件中，可以指出别名配置。没有此文件创建即可。端口默认 22，可选。

```config
Host 别名
    Hostname 主机名
    Port 端口 # 可选
    User 用户名 

Host server
    HostName 服务器地址
    User 登录用户名
```

指定后，可以用别名登录：

```bash
ssh 别名 # enter
```

#### 公钥登录（免密码）

将公钥拷贝到远程主机，会添加到 /root/.ssh/authorized_keys 或 ~/.ssh/authorized_keys (优先) 文件中，用于验证登录此用户的公钥。

使用工具 ssh-copy-id 自动拷贝公钥到远程主机的 authorized_keys。

```bash
# 拷贝 ~/.ssh/id_rsa.pub
ssh-copy-id {{别名或 username@remote_host}}
# 指定公钥
ssh-copy-id -i {{path/to/certificate}} {{别名或 username@remote_host}}
```

或者手动拷贝，使用 scp 或 sftp 拷贝公钥到远程主机，并且添加到 authorized_keys 即可。

修改 sshd 配置，编辑 /etc/ssh/sshd_config，修改对应选项为 yes：

```
PubkeyAuthentication yes
```

重启服务：

```bash
sudo systemctl restart ssh
```

### 端口配置影响

代理的端口设置可能会导致 VsCode 远程不了。

## sudo 免密码

需要在 root 用户下执行。

```bash
echo "`whoami` ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
```

## tr

替换字符：

```bash
cat > a.txt
asd
ASD
AsD
```

### 转换字符

```bash
cat a.txt | tr AS jkl
asd
jkD
jsD
```

### 补集

处理补集，不匹配前面部分的，全都替换为后面的最后一个字符：

```bash
perl -wnl -e '$_ =~ tr/AS/jkl/c and print;' a.txt
lll
ASl
All
```

```bash
tr -c "AS" "jkl" # 同上
 ```

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
established => 1
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

## 构建工具集：build-essential 等

build-essential 包含了编译 C 程序所需的工具，比如 gcc、g++ 和 make 等。

```bash
sudo apt install build-essential -y
```

## top 和 htop：监控进程状态

单次显示实时的值，可以结合 tail 使用，查看 CPU，内存等占用。

```bash
top -b -n 1 -p {{PID}} | tail -n 2 | perl -nlae 'print "$F[8] $[9]";'
```

## 输入法

```bash
sudo apt update
sudo apt install fcitx fcitx-pinyin -y
```

## 字体

[nerd-fonts](https://github.com/ryanoasis/nerd-fonts)

下载一个 ttf 文件，比如 [CodeNewRoman Nerd Font Mono](https://link.zhihu.com/?target=https%3A//github.com/ryanoasis/nerd-fonts/releases/download/v3.2.1/CodeNewRoman.zip) 系列能够显示 icon，安装后，可以在终端设置此文字，便有等宽字体。

终端 -> 首选项 -> 未命名 -> 文字，选择自定义字体即可。可以设置白色背景，比如，在文字选项卡旁的颜色，在文字和背景颜色中，取消“使用系统主题中的颜色”，可以用 Light 版本，这样取消了茄色的终端背景。

## Ubuntu 界面

在外观 -> Dock -> 面板模式，选择关闭，就可以获取居中的图标。

## 用户管理：配置服务器

```bash
sudo useradd {{USER_NAME}} --create-home --groups adm,sudo,docker --shell /bin/bash
```

随后可以修改密码：

```bash
sudo passwd {{USER_NAME}}
```

可以创建组和添加用户到组：

```bash
sudo usermod -aG groupname[,groupname2,...] username[,username2,username3,...]
# 查看用户
cat /etc/passwd
# 查看用户组
cat /etc/group
```

## 一些意向不到的坑

### sed/perl 字符替换分隔符

使用 sed/perl 替换时，若涉及的变量为路径名或网络路径，内在的 `/` 极有可能导致模式替换出错。比如，配置 NTP 服务器时，从模板替换变量为环境变量，即 envsubst 的功能。使用环境变量替换和指定。

```bash
sed -i "s/\${SUBNET}/${SUBNET}/g" /etc/chrony/chrony.conf.template > /etc/chrony/chrony.conf
```

若 `SUBNET=10.42.0.0/24`，那么命令会替换为：

```bash
sed -i "s/\${SUBNET}/10.42.0.0/24/g" /etc/chrony/chrony.conf.template > /etc/chrony/chrony.conf
```

多出现了 `/`。解决方案是使用 `@` 即可。

## Ref and Tag

作为程序员的你，常用的工具软件有哪些？ - PegasusWang的回答 - 知乎
https://www.zhihu.com/question/22867411/answer/463974547