---
id: 7gk7py1tpsw1r4w8kms09q5
title: Frp_内网穿透
desc: ''
updated: 1744782274251
created: 1744446321714
---


## 安装

```bash
wget -c https://github.com/fatedier/frp/releases/download/v0.55.1/frp_0.55.1_linux_amd64.tar.gz
tar -xvf frp_0.55.1_linux_amd64.tar.gz
```

进入后，可以看到有：
- frpc： frp 客户端执行程序
- frpc.toml：frp 客户端配置文件
- frps：frp 服务端执行程序
- frps.toml：frp 服务端配置文件
- LICENSE：frp 软件开源协议，不用管

pub 用于转发网络请求实现内网穿透，所以它是服务端，使用 frps。loc 是被转发的对象，所以是客户端，使用 frpc. 本文所有技术的整体访问模型如下：

```mermaid
flowchart TD
    subgraph 你的电脑
    Term(终端)
    Borwser(浏览器)
    end
    subgraph pub
    S[./frps -c frps.toml]
    end
    subgraph loc
    CliSsh(./frpc -c ssh.toml)
    CliHttp(./frpc -c http.toml)
    end

    CliSsh <--ssh--> S
    CliHttp <--http--> S
    Term <--ssh--> S
    Borwser <--http--> S
```

## SSH 内网穿透

在 pub 上，修改 frps.toml：

```toml
bindPort = 7000
```

执行：

```bash
# 开启防火墙，一个绑定端口，一个转发端口
sudo ufw allow 7000
sudo ufw allow 7001
./frps -c frps.toml
```

**注意**，如果使用云服务器，需要在网页的控制台上设置防火墙来打开端口，仅仅在命令行中使用 ufw 打开端口是无效的。

在 loc (需要被穿透的设备，比如 NAS 服务器) 上，创建 ssh.toml：

```toml
serverAddr = "x.x.x.x"
serverPort = 7000

[[proxies]]
name = "ssh"
type = "tcp"
localIP = "127.0.0.1"
localPort = 22
remotePort = 7001
```

- 其中 serverAddr 是 pub 的公网 IP，serverPort 和 刚刚 pub 上的 bindPort 保持一致。
- 请保证 localPort 是 loc 上的 ssh 服务端端口号，此处填写 22 是因为 sshd 默认开放的端口就是 22。
- remotePort 是我们后面访问 pub 时需要使用的转发端口。

然后，在后台启动 frpc ：

```bash
./frpc -c ./ssh.toml
```

接下来，可以在电脑上访问 pub 上转发的 ssh 服务了：

```bash
ssh loc的用户名@pub的公网IP -p 7001
```

## HTTP 内网穿透

在 pub 上，修改 frps.toml：

```toml
bindPort = 7000
vhostHttpPort = 7002
```

```bash
sudo ufw allow 7002
./frps -c frps.toml
```

在 loc 一侧，创建 http.toml:

```toml
serverAddr = "x.x.x.x"
serverPort = 7000

[[proxies]]
name = "web"
type = "http"
localPort = 8080
customDomains = ["kirigaya.cn"]
```

然后，在后台启动 frpc ：

```bash
./frpc -c ./http.toml
```

localPort 是具体本地服务器需要的端口号。customDomains 可以绑定公网域名。

这样就可以通过 http://pub的公网IP:7002 来访问 loc 上的 http 服务了。在浏览器输入 `http://<customDomains[0]>:<vhostHTTPPort>` 即可。

配置后，可能发现访问出现问题，pub 端 frps 报错：

```
2025-04-16 13:42:08.108 [W] [vhost/http.go:121] do http proxy request [host: 117.72.39.249:8080] error: no route found: 117.72.39.249 /
...
```

网页提示错误：The page you requested was not found.

分析，没有找到 `/` 请求的路由，可能是没有正确配置 customDomains。

## Multiple SSH services sharing the same port

## 注册到 systemd 服务

```ini
[Unit]
Description = frp server
After = network.target syslog.target
Wants = network.target

[Service]
Type = simple
# 启动frps的命令，需修改为您的frps的安装路径
ExecStart = /root/frp/frps -c /root/frp/frps.toml

[Install]
WantedBy = multi-user.target
```

管理：

```bash
# 启动frp
sudo systemctl start frps
# 停止frp
sudo systemctl stop frps
# 重启frp
sudo systemctl restart frps
# 查看frp状态
sudo systemctl status frps
#开机启动frp
sudo systemctl enable frps
```

## 安全

### 使用 auth.token

在服务器端，编辑 frps.toml，添加 auth.token。客户端要提供一致的 token：

```toml
bindPort = 7000
auth.token = "abc"
```

客户端 frpc.toml 等：

```toml
serverAddr = "x.x.x.x"
serverPort = 7000
auth.token = "abc"
...
```

### 通过 stcp(secret tcp) 提升 frp 穿透的安全性

ftcp 连接意味着目标主机和本机都要安装 frpc，服务器配置不变。客户端配置需增加 secretKey 参数，secretKey 一致的用户才能访问此服务。

## 客户端使用 Docker 更方便


## 实践

### 结合 filebrowser



## 部署命令和脚本

服务器端，pub 端：

```bash
wget -c https://github.com/fatedier/frp/releases/download/v0.55.1/frp_0.55.1_linux_amd64.tar.gz
tar -xvf frp_0.55.1_linux_amd64.tar.gz
rm frp_0.55.1_linux_amd64.tar.gz
cd frp_0.55.1_linux_amd64
```

## Ref and Tag

https://github.com/fatedier/frp

50元云服务器+FRP，实现内网穿透自由 - 略懂的大龙猫的文章 - 知乎
https://zhuanlan.zhihu.com/p/695342265