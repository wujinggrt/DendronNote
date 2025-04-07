---
id: 2exktpevaxn28xkkqlgb22y
title: Ssh_用法和配置
desc: ''
updated: 1744010584333
created: 1744008822884
---

## 安装 ssh

```bash
sudo apt update
sudp apt -y install openssh-server
sudo systemctl enable ssh # 开机启动
sudo systemctl start ssh # 启动
sudo ufw allow ssh # 如果防火墙禁止了 ssh，开启
```

## 生成公钥私钥

生成公钥和私钥：

```bash
# -N 是密码
# -f 指定路径，公钥会在后面添加 .pub 后缀
ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ""
```

## 别名和公钥登录

### 别名

在 ~/.ssh/config 文件中，可以指出别名配置。没有此文件创建即可。端口默认 22，可选。

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

### 公钥登录（免密码）

将公钥拷贝到远程主机，保存到 /root/.ssh/authorized_keys 或 ~/.ssh/authorized_keys 文件中。

使用工具 ssh-copy-id 自动拷贝公钥到远程主机的 authorized_keys。

```bash
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

## Ref and Tag