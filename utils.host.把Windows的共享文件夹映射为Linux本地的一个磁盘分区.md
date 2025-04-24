---
id: qfeqj0o8xylxbs32goxmvql
title: 把Windows的共享文件夹映射为Linux本地的一个磁盘分区
desc: ''
updated: 1745478553160
created: 1745474388082
---

在 Windows 上，把需要分享的目录设置共享。操作如下：右键需要共享的文件夹 → 属性 → 共享 → 选择共享用户（如Everyone或指定用户），设置读写权限。

## 使用 Docker 启动 WebDAV 服务

参考 [hacdias/webdav](https://github.com/hacdias/webdav)。

在 wsl2 下，拉取镜像：

```bash
docker pull ghcr.io/hacdias/webdav:latest
```

在当前目录下，编辑配置：

```bash
cd ~/packages/webdav
vim config.yml
```

内容如下：

```
port: 6060
directory: /data

users:
  - username: wj-24
    password: sciencerobotics
```

登录需要上述用户名和密码，为了安全。

紧接着，启动镜像：

```bash
export SHARED_DIR=/mnt/d/Dendron/notes
docker run --rm \
  -p 6060:6060 \
  -v $(pwd)/config.yml:/config.yml:ro \
  -v $SHARED_DIR:/data \
  ghcr.io/hacdias/webdav -c /config.yml
```

### 配置内网穿透

公网服务器启动了 frps，需要把此端口的内容转发过去。使用 `ss -l src :6060` 可以看到端口 6060 是 TCP 协议的。所以使用 tcp 转发。编辑 ./webdav.toml 如下：

```toml
serverAddr = "117.72.39.249"
serverPort = 7000
auth.token = "xxx"

[[proxies]]
name = "webdav"
type = "tcp"
localIP = "127.0.0.1"
localPort = 6060
remotePort = 6060
```

启动转发：

```bash
./frpc -c webdav.toml
```

接下来，可以在公网上看到此服务。

## Linux 下挂载 WebDAV

使用 davfs2 实现挂载。首先安装：

```bash
sudo apt -y install davfs2
```

挂载如下：

```bash
mkdir -p ~/webdav/win_notes
# 可选，自动挂在
echo "/your/local/path /mnt/webdav davfs user,noauto 0 0" | sudo tee -a /etc/fstab
sudo mount -t davfs wujingdp.xyz:6060 ~/webdav/win_notes
```

会提示输入用户名密码，根据设置输入即可。如果没有，则回车即可。

设置自动挂仔，​​编辑/etc/fstab添加：

```bash
wujingdp.xyz:6060 ~/webdav/win_notes davfs _netdev,noauto,user 0 0
```

取消挂载：

```bash
sudo umount /mnt/webdav
```

## Ref and Tag

怎么把Windows的共享文件夹映射为Linux本地的一个磁盘分区？ - youjia的回答 - 知乎
https://www.zhihu.com/question/451313514/answer/1806390822