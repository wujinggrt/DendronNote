---
id: tz6m430sfzjehicuojbggws
title: 代理设置
desc: ''
updated: 1744041043409
created: 1742381049066
---

如果有 clash 等代理，可以设置如下，使得 docker 也能享受代理，能够 pull 镜像。

```bash
sudo vim /etc/docker/daemon.json
```

添加如下内容：
```json
{
    "proxies": {        
        "http-proxy": "http://127.0.0.1:7890",
        "https-proxy": "http://127.0.0.1:7890",
        "no-proxy": "localhost,127.0.0.0/8"
    }
}
```

```bash
sudo vim ~/.docker/config.json
```

添加如下：

```json
{
    "auths": {
        "registry.cn-hongkong.aliyuncs.com": {
            "auth": "xxxxxx="
        }
    },
    "proxies": {
        "default": {
            "httpProxy": "http://127.0.0.1:7890",
            "httpsProxy": "http://127.0.0.1:7890",
            "noProxy": "127.0.0.0/8"
        }
    }
}
```

```bash
sudo systemctl daemon-reload #重载systemd管理守护进程配置文件
sudo systemctl restart docker #重启 Docker 服务
```

## 注意

上述设置，会影响容器的环境变量。容器会继续沿用 `http-proxy` 等环境变量，可能导致网络问题。此外，其他程序也可能会收到这两个配置文件的影响，特别是 ~/.docker/config.json 文件。可以在下载完镜像后，把这些配置都删除，避免影响。

```bash
sudo cp ~/.docker/config.json ~/.docker/config.json.bak
sudo rm ~/.docker/config.json
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak
sudo rm /etc/docker/daemon.json
```

## Ref and Tag