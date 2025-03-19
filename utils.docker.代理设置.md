---
id: tz6m430sfzjehicuojbggws
title: 代理设置
desc: ''
updated: 1742381153256
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

## Ref and Tag