---
id: tz6m430sfzjehicuojbggws
title: 端口映射_代理_科学上网
desc: ''
updated: 1745310762855
created: 1742381049066
---

## 端口映射

把容器的 80 端口映射到主机的 8080 端口。

```bash
docker run ... -p 8080:80 ...
```

参数格式 `-p [宿主机IP:]宿主机端口:容器端口[/协议]`，比如 `-p 127.0.0.1:5001:5000/udp`，具体如下：

```
-p, --publish ip:[hostPort]:containerPort | [hostPort:]containerPort
    Publish a container's port, or range of ports, to the host.
```

```bash
docker run -d -p 8080:80 nginx  # 将容器的80端口映射到宿主机的8080端口（TCP）
```

- ​​容器内​​：Nginx 默认监听 80 端口。
- ​​宿主机​​：所有发往宿主机 8080 端口的请求会被自动转发到容器的 80 端口。
​​访问方式​​：

在浏览器输入 http://宿主机IP:8080 → 实际访问的是容器的 80 端口服务（Nginx）。

注意，如果制定 `--net=host`，映射会失效。

## 使用代理

目标是需要让 docker pull 命令享受到 clash 等代理。

## 修改配置文件的方法（推荐）

### docker pull 使用代理

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

注意，端口号设置为对应的值，有可能不是 7890。

### 容器使用代理

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

或者在 docker run 的时候指定显示指定代理的环境变量，参考 Linux 使用代理的设置。

```bash
sudo systemctl daemon-reload #重载systemd管理守护进程配置文件
sudo systemctl restart docker #重启 Docker 服务
```

### 注意

上述设置，会影响容器的环境变量。容器会继续沿用 `http-proxy` 等环境变量，可能导致网络问题。此外，其他程序也可能会收到这两个配置文件的影响，特别是 ~/.docker/config.json 文件。可以在下载完镜像后，把这些配置都删除，避免影响。

```bash
sudo cp ~/.docker/config.json ~/.docker/config.json.bak
sudo rm ~/.docker/config.json
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak
sudo rm /etc/docker/daemon.json
```

## 修改环境变量

在 docker service 启动脚本中，修改启动环境变量。使用 systemctl edit docker.service 命令，会启动编辑 /etc/systemd/system/docker.service.d/override.conf 文件。添加如下内容：

```ini
[Service]
Environment="HTTP_PROXY=http://192.168.1.1:7890"
Environment="HTTPS_PROXY=http://192.168.1.1:7890"
```

修改后，会生成新的配置项文件，启动服务时与原有的 service 文件合并。如果直接在原有 service 文件修改配置，那么更新软件后，修改的内容（所有旧的内容）会被覆盖，需要重新修改。

## 容器使用代理

在 ~/.docker/config.json 同一设置

## Ref and Tag