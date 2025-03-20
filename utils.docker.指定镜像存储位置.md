---
id: l1as99asaza2ux0otwtb82q
title: 指定镜像存储位置
desc: ''
updated: 1742451130289
created: 1740389762581
---


镜像和容器数据默认存储在 /var/lib/docker，修改为自定义路径，比如 /data1/docker。查看到 /var/lib/docker 的权限是 710，owner 和 group 都是 root，所以用 sudo 创建一个目录，作为存储的地方。

```bash
sudo mkdir -p /data1/docker
sudo chmod 710 /data1/docker
```

修改 `/etc/docker/daemon.json` 的 "data-root"：

```json
{
    "data-root": "/data1/docker"
}
```

迁移现有数据（可选）：

```bash
sudo rsync -a /var/lib/docker/ /path/to/new/docker/storage
```

重启 Docker 服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
docker info # 查看 Docker Root Dir 是否指向了你指定的路径。
```

验证：

```bash
journalctl -u docker.service
```

## 查看存储本地的镜像

```bash
❯ docker images
REPOSITORY    TAG                              IMAGE ID       CREATED         SIZE
ubuntu        22.04                            a24be041d957   7 weeks ago     77.9MB
ubuntu        20.04                            6013ae1a63c2   5 months ago    72.8MB
```

可以看到 ubuntu 22.04 的 ID 有 a24be041d957，于是，在我们指定的 /data1/docker 查看：

```bash
❯ sudo find /data1/docker -name "*a24be04*"
/data1/docker/image/overlay2/imagedb/content/sha256/a24be041d9576937f62435f8564c2ca6e429d2760537b04c50ca50adb0c6d212
```

/data1/docker/image/overlay2/imagedb/content/sha256 保存了镜像元数据，具体来说是哈希值。镜像分层文件则在 /data1/docker/image/overlay2/layerdb/sha256 中。

## Ref and Tag

#Docker