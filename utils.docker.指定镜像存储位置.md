---
id: l1as99asaza2ux0otwtb82q
title: 指定镜像存储位置
desc: ''
updated: 1740390288995
created: 1740389762581
---


镜像和容器数据默认存储在 /var/lib/docker，修改为自定义路径，比如 /data1/docker。查看到 /var/lib/docker 的权限是 710，owner 和 group 都是 root，所以用 sudo 创建一个目录，作为存储的地方。

```bash
sudo mkdir -p /data1/docker
sudo chmod 710 /data1/docker
```

修改 `/etc/docker/daemon.json`：

```json
{
    "data-root": "/data1/docker"
}
```

重启 Docker 服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
docker info # 查看 Docker Root Dir 是否指向了你指定的路径。
```

## Ref and Tag

#Docker