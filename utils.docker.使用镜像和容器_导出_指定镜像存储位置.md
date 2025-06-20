---
id: l1as99asaza2ux0otwtb82q
title: 使用镜像和容器_导出_指定镜像存储位置
desc: ''
updated: 1750309402053
created: 1740389762581
---

## 使用镜像

打标签来重命名镜像：

```bash
docker tag <image_id> <new_image_name>[:<tag>]
```

## 指定镜像存储位置

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

### 查看存储本地的镜像

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

## 导出和导入容器

### save 导出镜像

```bash
# 导出单个镜像
docker save -o my_image.tar image_name:tag
# 恢复镜像
docker load -i my_image.tar
```

### export 导出容器

```bash
docker export {{container_id}}  -o {{name}}.tar
```

导入：

```bash
docker import {{/path/to/your_container.tar}} {{repository:tag}}
```

或通过管道直接导入：

```bash
cat your_container.tar | docker import - repository:tag
```



注意事项，docker export/import ​​与 docker save/load 的区别​​：
- export/import 操作的是容器文件系统，会丢失历史记录和元数据（如环境变量、端口映射等），仅保存容器当前状态。
- save/load 操作的是完整镜像，保留所有历史记录和层结构，适合迁移镜像。

#### **何时用 `docker save`？**

-   ✅ 需要完整备份或迁移镜像（包括开发、测试、生产环境的镜像版本控制）
    
-   ✅ 需要保留镜像的层结构以优化存储（如共享基础镜像）
    
-   ✅ 需保留构建历史用于调试（如 `docker history`）
    

#### **何时用 `docker export`？**

-   ✅ 只需提取容器的当前文件系统（如分析黑客入侵后的容器状态）
    
-   ✅ 将容器文件系统提供给非 Docker 工具使用（如挂载到虚拟机）
    
-   ✅ 快速创建一个无历史的轻量级镜像（需手动补全配置）


## Ref and Tag

#Docker