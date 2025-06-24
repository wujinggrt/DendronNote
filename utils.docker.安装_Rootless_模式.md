---
id: jc1nt5ed5s5obrls1875g1w
title: 安装_Rootless_模式
desc: ''
updated: 1750407438168
created: 1750384217871
---

## 安装和运行

参考 [Rootless mode|Docekr Docs](https://docs.docker.com/engine/security/rootless/#distribution-specific-hint)。

安装依赖，通常是 uidmap 包。

若当前运行了 Docker，需要停止：

```bash
sudo systemctl disable --now docker.service docker.socket
sudo rm /var/run/docker.sock
```

## 最佳实践

### 在 Rootful Docker 运行 Rootless Docker

如果当前 Docker 环境重度依赖 Rootful Docker，不便迁移，可以拉取 Rootless Docker 镜像并运行：

```bash
docker pull docker:dind-rootless
```

```bash
docker run -d --name dind-rootless --privileged docker:dind-rootless
```

随后可以进入此容器，注意没有 bash，只有 sh。

## Ref and Tag

[Rootless mode|Docekr Docs](https://docs.docker.com/engine/security/rootless/)


