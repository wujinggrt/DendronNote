---
id: noqowzv4vfcwhq9rvzq28te
title: 搭建本地仓库
desc: ''
updated: 1742057907651
created: 1742053919752
---

创建私人仓库有两种方式。一种是 docker 自带的本地私有仓库，另一种是 harbor 私有仓库。

## 建立本地 Docker Registry

拉取 registry 的镜像。

```bash
docker pull registry
docker run -d \
  --name registry \
  -p 5000:5000 \
  -v /path/to/your/registry:/var/lib/registry \
  registry:latest
```

将宿主机上的目录挂载到容器内的 /var/lib/registry 目录，用于持久化存储镜像数据。

### 验证

验证如下：

```bash
curl http://localhost:5000/v2/_catalog
```

返回类似如下，说明配置成功。

```console
{"repositories":[]}
```

### 推送

标记、推送私有仓库并验证：

```bash
docker tag my-image:latest localhost:5000/my-image:latest
docker push localhost:5000/my-image:latest
curl http://localhost:5000/v2/_catalog
```

返回如下说明成功：

```console
{"repositories":["my-image"]}
```

### 拉取

```bash
docker pull localhost:5000/my-image:latest
```

### 配置 HTTPS（可选，推荐）

默认使用 HTTP，但是用 HTTPS 更安全，推荐。

## Ref and Tag