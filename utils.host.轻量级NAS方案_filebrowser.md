---
id: 587qg83fxdhz2ho0pfz85c8
title: 轻量级NAS方案_filebrowser
desc: ''
updated: 1744711252672
created: 1744698789417
---

## 安装

### Docker

使用 Docker 的方式最方便，参考 [dokcer hub](https://hub.docker.com/r/filebrowser/filebrowser)。

```bash
docker pull filebrowser/filebrowser
docker run \
    -v ~/share:/srv \
    -v ~/tools/package/filebrowser/filebrowser.db:/database/filebrowser.db \
    -v ~/tools/package/filebrowser/settings.json:/config/settings.json \
    -e PUID=$(id -u) \
    -e PGID=$(id -g) \
    -p 8080:80 \
    --name=jing-nas \
    -d  \
    filebrowser/filebrowser
```

需要我们自己创建一个 filebrowser.db 和 settings.json 文件，放在指定目录，否则可能映射出错。filebrowser.db 是可以使用 touch 创建一个空文件，否则会创建为目录，导致出错。容器内部，项目默认提供了 settings.json 参考 [settings](https://github.com/filebrowser/filebrowser/blob/master/docker/root/defaults/settings.json)：

```json
{
  "port": 80,
  "baseURL": "",
  "address": "",
  "log": "stdout",
  "database": "/database/filebrowser.db",
  "root": "/srv"
}
```

可以看到，root 指定了文件根目录，在此目录下的文件都会被分享。我们可以不指定 filebrowser.db 和 settings.json 选项。

```bash
docker run \
    -itd \
    -v ~/share:/srv \
    -e PUID=$(id -u) \
    -e PGID=$(id -g) \
    -p 8080:80 \
    --name="jing-nas" \
    filebrowser/filebrowser
```

### 普通方式

```bash
mkdir -p package/filebrowser && cd package/filebrowser
wget -c https://github.com/filebrowser/filebrowser/releases/download/v2.30.0/linux-amd64-filebrowser.tar.gz
tar -xvf linux-amd64-filebrowser.tar.gz && rm *.tar.gz
```

### 启动

```bash
./filebrowser --address 0.0.0.0 --port 8080 --root ~/files/filebrowser
```

启动后，在 PC 的浏览器中输入 http://<ip>:8080 即可访问文件系统了。初始情况下，账号和密码都为 admin。

## 修改配置

账号和密码都是 admin 太容易被黑了，所以启动完成后的第一件事情就是要先修改账号密码，点击左侧的 Settings，先把语言切换成 简体中文。然后点击 【用户管理】，点击第一个 admin 用户右侧的 修改符号。然后写入新的账号和密码，保存退出即可。

## Ref and Tag

https://zhuanlan.zhihu.com/p/711190609

https://filebrowser.org/installation

https://filebrowser.org/configuration/authentication-method