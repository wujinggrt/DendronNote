---
id: p66bmobhov4g6qkf1fposfc
title: Ubuntu镜像源更新出现错误The_certificate_issuer_is _nknown
desc: ''
updated: 1745560491035
created: 1745560074091
---

通常在镜像中，更换源时发生错误。解决方案是把 https 的镜像缓存 http 的。更新之后，再安装：

```bash
apt -y install ca-certificates
update-ca-certificates
```

再把源的 http 改成 https。或者直接忽略。

**Ubuntu 22.04**

```bash
sudo sed -i "s@http://.*archive.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

sudo apt update && sudo apt -y install ca-certificates
update-ca-certificates
sudo sed -i "s@http://mirrors.tuna.tsinghua.edu.cn@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
sudo sed -i "s@http://mirrors.tuna.tsinghua.edu.cn@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
```

**Ubuntu 24.04**

```bash
cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.bak
perl -i.bak -wple 's@(URIs:) http://archive.ubuntu.com/ubuntu/@$1 http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g;' \
    /etc/apt/sources.list.d/ubuntu.sources

apt update && apt -y install ca-certificates
update-ca-certificates

perl -i.bak -wple 's@(URIs:) http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@$1 https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g;' \
    /etc/apt/sources.list.d/ubuntu.sources
```

## Ref and Tag