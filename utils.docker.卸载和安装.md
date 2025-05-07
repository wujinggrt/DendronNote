---
id: h8tb3c7d4h53sonlniytpp3
title: 卸载和安装
desc: ''
updated: 1746456352039
created: 1744041065305
---

## 安装

### 鱼香 ros 安装

使用鱼香 ros 的脚本安装：

```bash
wget http://fishros.com/install -O fishros && echo 8 | . fishros
sudo systemctl enable --now docker > /dev/null 2>&1 # 启动
# 非 docker 用户添加到组，否则每次都要 sudo
sudo usermod -aG docker your-user
```

### 官网脚本安装

可能需要代理上网。

```bash
curl -fsSL https://test.docker.com -o test-docker.sh
sudo sh test-docker.sh
 ```

## 卸载

```bash
# 删除以前的源的配置
rm /etc/apt/sources.list.d/docker.list
sudo apt update # 重要，重新处理修改后的源
# 删除 Docker 主程序和依赖包
sudo apt-get purge -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

删除安装包，需要把 /etc/docker 目录下的配置删除，否则不会删除安装。

```bash
sudo apt-get -y purge docker-ce
```

/var/lib/docker 的内容，包括镜像、容器、卷和网络，可以保留也可以删除。删除使用：

```bash
sudo rm -rf /var/lib/docker
```

## 研究 fishros 脚本是如何安装 Docker 的

```bash
wget http://mirror.fishros.com/install/install.py -O install.py
```

考察此文件，安装 Docker 对应脚本为 tools/tool_install_docker.py。

## Ref and Tag

https://www.runoob.com/docker/ubuntu-docker-install.html