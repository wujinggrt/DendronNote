---
id: 0l1hfdiwxqo6uoqpldacdvc
title: LocalServerGitRepository
desc: ''
updated: 1739802285988
created: 1739759538365
---

如何在本地服务器构建 git 远程仓库？

确保服务器已安装 Git：
```sh
 sudo apt-get update && sudo apt-get install git -y
```

为安全起见，建议创建一个专用用户（如 git）管理仓库：

```sh
sudo adduser git /
su - git  # 切换到 git 用户
```

## 创建裸仓库
裸仓库没有工作目录，适合作为远程仓库：

```sh
mkdir -p ~/repos/myrepo.git  # 在 git 用户目录下创建仓库目录
cd ~/repos/myrepo.git
git init --bare --initial-branch=main /path/to/repo.git  # 初始化裸仓库
```

## 配置 SSH 访问和仓库推送权限
确保 git 用户的 ~/.ssh 目录存在：

```sh
mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
```
将用户的公钥（id_rsa.pub）添加到 authorized_keys 文件中，每行一个。

客户端配置：用户生成 SSH 密钥（如果尚未生成）：
```sh
ssh-keygen -t rsa -b 4096  # 本地机器执行
```
将公钥内容发送给服务器管理员，或自行添加到服务器的 authorized_keys。


为了确保其他用户能够推送代码到这个仓库，你需要正确配置仓库的权限。一种常见的做法是将仓库的所有者设置为 git 用户，并确保适当的组权限。

```sh
sudo chown -R git:git /srv/git/myrepo.git
sudo chmod -R 755 /srv/git/myrepo.git
```

如果你希望多个用户能够访问这个仓库，可以考虑创建一个专用的组并将这些用户添加到该组中。
```sh
sudo groupadd gitusers
sudo usermod -aG gitusers yourusername
sudo chgrp -R gitusers /srv/git/myrepo.git
sudo chmod -R g+rwX /srv/git/myrepo.git
```

## 禁用 Shell 登录（可选）
为增强安全，限制 git 用户仅能使用 Git 相关操作：

```sh
#修改 /etc/passwd，将 git 用户的 shell 改为 git-shell
sudo chsh git -s $(which git-shell)
```

## 测试远程仓库
客户端克隆仓库
```sh
git clone git@server_ip:~/repos/myrepo.git
```
替换 server_ip 为服务器 IP 或域名。如果在 git 用户下，仓库如果放在 git 的 ~ 目录下，比如 ~/repos/demo.git，在 git clone 时，可以直接省略路径的前缀 `~/`，用法比如:
```sh
git clone git@servier_ip:repos/demo.git
```

如果使用非默认 SSH 端口，需指定端口：
```sh
git clone ssh://git@server_ip:port/~/repos/myrepo.git
```
6.2 推送现有项目
```
cd existing_project
git remote add origin git@server_ip:~/repos/myrepo.git
git push -u origin master
```
7. 权限管理（可选）
简单场景：通过文件系统权限控制（chmod 和 chown）。

复杂场景：使用工具如 Gitolite 或 GitLab 实现细粒度权限控制。

8. 高级配置（可选）
HTTP 访问：配置 Web 服务器（如 Nginx/Apache）并启用 Git HTTP 协议。

钩子脚本：在 myrepo.git/hooks/ 中添加脚本（如 post-receive）自动化部署或通知。

常见问题排查
权限错误：确保仓库目录及其父目录权限允许 git 用户读写。

SSH 连接失败：检查 sshd 服务状态、防火墙设置及 authorized_keys 文件格式。

钩子不生效：确保脚本有可执行权限（chmod +x post-receive）。

通过以上步骤，您已成功在本地服务器搭建了一个 Git 远程仓库，团队成员可通过 SSH 协作开发。根据需求调整权限和扩展功能即可满足不同场景需求。