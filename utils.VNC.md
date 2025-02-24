---
id: jj9thgwbmiz310p7b21164d
title: VNC
desc: ''
updated: 1740384280528
created: 1740384014855
---

使用 TigerVNC。

## 分配端口

申请的端口是根据 `5900 + <端口号>` 决定，使用前用命令 `vncserver -list` 查看使用中的端口，避免发生冲突。

分配命令和关闭命令如下：

```bash
vncserver :<端口号> -localhost no
vncserver -kill :<port num>
```

## 防锁屏

有时，锁屏后，会出现键盘不能输入的情况。最简单的方法是，设置为不锁屏。Settings -> Privacy -> Screen 中，关闭 Automatic Screen Lock 和 Lock Screen on Suspend 选项。

## Ref and Tag