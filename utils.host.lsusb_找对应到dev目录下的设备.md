---
id: 6r29c6cv87kbdncc60fs6sz
title: Lsusb_找对应到dev目录下的设备
desc: ''
updated: 1748327631402
created: 1748325958532
---

lsusb 命令会列出设备，关键信息：
关键信息：
- Bus XXX - USB总线号
- Device YYY - 设备号
- ID aaaa:bbbb - 厂商ID:产品ID

在 /dev 目录下找到对应设备，方便程序打开。

使用 dmesg 或 journalctl 命令：

```bash
dmesg | grep -i usb
journalctl -b | grep -i usb
```

可以看到日志，其中有 idVendor

```
...
[   12.123456] usb 3-5.1: new full-speed USB device number 6 using xhci_hcd
[   12.123456] usb 3-5.1: New USB device found, idVendor=046d, idProduct=c539, bcdDevice= 39.06
...
```

不是所有USB设备都会在/dev下有对应节点（如HID设备）。存储设备通常是/dev/sdX，串行设备是/dev/ttyUSBX。设备节点可能会在每次插入时变化，建议使用by-id或by-path链接。

## Ref and Tag