---
id: n89ajv3jnm7v87o7x0z3apa
title: 磁盘分区和挂载
desc: ''
updated: 1741951180722
created: 1741940358214
---

挂载和操作磁盘，需要在 root 用户环境下才能操作。以下操作都默认为 root 权限，或是 sudo 执行。parted 命令中，-l 显示磁盘信息，-h 帮助。

## 分区工具 parted

使用 parted <设备或分区> 进入磁盘分区交互界面。不指定则进入交互界面再选址。例如：

```bash
parted /dev/nvme0n1
```

一般设备名的最后为 p<分区号>，比如 `/dev/nvme0n1p1`。

以 /dev/sdb 设备为例。进入交互界面：

```bash
parted /dev/sdb
```

在交互界面，可以使用 print 命令查看选中设备的分区情况。

### 使用 select 命令选择编辑的设备

如果在进入交互界面时，没有指定设备，可以使用 select <设备> 来选择。比如：

```bash
(parted) select /dev/sdb
```

### 设置分区

使用 mklabel gpt 设置设备分区表，格式为 gpt。使用 help mklabel 查看如何使用。

紧接着，开始对设备对应区域分区，比如：

```bash
 (parted) mkpart /dev/sdb1 xfs  0%  100%
```

用法为 mkpart PART-TYPE [FS-TYPE] START END。
- START END 选择可以是百分比，避免出现对齐块问题。也可以是 数字<单位>。单位为 MiB/GiB/TiB，比如 1G。END 可以为 -1，代表最后一个扇区。

### 例子

```bash
(parted) select /dev/nvme0n1
使用 /dev/nvme0n1
(parted) print
型号：ZHITAI TiPlus7100 1TB (nvme)
磁盘 /dev/nvme0n1: 1024GB
扇区大小 (逻辑/物理)：512B/512B
分区表：msdos
磁盘标志：

编号  起始点  结束点  大小    类型      文件系统  标志
 1    1049kB  524MB   523MB   primary   fat32     启动, esp
 2    525MB   1024GB  1024GB  extended
 6    525MB   512GB   512GB   logical   ext4
 5    512GB   1024GB  512GB   logical   ext4
```

在另一个终端，使用 lsblk 查看已经挂载的分区，找出未挂载的，进一步初始化。

```bash
> lsblk
...
nvme0n1     259:0    0 953.9G  0 disk
├─nvme0n1p1 259:1    0   499M  0 part /boot/efi
├─nvme0n1p2 259:2    0     1K  0 part
├─nvme0n1p5 259:3    0 476.7G  0 part
└─nvme0n1p6 259:4    0 476.7G  0 part /
```

可以看到，需要初始化并挂在 /dev/nvme0n1p5，并挂载到根目录。可以看出，目录已经分区，但是没有初始化文件系统，没有挂载。不再需要重新分区，初始化文件系统和挂载即可。

## 初始化文件系统

```bash
sudo mkfs -t ext4 /dev/nvme0n1p5
```

## 挂载文件系统

磁盘分区，初始化文件系统后，需要挂载到目录下。注意，需要 root 用户。

```bash
mkdir /path/to/mountpoint
mount /dev/nvme0n1p5 /path/to/mountpoint
chmod a+x /path/to/mountpoint
```

卸载文件系统，从挂载点移除。

```bash
umount -f /path/to/mountpoint # 卸载 FS，-f 代表 force
```

## 持久化挂载

实现开机自动挂在，比如：

```bash
echo "/dev/sdb1   /home/wujing/sdb1  xfs   defaults  0 0 " >> /etc/fstab 
```

可以使用 UUID 替换设备名，避免设备位置更换后，名字修改。使用如下命令查看 sdb1 UUID：

```bash
$ sudo blkid | grep sdb1
/dev/sdb1: UUID="d9ae7fb6-ca5a-4e94-b5dc-41f7a6021e8f" TYPE="ext2" PARTLABEL="sdb1" PARTUUID="9594e121-ba23-4f8d-a6ad-984056502559"
```

## 合并同一设备两个分区


## sd

160 230400 /[]

4x8x16，

## Ref and Tag