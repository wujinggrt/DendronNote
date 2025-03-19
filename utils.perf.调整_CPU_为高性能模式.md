---
id: c94nomsi6llb1p1j0av6vx1
title: 调整_CPU_为高性能模式
desc: ''
updated: 1742350325827
created: 1742350274860
---

调整CPU为高性能模式需要安装cpufrequtils，随后调整。要求在root用户下执行，命令总结如下：

```bash
sudo su
apt install -y cpufrequtils
for ((cpu=0; cpu<$(nproc); cpu++)); do cpufreq-set -c $cpu -g performance; done # 设置高性能模式
for ((cpu=0; cpu<$(nproc); cpu++)); do cpufreq-set -c $cpu -g ondemand; done # 设置动态调整模式
```

查看 CPU 核心的状态：

```bash
cpufreq-info
```

## Ref and Tag