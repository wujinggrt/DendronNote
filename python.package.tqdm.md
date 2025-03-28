---
id: 0z88wmjgbyi3yli8ctgqmyz
title: Tqdm
desc: ''
updated: 1742654662905
created: 1742544199779
---

tqdm 在 pip 或 conda 安装之后，只要传给 tqdm 迭代器，即可使用。leave=True 代表程序运行结束后，在终端留下进度条，不会清空终端。leave=False 则在程序运行结束后，不显示进度条。

```py
for i in tqdm(range(100), desc="Processing"):
    time.sleep(0.1)

for i in tqdm(range(100), leave=False):
    time.sleep(0.1)
```

```py
pbar = tqdm(total=100)
for i in range(100):
    # 模拟耗时操作
    time.sleep(0.1)
    pbar.update(1)

with tqdm(total=100) as pbar:
    for i in range(100):
        time.sleep(0.1)
        pbar.set_postfix(loss=i/100, epoch=i)
```

多层嵌套：

```py
for i in tqdm(range(3), desc="Outer Loop"):
    for j in trange(100, desc="Inner Loop", leave=False):
        time.sleep(0.1)
```

## Ref and Tag