---
id: jfnhawp9z6yefk5zedupbht
title: 画布划分_subplot
desc: ''
updated: 1742617134718
created: 1742573563940
---

## plt.subplot() 函数

- `subplot(numRows,numCols,plotNum)`
- `subplot(RCN)`

有两种形式绘制几何形状相同的布局。从 1 开始计数，比如 plt.subplot(2, 3, 4) 和 plt.subplot(234) 代表二行三列的 6 个区域中第 4 个，即第二行第一个。

```py
import matplotlib.pyplot as plt
import numpy as np
X=np.linspace(0,2*np.pi,100)
Y1=np.sin(X)
Y2=np.cos(X)
plt.subplot(121) # 注意，可以看到 1 横跨两行
plt.plot(X,Y1)
plt.subplot(222)
plt.plot(X,Y2)
plt.subplot(224)
plt.plot(X,Y2)
plt.show()
```

![subplot_1](assets/images/python.plt.画布划分_subplot/subplot_1.png)

## 对象

整个图像是一个 Figure 对象。一个 Figure 对象包含一个或多个 Axies 对象。每个 Axies 对象有自己坐标系统的绘图区域。

![architecture](assets/images/python.plt.画布划分_subplot/architecture.png)

## 划分画布，直接操作 Figure 和 Axies 对象

plt.gca() 获取当前操作的 Axies 对象。或者使用 plt.subplots(2,2) 的形式操作 Figure 和 Axies 对象或 Axies 轴对象数组 (np.ndarray of Axies objects)。

```py
import matplotlib.pyplot as plt
import numpy as np
X=np.linspace(0,2*np.pi,100)
Y=np.sin(X)
fig,ax=plt.subplots(2,2)
ax[0][0].plot(X,Y)
ax[0][1].plot(X,Y)
ax[1][0].plot(X,Y)
ax[1][1].plot(X,Y)
plt.show()
```

![axies](assets/images/python.plt.画布划分_subplot/axies.png)

## 展示图像

```py
plt.imshow(image)
plt.imshow(mask)
plt.axis("off") # 不展示轴
plt.title("Image")
plt.show()
```

### 展示边框

```py
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
```

## 展示动画（时间序列的图）



## Ref and Tag