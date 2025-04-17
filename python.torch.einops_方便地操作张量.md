---
id: pycd8tjfjszc6x3g7ghbhs8
title: Einops_方便地操作张量
desc: ''
updated: 1744880684241
created: 1741161107888
---

## demo

![concat_along_width](assets/images/python.torch.einops_方便地操作张量/concat_along_width.png)

图中，展示了 b 张 RGB 的图像，所以维度是 (b, h, w, c)。沿着宽度拼接得到如下：

![concate_depth_to_width_height](assets/images/python.torch.einops_方便地操作张量/concate_depth_to_width_height.png)

得到 (h, w * b, c)。可以看到，height 部分保持不变。而 width 变为了 b 张图像宽度的拼接，即 b * w。紧接着再分解深度不分，到宽度和高度。

![diff](assets/images/python.torch.einops_方便地操作张量/diff.png)

如上图，(b1 b2) 部分中，指定了 b1=2，b2 根据 b 自动求出。将原来的 b 拆分为 (b1 b2) 的两个形状，如上图的 (2 3)。转换后，有 (b1 h) 和 (b2 w) 代表了 (2 h) 和 (3 w)，可以看到后面的六张图像变为两行三列，即每列三张。

指定 ims[0] 为 b 张 RGB 图像的第一张，即字母 e。那么操作如下：

```py
rearrange(ims[0], 'h w c -> w h c')
```

![e_0](assets/images/python.torch.einops_方便地操作张量/e_0.png)

```py
rearrange(ims, 'b h w c -> (b h) w c')
```

![e_1](assets/images/python.torch.einops_方便地操作张量/e_1.png)

## rearrange

使用括号，代表展平和转置对应部分。

多数组拼接

```py
import numpy as np
from einops import rearrange, repeat, reduce

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
# r0 是三维的，在 0 维堆叠两个数组
r0 = rearrange([arr1, arr2], 'x y z -> x y z')
```

### 张量的轴 (Axis)

在 rearrange 中，理解张量中**轴 (axis)**的变换是关键。原来轴上的数，会被重排到指定地方​。轴在不同场景有不同含义。

​图像数据：通常为 4D 张量 (batch_size, channels, height, width)。
- 轴 0：批量中的样本数。
- 轴 1：颜色通道（如RGB）。
- 轴 2 和 3：像素的行和列。

​自然语言处理：3D 张量 (batch_size, sequence_length, embedding_dim)。
- 轴0：批量中的文本数。
- 轴1：句子中单词的位置。
- 轴2：词向量的维度。

以**轴**的视角审视数据，会更容易理解 rearrange。比如，在图像场景下，rearrange 指定 "B C H W -> B H W C" 时，变换前的含义如下：原来第二维度指定了颜色通道，指定第二维的下标后，代表后面的 H 和 W 维度对应此通道的内容。变换后的含义如下：置换后，第二维度代表了 H 维度（高度），指定第二维度后，后面维度的内容现在代表此高度下，宽度和通道的值。根据不同的轴，取出对应含义的内容，比如对应哪个通道，哪个批次，甚至高度和宽等。

具体的例子：比如，如下第二维换到第一维，第三维换到第二维，第一维换到最后，即 "k i j -> i j k"。以前最后一维，比如 arr[0, 0, :] 代表 [1, 2, 3]。代表 j 轴在 k 和 i 轴处索引 0 的位置，有切片 0:3 的值为 [1, 2, 3]。如果 rearrange 后，j 轴重排到第二维。要获取 j 轴在 i 和 k 轴 0 索引处的切片 0:3 的数据，需要如此索引 arr[0, :, 0]。原来轴的数，放到不同地方了。存储方面，以前是连续的，现在可能相隔一定距离了。

```py
r1 = rearrange([arr1, arr2], 'k i j -> i j k')
assert np.allclose(r0[0,0,:], r1[0,:,0])
# 沿着一个新维度，第二维 i，拼接并展平
# k i j -> i j k，再 i j k -> i (j k)
result = rearrange([arr1, arr2], 'k i j -> i (j k)')
print(result)
```

```py
# (32, 30, 40, 3)
images = [np.random.randn(30, 40, 3) for _ in range(32)]
# (32, 40, 30, 3)
print(rearrange(images, "b h w c -> b w h c").shape)
# concatenate images along height (vertical axis), 960 = 32 * 30 ：(960, 40, 3)
# 在高度拼接 b 张图像，即 b x h ...
int(rearrange(images, "b h w c -> (b h) w c").shape)
# concatenated images along horizontal axis, 1280 = 32 * 40 ：(30, 1280, 3)
print(rearrange(images, "b h w c -> h (b w) c").shape)
# reordered axes to "b c h w" format for deep learning ：(32, 3, 30, 40)
print(rearrange(images, "b h w c -> b c h w").shape)
# flattened each image into a vector, 3600 = 30 * 40 * 3 ：(32, 3600)
print(rearrange(images, "b h w c -> b (c h w)").shape)
# (c ph) 代表 (1 3)，1 个 3 维的向量。
print(rearrange(images, "b h w (c ph) -> b (c h) (w ph)", ph=1).shape)
```

注意拼接中的顺序：
```py
t1 = torch.arange(24).reshape(2, 3,4)
print(t1)
# 注意后两维度拼接顺序，w在前，所以不再是0 1 2 3 ...，
# 而是先交换两个维度，再 flatten。即 b w h 再 flatten.
print(rearrange(t1, 'b h w -> b (w h)'))
```

在原有维度上，可以添加一个 ph，但是需要保持左右两边维度相同，不能出现左右有不同的维度。新的维度只能用其他方法添加。和上面结果一样，交换 h 和 w 而已。但是新的 ph 维度已经计算。
```py
print(rearrange(t1, 'b h (ph w) -> b w (h ph)', ph=1))
```

使用 ... 表示剩余部分，比如展平剩余部分：

```py
print(rearrange(t1, "b ... -> b (...)"))
```

## repeat 增加维度

注意 "h repeat" 与 "repeat h" 代表不同，repeat 代表重复所在维度，repeat 所在前后，会代表所在轴的位置不同。从而拼接起来，(h repeat) 与 (repeat h) 不同。(h repeat) 中，每连续的 repeat 个都是相同元素，以 repeat 个一组，连续的重复 h 次。(repeat h) 中，每连续 h 个代表原来的张量，以 h 组为单位重复 repeat 次。

如果重复的轴是最后一维，那就将原维度的每个最后分量重复，得到新的轴。d 是新的，重复的维度。由于后面没有维度，所以对原来的 c 轴的每个分量重复 2 次，将以前 c 轴的每个分量变为 2 维的向量，得到形状 (2,2,2)。比如 `[1 2] -> [[1 1] [2 2]]`

```py
arr1 = np.array([[1, 2], [3, 4]])
print(repeat(arr1, 'b c -> b c d', d=2))
```

输出：

```
[[[1 1]
  [2 2]]

 [[3 3]
  [4 4]]]
```

与第一维合并起来：

```py
print(repeat(arr1, 'b c -> (b c) d', d=2))
```

```
[[1 1]
 [2 2]
 [3 3]
 [4 4]]
```

如果重复的轴后面还有轴，那就将后面轴重复。可以这样理解，前面不分的最后一轴的每个分量，就是后面所有轴。重复即重复前面最后一轴的每个分量，并形成新的轴。

```py
print(repeat(arr1, 'b c -> b d c', d=2))
```

输出可以看到，重复了 b 轴的各个分量，例如第一个分量是： `[1 2] -> [[1 2] [1 2]]`：

```
[[[1 2]
  [1 2]]

 [[3 4]
  [3 4]]]
```

拼接：

```py
print(repeat(arr1, 'b c -> (b d) c', d=2))
```

```
[[1 2]
 [1 2]
 [3 4]
 [3 4]]
```

更多的例子：

```py
import numpy as np
from einops import rearrange, repeat, reduce
# a grayscale image (of shape height x width)
image = np.random.randn(30, 40)
# change it to RGB format by repeating in each channel：(30, 40, 3)
# 最后会产生新的维度，将原来最后一维重复 c 次，作为新的最后一维。
# c 所在维度复制，即最后一维复制，最后是连续 c 个相同的数字
# 最后一维是重复的，得到 (h, w, 3)
print(repeat(image, "h w -> h w c", c=3).shape)
# c 所在维度复制，即第二维，连续 c 个 w 维张量相同（复制而来）
# 即 [i, :c, ...] 全是相同的。
print(repeat(image, "h w -> h c w", c=3).shape) # (h, 3, w)
print(repeat(image, "h w -> h (c w)", c=3).shape) # 展平两维，最后一维是重复的 c 次 w
# repeat image 2 times along height (vertical axis)：(60, 40)
print(repeat(image, "h w -> (repeat h) w", repeat=2).shape)
# repeat image 2 time along height and 3 times along width：(30, 120)
print(repeat(image, "h w -> h (repeat w)", repeat=3).shape)
# convert each pixel to a small square 2x2. Upsample image by 2x：(60, 80)
# (w w2) 部分即最后一维重复两次并且拼接。(h h2) 代表 h h2 的拼接。重复 h2 组 (w w2) 后，
# 再拼接起来如此的 h 组。
print(repeat(image, "h w -> (h h2) (w w2)", h2=2, w2=2).shape)
# pixelate image first by downsampling by 2x, then upsampling：(30, 40)
downsampled = reduce(image, "(h h2) (w w2) -> h w", "mean", h2=2, w2=2)
print(repeat(downsampled, "h w -> (h h2) (w w2)", h2=2, w2=2).shape)

# (b repeat h w) 后，最后两维度 (h w) 重复 repeat 次，形成新的一维。
# 再 flatten 中间的 (repeat h)
print(repeat(t1, 'b h w -> b (repeat h) w', repeat=2))
# (b h repeat w) 后再 flatten 中间的 (h repeat)
print(repeat(t1, 'b h w -> b (h repeat) w', repeat=2))
```

## reduce

$$
x = x 
$$

```py
import numpy as np
from einops import rearrange, reduce
x = np.random.randn(100, 32, 64)
# perform max-reduction on the first axis:(32, 64)
print(reduce(x, "t b c -> b c", "max").shape)
# same as previous, but with clearer axes meaning:(32, 64)
print(reduce(x, "time batch channel -> batch channel", "max").shape)
x = np.random.randn(10, 20, 30, 40)
# 2d max-pooling with kernel size = 2 * 2 for image processing:(10, 20, 15, 20)
# 注意括号部分，(h1 h2) 要凑到 30，h2 在参数指定为了 2
y1 = reduce(x, "b c (h1 h2) (w1 w2) -> b c h1 w1", "max", h2=2, w2=2)
print(y1.shape)
# if one wants to go back to the original height and width, depth-to-space trick can be applied:(10, 5, 30, 40)
y2 = rearrange(y1, "b (c h2 w2) h1 w1 -> b c (h1 h2) (w1 w2)", h2=2, w2=2)
print(y2.shape)
# Adaptive 2d max-pooling to 3 * 4 grid:(10, 20, 3, 4)
print(reduce(x, "b c (h1 h2) (w1 w2) -> b c h1 w1", "max", h1=3, w1=4).shape)
# Global average pooling:(10, 20)
print(reduce(x, "b c h w -> b c", "mean").shape)
```

执行 reduce 操作，即对两个 operand 执行 combination 操作，操作包含 sum, mean, max, min, prod。比如：

```py
t1 = torch.arange(24).reshape(2, 3, 4)
# 从第一维中，找到 max，即 dim=0，最后得到 shape 为 (h w)
# 可以看做 从 t1[0] 开始，作为 tmp
print(reduce(t1, "b h w -> h w", "max"))
```

按照 "b h w" 的 view 看待 t1，一次取维度 "h w"，在 b 这一维度索引下一个维度为 "h w" 的来操作。即 res = t1[0], res = max(res, t1[1]), res = max(res, t1[2]) ...，不断 reduce。

```py
x = torch.arange(2*(3*2)*(4*3)).reshape(2,6,12)
print(x)
# 2d max-pooling with kernel size 3*4，stride 也是 kernel size，padding 为 0
# 最后得到 c h1 w1 的 shape。也就是从每个 kernel 窗口选择最大值，stride 为 h1 w1
# 等价的 max-pooling 操作如下：
# 第一个 kernel 规定的窗口是 3x4，对应 x[:, :3, :4]，选取其中的 max 值，即窗口右下角。
# 紧接着 kernel 向右移动 4 个单位，即 w2 个单位，找到第二个 max 值，31；当右侧结束时，
# 行方向向下移动 h2 个单位，即 3 个单位。继续开始。
# 用 reduce 的思想思考，是一并进行的。没在箭头右侧指定的维度都会遍历。
# h1 所在维度方面，保留 h1 的话，需要在展平的 (h1 h2) 中以 h2 为间隔，得到 h1 个值。
# 紧接着 h1 个值每次前进一步，直到遍历完 h2 个值后，所有值都比较了，以 h2 为区间 reduce 出了结果。
# 首先指定的初始张量为：
# i, j = 0, 0
# res = x[:, ::h2, ::w2]
# 相当于:
# for i in range(1, h2):
#     for j in range(1, w2):
#         res = max(res, x[:, i::h2, j::w2])
# 首先移动最后维度，先 ++j，即 for j in range(w2)。结束后再比较 for i in range(h2)，如此循环。
# 每次，选取列方向的移动 w2 的位置的 x[:, :] 与其比较。reduce 后行方向是 h2 个。
# 比较后留下 max 结果。
# 注意，要指定 h2，而非 h1，顺序调换得到结果可能不同。
print(reduce(x, "c (h1 h2) (w1 w2) -> c h1 w1", "max", h2=3, w2=4))
```

    tensor([[[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11],
            [ 12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23],
            [ 24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35],
            [ 36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47],
            [ 48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59],
            [ 60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71]],

            [[ 72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83],
            [ 84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95],
            [ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107],
            [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131],
            [132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143]]])
    tensor([[[ 14,  17,  20,  23],
            [ 38,  41,  44,  47],
            [ 62,  65,  68,  71]],

            [[ 86,  89,  92,  95],
            [110, 113, 116, 119],
            [134, 137, 140, 143]]])

## 嵌入到网络结构中

可以像一层网络，嵌入到 nn.Sequential 中。比如：

```py
from einops.layers.torch import Rearrange
...

net = nn.Sequential(nn.Conv2d(...), nn.Flatten(2), Rearrange("b c n -> b n c"), nn.LayerNorm(dim))
```

## Ref and Tag

[Github](https://github.com/arogozhnikov/einops)
[学习link](https://zhuanlan.zhihu.com/p/342675997)

#Tricks