---
id: spdyvdpzn03gdr0e9qabkp7
title: HDF5_h5py
desc: ''
updated: 1740166250503
created: 1740160885848
---

安装：`pip install h5py`。

## 核心概念
h5py 是 Python 接口，用于访问 HDF5 二进制数据格式。HDF5 数据格式存储在单一文件，支持内存映射。HDF5 文件命名通常为 `<filename>.hdf5`。

HDF5 文件存放两类对象：Dataset 和 Group。Dataset 类似 Numpy Array。Group 类似目录，存放 Dataset 和其他 Group，Group 就像树结构的节点，所以是分层结构。

```py
import h5py
f = h5py.File('mytestfile.hdf5', 'r')
```
返回的 `h5py.File` 对象就像字典，可以使用 `keys()` 方法查看所有 keys。随后使用 `[]` 访问 key 对应的 value，value 可能是 `Group`，可能是 `Dataset`。

### 创建文件
传递 `w` 给 `File` 对象即可。

```py
f = h5py.File("mytestfile.hdf5", "w")

# 创建 Dataset
dset = f.create_dataset("mydataset", (100,), dtype="i")

# 需要关闭文件对象
f.close()
```

官方推荐使用 with as 语句:
```py
with h5py.File("mytestfile.h5py", "r") as f:
    print(f.keys())
    print(f["mydataset"])
<KeysViewHDF5 ['mydataset']>
<HDF5 dataset "mydataset": shape (100,), type "<i4">
```

### 创建 Group 和 Dataset
使用 `create_group()` 方法创建 `Group`。创建后的 `Group` 在文件根目录下，此 `Group` 也可使用 `create_<group|dataset>()` 在其下创建 `Group` 和 `Dataset`。HDF5 会像 POSIX 风格一样以 `/` 分隔 `Group`。

```py
with h5py.File("mytestfile.h5py", "a") as f:
    grp = f.create_group("subgroup")
    dset2 = grp.create_dataset("grp_dataset", (50,), dtype='f')

    grp2 = grp.create_group("subgroup2")
    dset3 = grp2.create_dataset("grp2_dataset", (50,), dtype='f')

    print(f.name) # /
    print(dset2.name) # /subgroup/grp_dataset
    print(dset3.name) # /subgroup/subgroup2/grp2_dataset
    print(f["subgroup/subgroup2/grp2_dataset"])
    # <HDF5 dataset "grp2_dataset": shape (50,), type "<f4">

    # 迭代查询文件下，即根 Group 下的内容。注意，不会迭代子 Group 下的内容，比如 subgroup2
    for name in f:
        print(name)
    # mydataset
    # subgroup
```

### 判断元素存在与否
`File` 和 `Group` 可以用 `in` 判断 key 是否存在。key 可以用 `/` 分割。
```py
with h5py.File("mytestfile.h5py", "r") as f:
    print("grp_dataset" in f) # False
    print("subgroup2" in f) # False
    print("grp2_dataset" in f["subgroup/subgroup2"]) # True
    print("subgroup2/grp2_dataset" in f["subgroup"]) # True
```

还有 `keys()`, `values()`, `items()`, `iter()` 和 `get()` 等方法可以使用于 `File` 和 `Group`。

### 遍历
遍历一个 `Group` 只能得到直接相连的成员，如果迭代所有，包括子孙，需要使用 `Group` 的 `visit()` 和 `visititems()` 方法：
```py
def printname(name):
    print(name)
f.visit(printname)
mydataset
subgroup
subgroup/another_dataset
subgroup2
subgroup2/dataset_three
```

### 属性
HDF5 可以存储元数据，称为属性 (attributes)。

```py
dset.attrs['temperature'] = 99.5
dset.attrs['temperature']
99.5
'temperature' in dset.attrs
True
```

## Dataset
### 常用属性
- ds.shape (`tuple(int)`): 类似 Numpy 数组的 shape。
- ds.dtype (`dtype`): 比如 `dtype(int32)`


### 支持切片
```py
ds[...] = np.arange(100)
```

## 高性能实战
### 高效率切片
如果 key 不存在，则直接创建 `Dataset`。

```py
with h5py.File("mytestfile.h5py", "w") as f:
    f["data1"] = np.random.rand(100, 1000) - 0.5
    dset = f["data1"]
    print(dset)  # <HDF5 dataset "data1": shape (100, 1000), type "<f8">

    # 切片
    out = dset[0:10, 20:70]  # shape (10, 50)
```

切片操作背后的细节：
1. h5py计算出结果数组对象的形状是(10, 50);
2. 分配一个空的NumPy数组，形状为(10, 50);
3. HDF5选出数据集中相应的部分;
4. HDF5将数据集中的数据复制给空的NumPy数组;
5. 返回填好的NumPy数组。

读取数据之前有不少隐藏的开销。不仅需要为每一次切片创建一个新的NumPy数组，还必须计算数组对象的大小，检查切片范围不超出数据集边界，让HDF5执行切片查询。这引出了我们在使用数据集上的第一个也是最重要的性能建议：**选择合理的切片的大小。**

比较执行效率：
```py
def run1():
    for ix in range(100):
        for iy in range(1000):
            val = dset[ix, iy]
            if val < 0:
                dset[ix, iy] = 0

def run2():
    for ix in range(100):
        val = dset[ix, :]
        val[ val < 0 ] = 0
        dset[ix, :] = val
```

执行run1用时11.3秒，执行run2用时27.8毫秒，相差大约400倍！run1进行了100 000次切片操作，run2则仅有100次。事实：对内存中的NumPy数组切片速度非常快，但对磁盘上的HDF5数据集进行切片读取时就会陷入性能瓶颈。所以尽可能读到内存操作，避免过多 IO。

写入切片的步骤，以 `dset[0:10, 20:70] = out*2` 为例：
1. h5py计算出切片大小并检查是否跟输入的数组大小匹配；
2. HDF5选出数据集中相应的部分；
3. HDF5从输入数组读取并写入文件。

### 分块存储
HDF5 数据集默认连续存储，扁平地保存。比如，100 张灰度图像数据集：

```py
dset = f.create_dataset("Images", shape=(100, 480, 640), dtype=np.uint8)
```

连续存储的数据集，将图像以 640 个元素的扫描线的形式一条一条地保存在磁盘上。即维度从后向前地顺序来保存。如果我们读取第一张图像，切片的代码将会是：
```py
image = dset[0, :, :]
print(image.shape) # (480, 640)
```

这会选取连续的 480 个以 640 元素为单位的块。取出连续分块的内容速度较快。但是，如果有跳步，那么将会消耗更多 IO。比如：
```py
image = dset[0, 0:64, 0:64] # shape of (64, 64)
```

利用分块存储优化，按照指定形状扁平地写入磁盘，这些块存在文件各地，由 B 树索引：
```py
dset = f.create_dataset('chunked', shape=(100, 480, 640), dtype=np.uint8, chunks=(1, 64, 64))
print(dset.chunks) # (1, 64, 64)
```

chunks 在指定后无法更改。如果是 None 则代表没有分块。

## 补充材料
[Doc: Quick Start](https://docs.h5py.org/en/latest/quick.html)
[当Python遇上HDF5--性能优化实战 - 张玉腾的文章 - 知乎](https://zhuanlan.zhihu.com/p/34405536)