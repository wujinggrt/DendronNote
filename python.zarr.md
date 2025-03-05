---
id: 7cd9he4w15xq7jbt88b3xp4
title: Zarr
desc: ''
updated: 1741166359120
created: 1739872667023
---

Zarr是一种存储分组、压缩的N维数组的格式。

官方教程：https://zarr.readthedocs.io/en/stable/tutorial.html

## Arrays

类似 numpy 的 np.array，例子：
```py
import zarr
store = zarr.storage.MemoryStore()
z = zarr.create_array(store=store, shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
# <Array memory://... shape=(10000, 10000) dtype=int32>
```

接口类似 numpy，读写也有相似之处：
```py
import numpy as np
z[0, :] = np.arange(10000)
z[:, 0] = np.arange(10000)
```

### 打开

如果数据目录不是 Zip 文件，可以直接简化打开，使用：

```py
src = zarr.open(path="path/to/file", mode="r")
```

此时，src 使用了 DirectoryStore。

### 转换为 numpy.ndarry

直接对 zarr 的 Array 使用切片 [:] 访问，便可获得 numpy.ndarray。

### chunks 参数
[ref](https://zarr.readthedocs.io/en/v3.0.0/user-guide/performance.html#chunk-size-and-shape)

决定了数据如何被分割成较小的块（chunks），并存储在存储后端中。合理的 chunks 设置可以显著影响数据的读写性能、压缩效率和存储开销。chunks 参数指定了每个维度上的块大小。Zarr 使用这些块来组织和存储数据，每个块作为一个独立的存储单元进行处理。通过合理设置块大小，可以优化数据访问模式，尤其是在处理大规模数据集时。
```py
zarr.array(data=None, shape=None, chunks=None, dtype=None, compressor=None, fill_value=None, order='C', store=None, overwrite=False, path=None, chunk_store=None, synchronizer=None, cachesize=None, cache_type=None, **kwargs)
```

```py
import zarr

# 创建一个形状为 (100, 100) 的数组，第一维度块大小为 20，第二维度块大小为 10
array_custom_chunks = zarr.zeros((100, 100), chunks=(20, 10), dtype='f8')

# 打印数组信息
print("Array shape:", array_custom_chunks.shape)
print("Chunks shape:", array_custom_chunks.chunks)
# Array shape: (100, 100)
# Chunks shape: (20, 10)
```

### 持久化
```py
z1 = zarr.create_array(store='data/example-1.zarr', shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
z1[:] = 42
z1[0, :] = np.arange(10000)
z1[:, 0] = np.arange(10000)
```

创建新的持久化的 array，不用 flush，数据自动落盘。比如打开验证：
```py
z2 = zarr.open_array('data/example-1.zarr', mode='r')
np.all(z1[:] == z2[:])
```

直接保存 NumPy arrays：
```py
a = np.arange(10)
zarr.save('data/example-2.zarr', a)
zarr.load('data/example-2.zarr')
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

## zarr.Group
以层次化的方式组织和管理 arrays。zarr.Group 也是保存在内存和磁盘上，或者其他存储系统。

### 创建
使用`zarr.group(store: zarr.storage.StoreLike | None = None, *, overwrite: bool = False, ...)`。

```py
import zarr

store = zarr.storage.MemoryStore()
# 3.0 才有 zarr.create_group()，否则 zarr.group()
root = zarr.create_group(store=store)
r2 = zarr.group(store=store)
print(root, r2)
# <Group memory://140453554680576> <Group memory://140453554680576>
```

### 方法 values()

keys() 和 values() 方法可以获得 KeysView 和 ValuesView 对象。可以使用迭代器迭代并访问内容。KeysView 则对应各个 keys，字符串。ValuesView 则对应 Array 对象。

```py
# src 是一个 Group 实例，下面返回第一个保存的 Array
print(next(iter(src["meta"].values())))
```

### 层次化结构

root 代表当前 Group 最顶层，接下来的 Group 便可继续构造下一层。就像树的结构一样，Group 类比节点。
```py
foo = root.create_group("foo")
bar = foo.create_group("bar")
print(foo, bar)
# <Group memory://140453528214656/foo> <Group memory://140453528214656/foo/bar>
```
可以看到，Group 对象就像树的节点，就像文件系统，以`/`作为路径分隔，依附于上一次。比如 bar 依附在 foo。

#### 每层 Group 可以包含 arrays
```py
z1 = bar.create_array(
    name="baz", shape=(10000, 10000), chunks=(1000, 1000), dtype="int32"
)
z1
# <Array memory://140453528214656/foo/bar/baz shape=(10000, 10000) dtype=int32>
```

#### 上层 Group 可以访问其成员 Group
```py
root["foo"]
# <Group memory://140453528214656/foo>
```

使用 `/` 访问不同层次的成员和 arrays
```py
print(root["foo/bar"])
# <Group memory://140453528214656/foo/bar>
print(root["foo/bar/baz"])
# <Array memory://140453528214656/foo/bar/baz shape=(10000, 10000) dtype=int32>
```

#### zarr.Group.tree() 查看分层结构
```py
root.tree()
# /
# └── foo
#     └── bar
#         └── baz (10000, 10000) int32
```

#### zarr.open_group()
创建或重新打开 Group，其保存在文件系统，而子 Group 保存在对应子目录下。
```py
root = zarr.open_group("data/group.zarr", mode="w")
root
z = root.create_array(
    name="foo/bar/baz", shape=(10000, 10000), chunks=(1000, 1000), dtype="int32"
)
z
# <Array file://data/group.zarr/foo/bar/baz shape=(10000, 10000) dtype=int32>
```

最开始，执行的当前目录下，没有 data 目录。执行之后，本地出现了对应文件：
```sh
❯ tree data
data
└── group.zarr
    ├── foo
    │   ├── bar
    │   │   ├── baz
    │   │   │   └── zarr.json
    │   │   └── zarr.json
    │   └── zarr.json
    └── zarr.json

5 directories, 4 files
```

#### zarr.<Array|Group>.attrs 
Zarr arrays 和 groups 有 attrs，能够提供自定义的 key/value 属性。由于 Zarr 使用 JSON 存储 array 的 attributes，所以 value 必须是 JSON 可序列化的对象。

注意，子 Group 不算属性。

#### zarr.Group.keys()
返回一个生成器，遍历此 Group 对象的成员 Group 的 name。也可以直接使用`in`判断是否存在对应的 name：
```py
meta = root.require_group("meta", overwrite=False)
if "episode_ends" not in meta:
    episode_ends = meta.zeros("episode_ends", shape=(0,), dtype=np.int64, compressor=None, overwrite=False)
```

#### zarr.Group.store
获取内在维护的 zarr.store，比如 MemoryStore，ZipStore 等。

#### zarr.Group.require_group()
```py
require_group(name: str, **kwargs: Any) → Group
```
Obtain a sub-group, creating one if it doesn’t exist. 

参数`name`可以是绝对路径和相对路径。

参数默认`overwrite=False`代表如果存在，则返回；否则创建。`True`代表总会覆盖。

#### zarr.copy_store()
```py
zarr.copy_store(
    source,          # 源存储（zarr.Store 对象）
    dest,            # 目标存储（zarr.Store 对象）
    source_path="",  # 源数据集在存储中的路径（可选）
    dest_path="",    # 目标路径（可选）
    log=sys.stdout,  # 日志输出流（None 表示不输出）
    excludes=None,   # 排除复制的文件（如 ["group/.zgroup"]）
    includes=None,   # 指定仅复制的文件（与 excludes 互斥）
    dry_run=False    # 模拟运行（不实际复制）
)
```

其中，source_path 和 dest_path，接受字符串对象，从指定的 source 中复制 source_path，放置到 dest 的 dest_path 中。默认值 "" 则为 store 的根目录。用于控制拷贝。

## Storage

zarr.ZipStore 是存储后端，将数据保存于 zip 文件，适用数据归档、传输和临时存储场景。常见的还有 DirectoryStore，本地存储，用于大规模数据集和高性能计算，直接打开非压缩的本地文件。FSStore 适用于云存储和分布式存储。

```py
store = zarr.storage.ZipStore("data.zip", mode="w")
zarr.create_array(store=store, shape=(2,), dtype="float64")
# <Array zip://data.zip shape=(2,) dtype=float64>
```

MemoryStore 操作。
```py
data = {}
store = zarr.storage.MemoryStore(data)
# TODO: replace with create_array after #2463
zarr.create_array(store=store, shape=(2,), dtype="float64")
# <Array memory://140455700104128 shape=(2,) dtype=float64>
```

### ZipStore 例子

关于 ZipStore，可以使用 zarr.group 打开：
```py
# 打开 ZIP 文件并加载 Zarr 数据结构
zip_store = ZipStore('my_data.zip', mode='r')
root_loaded = zarr.open_group(zip_store, mode='r')

import zarr
from zarr import copy_store
from zarr.storage import MemoryStore

# 创建一个新的 MemoryStore
source_store = MemoryStore()

# 创建一个新的组
root_source = zarr.group(store=source_store)

# 在组中创建两个数组
array1 = root_source.zeros('array1', shape=(100, 100), chunks=(10, 10), dtype='f8')
array2 = root_source.zeros('array2', shape=(50, 50), chunks=(5, 5), dtype='i4')

# 修改数组中的某些值
array1[0:10, 0:10] = 1.0
array2[0:5, 0:5] = 2

# 创建另一个 MemoryStore 并从源 store 复制数据
target_store = MemoryStore()
copy_store(source_store, target_store)
# 从目标 store 加载数据
root_target = zarr.group(store=target_store)

# 打印组结构
print("\nTarget group structure:")
print(root_target.tree())
# /
#  ├── array1 (100, 100) float64
#  └── array2 (50, 50) int32

# 访问数组内容
array1_loaded = root_target['array1']
print("\nLoaded array1 content:")
print(array1_loaded[0:10, 0:10])

array2_loaded = root_target['array2']
print("\nLoaded array2 content:")
print(array2_loaded[0:5, 0:5])
```

## 异步
### AsyncGroup

## Tag
#Data