---
id: 7cd9he4w15xq7jbt88b3xp4
title: Zarr
desc: ''
updated: 1742665462432
created: 1739872667023
---


Zarr是一种存储分组、压缩的N维数组的格式。

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

此时，src 使用了 DirectoryStore。如果顶层是 Group，则是 Group。

注意，以 w 模式打开，会删除原文件上的所有内容。所以，以 a 模式打开更保险，可以追加或修改。

### 转换为 numpy.ndarry

直接对 zarr 的 Array 使用切片 [:] 访问，或者访问任意一个元素，便可获得 numpy.ndarray。

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

最佳实践，把 chunks 的 shape 选择为训练时，最常用的 (batch_size, data_shape[:])，而不是 (1, data_shape[:])。

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

### Resizing and appending

Zarr 数组可以 resize，其维度可以增加和减少，长度也是。内在的数据不会被重排。如果 resize 导致了缩小，在外部的数据会被丢弃。导致了扩充，则默认 0。

```py
store = zarr.storage.MemoryStore()
z = zarr.open_array(
    store="./data/example-1.zarr",
    shape=(10000, 10000),
    dtype="int32",
    chunks=(1000, 1000),
)
z[:] = 42
print(z[0, 0], z[-1, -1]) # 42 42
z.resize((20000, 10000))
print(z[0, 0], z[-1, -1]) # 42 0
z.resize((200, 100))
print(z[0, 0], z[-1, -1]) # 42 42
```

append(data: numpy.typing.ArrayLike, axis: int = 0) 方法则更方便，直接在任何轴上追加。比如：

```py
a = np.arange(10000000, dtype='int32').reshape(10000, 1000)
z = zarr.create_array(store='data/example-4.zarr', shape=a.shape, dtype=a.dtype, chunks=(1000, 100))
z[:] = a
z.shape
# (10000, 1000)
z.append(a)
# (20000, 1000)
z.append(np.vstack([a, a]), axis=1)
# (20000, 2000)
z.shape
# (20000, 2000)
```

### zarr.require_dataset()

用于安全创建或访问数据集。如果数据集不存在，按照给定参数创建并打开。如果存在，验证是否和指定参数的形状和数据类型保持一致。比如：

```py
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    for cam_id in range(n_cameras):
        name = f'camera{cam_id}_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['time'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )
```

out_replay_buffer.data 是一个 zarr.Group。随后可以安全使用 out_replay_buffer.data[f"camera{idx}_rgb"] 存入数据。

### 重命名

```py
group["new_name"] = group["old_name"]  # 将数据复制到新键即可，但是有开销
del group["old_name"]
```

替代方案：别名，在 group 中维护一个键名字典。

```py
group.attrs["alias"] = {"display_name": "old_name"}

# 通过映射访问数组
real_key = group.attrs["alias"]["display_name"]
array = group[real_key]
```

### 删除

```py
# 创建组并添加数组
group = zarr.open_group("data.zarr", mode="a")
group.create_dataset("my_array", shape=(2, 2), dtype=int)

# 删除数组
del group["my_array"]  # 或者 group.pop("my_array")

# 验证是否删除
print("my_array" in group)  # 输出 False
```

或直接删除对应目录下的子目录。

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

#### 每层 Group 可以包含和创建 arrays

```py
z1 = bar.create_array(
    name="baz", shape=(10000, 10000), chunks=(1000, 1000), dtype="int32"
)
z1
# <Array memory://140453528214656/foo/bar/baz shape=(10000, 10000) dtype=int32>
```

当然，我们也可以直接创建 array，简单的拷贝。chunks, dtype 等属性也会拷贝过来。

```py
bar["new_group/new_baz"] = bar["baz]
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

## 复制

### zarr.copy_store()

具体参考 site-packages/zarr/convenience.py 下的 copy_store() 和 copy() 函数的用法。copy_store() 函数可以更高效地复制 group 和 array，比 copy() 和 copy_all() 更快。

```py
zarr.copy_store(
    source,          # 源存储（zarr.Store 对象）
    dest,            # 目标存储（zarr.Store 对象）
    source_path="",  # 源数据集在存储中的路径（可选），仅拷贝原 store 的此路径下的数据
    dest_path="",    # 目标路径（可选），拷贝到此路径
    log=sys.stdout,  # 日志输出流（None 表示不输出）。拷贝到哪个 key，会打印，比如：copy xxx
    excludes=None,   # 排除复制的文件（如 ["group/.zgroup"]）。一个或多个正则字符串，对应的 key 不会从 source 拷贝。
    includes=None,   # 指定仅复制的文件（与 excludes 互斥）。一个或多个正则字符串，对应的 key 会拷贝，并且覆盖 excludes 选项。
    dry_run=False    # 模拟运行（不实际复制）
)
```

其中，source_path 和 dest_path，接受字符串对象。从指定的 source 中复制 source_path，放置到 dest 的 dest_path 中。默认值 "" 则为 store 的根目录。用于控制拷贝。

### zarr.copy()

h5py 的 group 或 dataset，可以与 zarr 的 group 或 dataset 互相拷贝。

```py
def copy(source, dest, name=None, shallow=False, without_attrs=False, log=None,
         if_exists='raise', dry_run=False, **create_kws)
    """Copy the `source` array or group into the `dest` group.

    Parameters
    ----------
    source : group or array/dataset
        A zarr group or array, or an h5py group or dataset.
    dest : group
        A zarr or h5py group.
    name : str, optional
        Name to copy the object to.
    shallow : bool, optional
        If True, only copy immediate children of `source`.
    without_attrs : bool, optional
        Do not copy user attributes.
    log : callable, file path or file-like object, optional
        If provided, will be used to log progress information.
    if_exists : {'raise', 'replace', 'skip', 'skip_initialized'}, optional
        How to handle arrays that already exist in the destination group. If
        'raise' then a CopyError is raised on the first array already present
        in the destination group. If 'replace' then any array will be
        replaced in the destination. If 'skip' then any existing arrays will
        not be copied. If 'skip_initialized' then any existing arrays with
        all chunks initialized will not be copied (not available when copying to
        h5py).
    dry_run : bool, optional
        If True, don't actually copy anything, just log what would have
        happened.
    **create_kws
        Passed through to the create_dataset method when copying an array/dataset.

    Returns
    -------
    n_copied : int
        Number of items copied.
    n_skipped : int
        Number of items skipped.
    n_bytes_copied : int
        Number of bytes of data that were actually copied.
    Here's an example of copying a group named 'foo' from an HDF5 file to a
    Zarr group::

        >>> import h5py
        >>> import zarr
        >>> import numpy as np
        >>> source = h5py.File('data/example.h5', mode='w')
        >>> foo = source.create_group('foo')
        >>> baz = foo.create_dataset('bar/baz', data=np.arange(100), chunks=(50,))
        >>> spam = source.create_dataset('spam', data=np.arange(100, 200), chunks=(30,))
        >>> zarr.tree(source)
        /
         ├── foo
         │   └── bar
         │       └── baz (100,) int64
         └── spam (100,) int64
        >>> dest = zarr.group()
        >>> from sys import stdout
        >>> zarr.copy(source['foo'], dest, log=stdout)
        copy /foo
        copy /foo/bar
        copy /foo/bar/baz (100,) int64
        all done: 3 copied, 0 skipped, 800 bytes copied
        (3, 0, 800)
        >>> dest.tree()  # N.B., no spam
        /
         └── foo
             └── bar
                 └── baz (100,) int64
        >>> source.close()

    The ``if_exists`` parameter provides options for how to handle pre-existing data in
    the destination. Here are some examples of these options, also using
    ``dry_run=True`` to find out what would happen without actually copying anything::

        >>> source = zarr.group()
        >>> dest = zarr.group()
        >>> baz = source.create_dataset('foo/bar/baz', data=np.arange(100))
        >>> spam = source.create_dataset('foo/spam', data=np.arange(1000))
        >>> existing_spam = dest.create_dataset('foo/spam', data=np.arange(1000))
        >>> from sys import stdout
        >>> try:
        ...     zarr.copy(source['foo'], dest, log=stdout, dry_run=True)
        ... except zarr.CopyError as e:
        ...     print(e)
        ...
        copy /foo
        copy /foo/bar
        copy /foo/bar/baz (100,) int64
        an object 'spam' already exists in destination '/foo'
        >>> zarr.copy(source['foo'], dest, log=stdout, if_exists='replace', dry_run=True)
        copy /foo
        copy /foo/bar
        copy /foo/bar/baz (100,) int64
        copy /foo/spam (1000,) int64
        dry run: 4 copied, 0 skipped
        (4, 0, 0)
        >>> zarr.copy(source['foo'], dest, log=stdout, if_exists='skip', dry_run=True)
        copy /foo
        copy /foo/bar
        copy /foo/bar/baz (100,) int64
        skip /foo/spam (1000,) int64
        dry run: 3 copied, 1 skipped
        (3, 1, 0)
        """
```

## Storage

zarr.ZipStore 是存储后端，将数据保存于 zip 文件，适用数据归档、传输和临时存储场景。常见的还有 DirectoryStore，本地存储，用于大规模数据集和高性能计算，直接打开非压缩的本地文件。FSStore 适用于云存储和分布式存储。

### ZipStore：压缩且持久存储

```py
store = zarr.storage.ZipStore("data.zip", mode="w")
zarr.create_array(store=store, shape=(2,), dtype="float64")
# <Array zip://data.zip shape=(2,) dtype=float64>
```

### MemoryStore：退出后丢失，用于临时场景

```py
data = {}
store = zarr.storage.MemoryStore(data)
# TODO: replace with create_array after #2463
zarr.create_array(store=store, shape=(2,), dtype="float64")
# <Array memory://140455700104128 shape=(2,) dtype=float64>
```

### DirectoryStore：本地持久化存储

```py
store = zarr.DirectoryStore('data.zarr')
root = zarr.group(store=store, overwrite=True)
# overwrite=True 代表如下的 "w"，会将所有内容清空。如果保留，仅做修改，可以使用修改模式 a
root = zarr.open(store=store, mode="a")
```

转储和拷贝的例子：

```py
def create_dexgraspvla_dataset_from_custom(src_path: str, dst_path: str) -> None:
    src = zarr.open(src_path, mode="r")

    store = zarr.storage.DirectoryStore(dst_path)
    # src = zarr.group(store=store, overwrite=True)
    target = zarr.open(store, mode="a")  # a for modification
    target["/meta/episode_ends"] = src["/meta/episode_ends"]
    target["/data/action"] = src["data/action"]
    target["/data/right_cam_img"] = src["data/hand"]
    target["/data/right_state"] = src["data/state"]

    data = target["data"]
    heads_shape = src["data/head"].shape
    rgbm = data.require_dataset(
        name="rgbm",
        shape=(heads_shape[0],) + heads_shape[1:3] + (4,),
        chunks=(1,) + heads_shape[1:3] + (4,),
        dtype=np.uint8,
    )
    rgbm[:] = np.concatenate(
        [
            src["data/head"][:],
            src["data/mask"][:],
        ],
        axis=-1,
    )
```

但是拷贝更推荐使用 zarr.copy_store()。

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

## 压缩保存和解压 RGB 图像

使用预处理的脚本得到 zarr 文件后，我想简单使用 zarr 库加载查看图像内容。data 下大部分内容能够正常加载。但是，camera0_rgb 加载缺出现了错误：ValueError: codec not available: 'imagecodecs_jpegxl'。这是因为压缩格式的问题。加载的代码如下：

```py
store = zarr.ZipStore("/data1/cola_big/dataset.zarr.zip", mode="r")
root = zarr.open_group(store, mode="r")
data = root["data"]
print(data["camera0_rgb"]) # ValueErro: codec not available: 'imagecodecs_jpegxl'r
```

根据报错，需要 jpegxl。根据官网的 [issue](https://github.com/cgohlke/imagecodecs/issues/82)，应当注册 codec 为 numcodecs/zarr，比如：

```py
 >>> import zarr 
 >>> import numcodecs 
 >>> from imagecodecs.numcodecs import Jpeg2k 
 >>> numcodecs.register_codec(Jpeg2k) # 关键在于此句
 >>> zarr.zeros( 
 ...     (4, 5, 512, 512, 3), 
 ...     chunks=(1, 1, 256, 256, 3), 
 ...     dtype='u1', 
 ...     compressor=Jpeg2k() 
 ... ) 
 ```

关键在于 numcodecs.register_codec(Jpeg2k)，Jpegxl 同理。注册后，zarr 可以在数据中找到对应的 compressor。

### 压缩图片以保存


查看 camera0_rgb 如何制作和压缩。在 scripts_slam_pipeline/07_generate_replay_buffer.py 下，使用 JpegXL 的实例作为 compressor，保存图像数据。

```py
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    for cam_id in range(n_cameras):
        name = f'camera{cam_id}_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['time'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )
```

根据 chunks，可以看到按照图片分块和时间步分块存储。第一维 time 作为时间步的，第二维 out_res 代表分辨率，第三维 (3,) 代表三通道。随后使用 compressor 压缩。

压缩保存后，查看训练时如何加载相机的图像数据，特别是解压方面，这对正确使用压缩后的数据有着参考作用。

## 异步
### AsyncGroup

## Tag
#Data

官方教程：https://zarr.readthedocs.io/en/stable/tutorial.html