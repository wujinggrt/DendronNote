---
id: 7cd9he4w15xq7jbt88b3xp4
title: Zarr
desc: ''
updated: 1739878731161
created: 1739872667023
---

Zarr是一种存储分组、压缩的N维数组的格式。

官方教程：https://zarr.readthedocs.io/en/stable/tutorial.html

## zarr.Group
允许用户以层次化的方式组织和管理数据集。zarr.Group 像文件系统一样组织数据，包含子组和数组。

### 创建
使用`zarr.group(store: zarr.storage.StoreLike | None = None, *, overwrite: bool = False, ...)`。

```py
import zarr
store = zarr.storage.MemoryStore()
root = zarr.create_group(store=store) # <Group memory://...>
```

## ZipStore 和 MemoryStore 等用法
zarr.ZipStore 是存储后端，将数据保存于 zip 文件，适用数据归档、传输和临时存储场景。常见的还有 DirectoryStore，用于大规模数据集和高性能计算。FSStore 适用于云存储和分布式存储。关于 ZipStore，可以使用 zarr.group 打开：
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

## Tag
#Data