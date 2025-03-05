---
id: 745l1jwrarwygpelwedsioj
title: 模块和包
desc: ''
updated: 1741140057437
created: 1740714013506
---

## 包 (Package)

### `__init__.py`

在较早版本的 Python 中（3.3 之前），`__init__.py` 文件是必需的，以使一个目录被视为一个包。从 Python 3.3 开始，引入了“命名空间包”概念，即使没有 `__init__.py` 文件，目录也可以被识别为包，但 `__init__.py` 仍然广泛使用，因为它提供了额外的功能。

当你导入一个包时，`__init__.py` 文件中的代码会被执行。这使得你可以在包加载时进行初始化操作，例如设置全局变量、导入子模块等。可以在 `__init__.py` 文件中定义便捷的导入路径，使得用户可以更方便地访问包内的模块或函数。

通过定义 `__all__` 列表，你可以控制 from package import * 语句导入哪些子模块或属性。

## 本地安装包

## 缓存 wheels 文件的目录

在 ~/.cache/<pip 或 uv> 目录下，保留了包的缓存。有时候特别大，可以手动删除此目录，也可以手动清理，比如 pip cache purge 或 uv cache prune。

修改缓存目录：

```bash
pip config set global.cache-dir "/path/to/your/custom/cache"
```

这会在 ~/.config/pip/pip.conf 或者 ~/.pip/pip.conf 文件添加或更新对应项。

对于 uv，修改环境变量配置：

```bash
export XDG_CACHE_HOME="/path/to/your/custom/cache"
```

## Ref and Tag