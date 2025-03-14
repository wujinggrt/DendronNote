---
id: 745l1jwrarwygpelwedsioj
title: 模块_包和安装
desc: ''
updated: 1741940096184
created: 1740714013506
---

## 包 (Package)

### `__init__.py`

在较早版本的 Python 中（3.3 之前），`__init__.py` 文件是必需的，以使一个目录被视为一个包。从 Python 3.3 开始，引入了“命名空间包”概念，即使没有 `__init__.py` 文件，目录也可以被识别为包，但 `__init__.py` 仍然广泛使用，因为它提供了额外的功能。

当你导入一个包时，`__init__.py` 文件中的代码会被执行。这使得你可以在包加载时进行初始化操作，例如设置全局变量、导入子模块等。可以在 `__init__.py` 文件中定义便捷的导入路径，使得用户可以更方便地访问包内的模块或函数。

通过定义 `__all__` 列表，你可以控制 from package import * 语句导入哪些子模块或属性。

## pip

### 设置镜像

可以每次设置环境变量，每次指出即可。

```bash
export THU_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
pip install -i $THU_MIRROR ...
```

设置全局：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 使用多个镜像源
pip config set global.index-url "<url1> <url2>..."
# 查看镜像
cat /home/wujing/.config/pip/pip.conf
# 验证
pip config list
# 清空，使用默认
pip config unset global.index-url
```

### 本地安装包： pip install -e . 会发生什么

-e 是 --editable，以可编辑模式（开发模式）安装包。有以下影响和作用：
- 包的代码不会复制到 Python 的 site-packages 目录。
- 直接在当前目录开发和修改代码，在当前目录修改代码，会对使用此包的代码可见，立即生效，不用重新安装。
- 在 site-packages 中，仅创建一个指向当前目录的链接文件，比如 .egg-link 或 .pth 文件。

`.` 代表当前目录，pip 会查找当前目录下的 setup.py 或 pyproject.toml，根据要求完成配置安装包。

比如：
```
my_package/
├── setup.py
├── my_package/
│   └── __init__.py
└── src/
    └── module.py
```

在 my_package 目录执行 pip install -e . 后，有如下影响：
- my_package 目录下创建 my_package.egg-info 目录，包含了元数据。
- 在 site-packages 中生成一个 .egg-link 文件，内容指向 my_package 的绝对路径。
- 直接修改 my_package/__init__.py 或 src/module.py 会立即生效。

比如，在 [legged gym](https://github.com/leggedrobotics/legged_gym) 中，在 isaacgym, rsl-rl 和 legged_gym 中执行了 pip install -e . 之后，可以看到项目目录生成了 legged_gym.egg-info 和 site-packages 目录下生成了对应名字的 .egg-link：
```bash
/data1/wj_24/projects/legged_gym
❯ ls
legged_gym  legged_gym.egg-info  LICENSE  licenses  README.md  resources  setup.py
❯ find /data1/wj_24/miniforge3/envs/leggedgym/ -regex '.*\.egg-link'
/data1/wj_24/miniforge3/envs/leggedgym/lib/python3.8/site-packages/isaacgym.egg-link
/data1/wj_24/miniforge3/envs/leggedgym/lib/python3.8/site-packages/legged-gym.egg-link
/data1/wj_24/miniforge3/envs/leggedgym/lib/python3.8/site-packages/rsl-rl.egg-link
```

卸载使用 pip uninstall PACKAGE_NAME 即可。

### 缓存 wheels 文件的目录

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