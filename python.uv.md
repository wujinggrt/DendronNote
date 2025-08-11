---
id: tmruqcalaaqsqzkj35nehsq
title: Uv
desc: ''
updated: 1753606664505
created: 1737822755943
---

conda 下载速度慢，推荐用 conda 创建环境，不要用它下载包。

通过安装脚本安装 uv，或者直接用 pip 安装。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
pip install uv
```

```bash
uv --help
```

## Ref
uv-速度飞快的pip替代 - 王云峰的文章 - 知乎
https://zhuanlan.zhihu.com/p/689976933

## 创建虚拟环境

类似 Python 的 venv，在当前目录或指定目录添加虚拟环境：

```sh
# 在当前目录创建 myenv 目录，作为环境
# 指定Python版本，注意需要对应版本的Python已经安装
uv venv myenv -p 3.12
# 激活类似于 venv 与 conda
source myenv/bin/activate
```

uv 会自动下载 Python 包，所以上述命令只能够在当前 Python 3.12 版本起作用，否则不行。

为了区分，通常在项目目录下创建虚拟环境，并且使用默认目录 `.venv`。

## 用法

### 安装

#### 宽松增量安装：uv pip install

与使用 pip 高度一致，只需要加上 uv 即可。比如：
```sh
# 从镜像网站上拉取安装包
uv pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
# 更新包版本，并安装特定版本的包：
uv pip install -U flask==3.0.0 # --upgrade
# 从当前目录安装，并且支持editable实时更新代码模式
uv pip install -e .
uv pip uninstall flask
```

注意，包名加上方括号通常代表安装包的额外依赖（extras）。

pip install 的 -f，--find-links 参数，uv pip 使用 --find-links。

```bash
# pip3 install -f / --find-links	--find-links ...
uv pip install --find-links {{url}}
```

安装的包会放到 .venv/lib/python3.12/site-packages 下。

`uv pip install --no-build-isolation` 表示​​禁用构建隔离环境​​。默认情况下，pip 在安装包时会创建一个临时的、隔离的虚拟环境（即“构建隔离”），仅在该环境中安装构建所需的依赖项（如 setuptools、wheel 等），以避免与用户全局环境的依赖发生冲突。使用此参数后，pip 将直接使用当前 Python 环境中已安装的依赖进行构建，跳过隔离环境的创建。

#### 声明式依赖管理

安装并声明 pyproject.toml，记录安装的环境。最后生成锁文件。

```bash
uv add {{包名}}
uv pip compile -o uv.lock
```

#### 严格同步安装

如果需要严格同步要求的包，需要使用：

```bash
uv sync
```

必须基于 uv.lock 或 requirements.txt 等锁文件执行，确保环境与声明完全一致。

#### 解决 uv.lock 中的已有包冲突

1.  环境中已有 `numpy==1.24`，但 `uv.lock` 要求 `numpy==1.26`
    - `uv pip install`：保留 1.24（可能引发冲突）
    - `uv add numpy==1.26`：更新声明并安装 1.26
    - `uv sync`：强制降级到 1.26（移除 1.24）



### 创建工程

像 cargo init 一样初始化项目，`--no-workspace` 代表不创建工作区结构。如果不指定，则会：

```bash
uv init --no-workspace
```

```
.
├── hello.py
├── pyproject.toml
└── README.md
```

添加相关的库：

```bash
uv add mcp "mcp[cli]"
```

类似 uv pip install，安装 mcp 包和依赖。随后会在 pyproject.toml 中声明依赖，组织在 project.optional-dependencies 数组中。

### 运行：uv run

无需手动激活虚拟环境运行。可以运行脚本：

```bash
uv run greet.py Hello Python  # 输出：Hello Python :cite[1]
echo 'print("临时脚本")' | uv run -       # 输出：临时脚本
uv run - <<EOF
print("多行脚本")
EOF                                   # 输出：多行脚本 :cite[1]echo 'print("临时脚本")' | uv run -       # 输出：临时脚本
```

若当前目录存在 .venv 虚拟环境，使用它，且不会自动安装依赖，所以需要提前 uv sync。不存在则自动创建临时虚拟环境，放到系统缓存目录，执行完成后自动清理。

参数：
- `--venv .venv` 强制使用环境
- `--with debugpy` 临时添加依赖。可以配合 --venv 使用

结合 uv add 管理依赖声明 + uv run 执行，可以高效率开发。

## 最佳实践

一般在一个工程根目录下，创建一个新的虚拟环境。即：

```bash
git clone URI
cd DIR
uv venv
source .ven/bin/activate
```

## Tag
#Python