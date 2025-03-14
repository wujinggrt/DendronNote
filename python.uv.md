---
id: tmruqcalaaqsqzkj35nehsq
title: Uv
desc: ''
updated: 1741916449703
created: 1737822755943
---

conda 下载速度慢，推荐用 conda 创建环境，不要用它下载包。

pip install uv

uv --help

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

uv 不会自动下载 Python 包，所以上述命令只能够在当前 Python 3.12 版本起作用，否则不行。

## 用法
与使用 pip 高度一致，只需要加上 uv 即可。比如：
```sh
# 从镜像网站上拉取安装包
uv pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
# 更新包版本，并安装特定版本的包：
uv pip install -U flask==3.0.0
# 从当前目录安装，并且支持editable实时更新代码模式
uv pip install -e .
uv pip uninstall flask
```

## 最佳实践

一般在一个工程根目录下，创建一个新的虚拟环境。即：

```bash
git clone URI
cd DIR
uv ven
source .ven/bin/activate
```

## Tag
#Python