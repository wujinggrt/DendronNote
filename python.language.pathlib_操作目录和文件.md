---
id: oyt05lfqh26c699x2kvokbw
title: Pathlib_操作目录和文件
desc: ''
updated: 1744167817902
created: 1741030522182
---

## Path 类

操作通常由 Path 类完成。常用内容如下：

Path 对象字段和属性，即 descriptor:
- parts   : 目录的每一层路径，比如 /home/wj-24/.zshrc 会得到 ('/', 'home', 'wj-24', '.zshrc')
- parent  : 父目录
- parents : 所有父目录
- stem    : 不带后缀的文件名
- name    : 文件名或目录名
- suffix  : 文件名后缀
- suffixes: 文件名后缀列表

方法:
- is_absolute(): 是否为绝对路径
- joinpath()   : 组合路径
- cwd()        : 当前工作目录
- home()       : 根目录
- rename()     : 重命名
- replace()    : 覆盖
- touch()      : 新建文件
- exists()     : 是否存在路径
- expanduser() : 返回带 ~ 和 ~user 的路径
- glob()       : 返回生成器。列出匹配的文件或目录，仅当前目录。
- rglob()      : 返回生成器。递归列出匹配的文件或目录，包括所有子目录。使用 .rglob("*") 会递归地获取所有文件和目录。
- iterdir()    : 返回生成器。列出当前路径下的文件和目录。不包含 . 和 ..，不展开子目录。
- is_dir()     : 是否为目录
- is_file()    : 是否为文件
- mkdir()      : 新建目录
- open()       : 打开文件
- resolve()    : 获取对应绝对路径的 Path 实例
- rmdir()      : 删除目录
- ...

如果想要获取字符串的路径，使用 `str(p)` 类型转换即可。

创建
```py
manage_path = Path("manage.py").resolve()  # 绝对路径
base_dir = manage_path.parent  # 父目录
another_manage_path = base_dir / "another_manage.py"  # 构成新路径
```

创建和重命名

```py
Path("./src/stuff").mkdir(parents=True, exist_ok=True)  # 构建目录./src/stuff
Path("./src/stuff").rename("./src/config")  # 将./src/stuff重命名为./src/config
```

## 查看和操作目录

递归列出某类型文件

```py
top_level_py_files = Path(".").glob("*.py") # 不进行递归
all_py_files = Path(".").rglob("*.py")  # 递归

print(list(top_level_py_files))
print(list(all_py_files))
# ** 表示递归此目录及所有子目录。
# 当前目录下所有 Python 源码，递归地。
print(list(p.glob('**/*.py')))
```

列出当前目录下的子目录（非递归）

```py
p = Path('.')
[x for x in p.iterdir() if x.is_dir()]
[PosixPath('.hg'), PosixPath('docs'), PosixPath('dist'),
 PosixPath('__pycache__'), PosixPath('build')]
 ```

使用 iterdir() 统计当前目录下文件个数，不包含 "." 和 ".."。

```py
now_path = pathlib.Path.cwd()
gen = (i.suffix for i in now_path.iterdir())
print(Counter(gen))  # Counter({'.py': 16, '': 11, '.txt': 1, '.png': 1, '.csv': 1})
```

路径中，通常由 / 分开目录或文件，获取分开部分方式如下：

```py
file_path = Path("F:/spug-3.0/spug-3.0/spug_api/pathlib_test.py")
print(file_path.parts)
# ('F:\\', 'spug-3.0', 'spug-3.0', 'spug_api', 'pathlib_test.py')
```

## Ref and Tag

[doc pathlib](https://docs.python.org/zh-cn/3.11/library/pathlib.html)
[pathlib, 一个优雅的python库 - 海哥python的文章 - 知乎](https://zhuanlan.zhihu.com/p/670865534)