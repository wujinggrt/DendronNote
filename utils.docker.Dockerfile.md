---
id: amykfzj12rgwy4jphkw8abd
title: Dockerfile
desc: ''
updated: 1745810535811
created: 1745783052655
---

## COPY

如果需要复制到目录下，最后一定要指定反斜杠 `/`，否则会重命名为路径的最后一项。路径不存在的话，会自动创建。

```Dockerfile
COPY hom* /mydir/
COPY hom?.txt /mydir/
```

## Tips

RUN 指令中默认使用的是 /bin/sh，而 source 是 bash/zsh 等 shell 的命令，不是标准的 POSIX shell 命令。最佳实践：运行脚本时，使用 `.` 而非 source。或者显示使用 bash，比如：

```Dockerfile
RUN bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh \
    && source /root/.local/bin/env"
```

注意，在 docker build . -t {{images}} 阶段，还没有打开终端，所以 `~` 并不会展开为具体用户的目录，`source ~/.local/bin/env` 不会正确解析到 /root/.local/bin/env，即使将 `~` 替换为 `$HOME` 也不行。最终提示错误 /bin/sh: 1: source: not found。

解决方案：指定绝对路径，比如 `source /root/.local/bin/env`，这是推荐的；或者使用 `USER root` 指定用户，提供可以展开内容的上下文。

## Ref and Tag