---
id: amykfzj12rgwy4jphkw8abd
title: Dockerfile
desc: ''
updated: 1745783139901
created: 1745783052655
---

## COPY

如果需要复制到目录下，最后一定要指定反斜杠 `/`，否则会重命名为路径的最后一项。路径不存在的话，会自动创建。

```Dockerfile
COPY hom* /mydir/
COPY hom?.txt /mydir/
```

## Ref and Tag