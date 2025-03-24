---
id: ofq00xwu5htfle9n5fxrx2f
title: If_判断条件细节
desc: ''
updated: 1742785136122
created: 1742784691859
---

## list 实例作为判断

```py
lst = []
if lst:
    print("True")
else:
    print("False")
False
```

## 普通 class 的实例

```py
class A(BaseModel):
    i: int
a = A(i=3)
if a:
    print("True")
else:
    print("False")
True
```

## Ref and Tag