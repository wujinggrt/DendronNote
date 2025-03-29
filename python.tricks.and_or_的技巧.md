---
id: kg25u98px0acpx0274z3wqp
title: And_or_的技巧
desc: ''
updated: 1743186179171
created: 1742976792797
---

除了类似三目运算符的 res0 if cond else res1，还可以使用 and or 简化 if 的长度。当然，可读性会变差。主要在于根据 and or 的表达式是否代表 True，返回对应表达式的结果，而非 True/False。

使用 and 和 or 时，类似 bash 的 && || 等短路执行一样，会返回能够判断表达式为 True 或 False 的对象或元素。比如 `1 and a or b`，中，1 代表 True，继续下一步判断。如果 a 为 True，则返回 a，后续不再判断了。若 a 为 False，还要继续判断 b，最后得到 b 的值。

(1 and [a] or [b])[0]，注意，即使可以保证 `[a]` 为 True，已经是一个 list 对象了。

and 和 or 中，True 和 False 判断如：0, 引号, 大中小括号, None 会代表 False，其余情况为 True。

```py
>>> 0 and 1
0
>>> 0 or 1
1
>>> '' and 'A'
''
>>> '' or 'A'
'A'
>>> () and 1
()
>>> () or 1
1
>>> 1 and 2 # 一直在找有没有 False，走到了 2 后停止
2
>>> 1 and 2 and 3 and 0
0
>>> False or "" or () # 一直判断找有没有 True 的，走到 ()，然后停止了
()
>>> False or "" or 1 or ()
1
```



## Ref and Tag