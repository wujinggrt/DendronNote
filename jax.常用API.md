---
id: wj3lx9g1hyku61mua06rj23
title: 常用API
desc: ''
updated: 1753810290580
created: 1753625581783
---

## jnp 参考 numpy

einsum:

```py
encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
```

where: True 返回 x 的部分，False 则 y。可以广播。

```py
numpy.where(condition, [x, y, ]/)
np.where(a < 4, a, -1)
```

dot：

```py
jnp.dot(a, b)
```

stack 会创建新的轴，concatenate 则修改既有的轴。

## jax.nn

softmax：

```py
probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
```

## 类型

```py
image_spec = jax.ShapeDtypeStruct(
    [batch_size, *_model.IMAGE_RESOLUTION, 3], 
    jnp.float32)
```

## Ref and Tag

https://docs.jax.dev/en/latest/_autosummary/jax.nn.softmax.html