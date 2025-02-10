---
id: aw6qydd9das3cs0ltxnu8qq
title: Fastapi
desc: ''
updated: 1737706410259
created: 1737472009426
---

[links](https://zhuanlan.zhihu.com/p/706632683)

#### 第一个应用
一般使用 uvicorn。

```py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def index():
    return {"msg": "Hello World!"}

@app.get("/items/{item_id}")
async def get_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```