---
id: blk1s4zbfhd016wg0ppuatd
title: Asyncio_异步
desc: ''
updated: 1742838989794
created: 1742836619998
---

asyncio 往往是构建 IO 密集型和高层级 结构化 网络代码的最佳选择。在 LLM 纪元，生成 tokens 需要耗时，可能也是一个优解。注意，一旦引入 async，整个代码都会大量引入 async。

高级层 API 提供：
- 并发地 运行 Python 协程 并对其执行过程实现完全控制;
- 执行 网络 IO 和 IPC;
- 控制 子进程;
- 通过 队列 实现分布式任务;
- 同步 并发代码;

低层级 API 以支持 库和框架的开发者 实现:
- 创建和管理 事件循环，它提供用于 连接网络, 运行 子进程, 处理 OS 信号 等功能的异步 API；
- 使用 transports 实现高效率协议;
- 通过 async/await 语法 桥接 基于回调的库和代码。

## 协程与任务

### 协程

```py
async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)

async def main():
    print(f"started at {time.strftime('%X')}")

    await say_after(1, 'hello')
    await say_after(2, 'world')

    print(f"finished at {time.strftime('%X')}")

asyncio.run(main())
```

输出：

```
started at 17:13:52
hello
world
finished at 17:13:55
```

async asyncio.sleep(delay, result=None) 阻塞 delay 指定的秒数。如果指定 result，则返回给调用者。将 delay 设为 0 将提供一个经优化的路径以允许其他任务运行。 这可供长期间运行的函数使用以避免在函数调用的全过程中阻塞事件循环。

不能简单调用 main()，只会得到协程对象。需要 asyncio.run() 执行。上述代码可以看到，没有并发地调度，甚至是串行的。但是调用 async 的 say_after() 函数时，使用了 await。使用 asyncio.gather() 时，可以并发地加入计划任务，在 await 处调度。

并发地运行 asyncio 任务的多协程：

```py
async def main():
    task1 = asyncio.create_task(
        say_after(1, 'hello'))
    task2 = asyncio.create_task(
        say_after(2, 'world'))
    print(f"started at {time.strftime('%X')}")
    # 等待直到两个任务都完成（会花费约 2 秒钟。）
    await task1
    await task2
    print(f"finished at {time.strftime('%X')}")
```

期待的输出：

```console
started at 17:14:32
hello
world
finished at 17:14:34
```

耗时更短了，有了并行的感觉，耗时更短。更现代的写法：

```py
async def main():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(
            say_after(1, 'hello'))

        task2 = tg.create_task(
            say_after(2, 'world'))

        print(f"started at {time.strftime('%X')}")
    # 当存在上下文管理器时 await 是隐式执行的。
    print(f"finished at {time.strftime('%X')}")
```

### 可等待对象

协程是可等待的对象，可以在其他协程等待：

```py
async def nested():
    return 42

async def main():
    # 如果我们只调用 "nested()" 则无事发生。
    # 一个协程对象会被创建但是不会被等待，
    # 因此它 *根本不会运行*。
    nested()  # 将引发 "RuntimeWarning"。
    # 现在让我们改为等待它：
    print(await nested())  # 将打印 "42"。
asyncio.run(main())
```

#### 任务

使用任务来“并行地”调度。协程通过 asyncio.create_task() 或等价的方式封装为任务时，协程会被自动调度。

#### Futures

是低级的可等待对象，表示异步操作的最终结果。当一个 Future 对象被等待时，协程保持等待到 Future 对象（在其他协程）操作完毕。asyncio 中，需要 Future 对象以便允许通过 async/await 使用基于回调的代码。通常情况下 没有必要 在应用层级的代码中创建 Future 对象。

### 创建任务

asyncio.create_task(coro, *, name=None, context=None) 封装协程为 Task，并调度其执行。返回 Task 对象。

### 并发运行任务

awaitable asyncio.gather(*aws, return_exceptions=False)

```py
import asyncio

async def factorial(name, number):
    f = 1
    for i in range(2, number + 1):
        print(f"Task {name}: Compute factorial({number}), currently i={i}...")
        await asyncio.sleep(1)
        f *= i
    print(f"Task {name}: factorial({number}) = {f}")
    return f

async def main():
    # 将三个调用 *并发地* 加入计划任务：
    L = await asyncio.gather(
        factorial("A", 2),
        factorial("B", 3),
        factorial("C", 4),
    )
    print(L)

asyncio.run(main())

# 预期的输出：
#
#     Task A: Compute factorial(2), currently i=2...
#     Task B: Compute factorial(3), currently i=2...
#     Task C: Compute factorial(4), currently i=2...
#     Task A: factorial(2) = 2
#     Task B: Compute factorial(3), currently i=3...
#     Task C: Compute factorial(4), currently i=3...
#     Task B: factorial(3) = 6
#     Task C: Compute factorial(4), currently i=4...
#     Task C: factorial(4) = 24
#     [2, 6, 24]
```

### async with 异步上下文管理器

一个类支持上下文管理，需要实现 `__aenter__` 和 `__aexit__` 方法。用于初始化资源和清理资源。

`@asynccontextmanager` 是一个异步上下文的注释神器，在标准库 contextlib 中，将一个异步函数转换为上下文管理器。核心作用是管理异步资源的生命周期，确保资源在使用完毕后能够被正确释放。

```py
from contextlib import asynccontextmanager

@asynccontextmanager
async def my_async_context():
    # 初始化逻辑
    resource = await initialize_resource()
    try:
        yield resource  # 返回资源给调用方
    finally:
        # 清理逻辑
        await cleanup_resource(resource)
```

### Timeout：超时

```py
asyncio.timeout(delay)
```

返回一个可被用于限制等待某个操作所耗费时间的 异步上下文管理器。

```py
async def main():
    try:
        async with asyncio.timeout(10):
            await long_running_task()
    except TimeoutError:
        print("The long operation timed out, but we've handled it.")

    print("This statement will run regardless.")
```

class asyncio.Timeout(when) 撤销过期协程的异步上下文管理器。
- when() → float | None 返回当前终止点，或者如果未设置当前终止点则返回 None。
- reschedule(when: float | None) 重新安排超时。
- expired() → bool

```py
async def main():
    try:
        # 当开始时我们并不知道超时值，所以我们传入 ``None``。
        async with asyncio.timeout(None) as cm:
            # 现在我们知道超时值了，所以我们将它重新加入计划任务。
            new_deadline = get_running_loop().time() + 10
            cm.reschedule(new_deadline)

            await long_running_task()
    except TimeoutError:
        pass

    if cm.expired():
        print("Looks like we haven't finished on time.")
```

还有 asyncio.timeout_at(when)。

### 等待

async asyncio.wait_for(aw, timeout)

```py
async def eternity():
    # 休眠一小时
    await asyncio.sleep(3600)
    print('yay!')

async def main():
    # 等待至多 1 秒
    try:
        await asyncio.wait_for(eternity(), timeout=1.0)
    except TimeoutError:
        print('timeout!')

asyncio.run(main())

# 预期的输出：
#
#     timeout!
```

### 在线程中运行

async asyncio.to_thread(func, /, *args, **kwargs)

```py
def blocking_io():
    print(f"start blocking_io at {time.strftime('%X')}")
    # 请注意 time.sleep() 可被替换为任意一种
    # 阻塞式 IO 密集型操作，例如文件操作。
    time.sleep(1)
    print(f"blocking_io complete at {time.strftime('%X')}")

async def main():
    print(f"started main at {time.strftime('%X')}")

    await asyncio.gather(
        asyncio.to_thread(blocking_io),
        asyncio.sleep(1))

    print(f"finished main at {time.strftime('%X')}")


asyncio.run(main())

# 预期的输出：
#
# started main at 19:50:53
# start blocking_io at 19:50:53
# blocking_io complete at 19:50:54
# finished main at 19:50:54
```

## Ref and Tag

https://docs.python.org/zh-cn/3/library/asyncio.html