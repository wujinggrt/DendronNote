---
id: s3dq7ejfno2ga3rm01hnpa6
title: 多进程_multiprocess
desc: ''
updated: 1745692328359
created: 1745688369812
---

## 多进程包 subprocess

### 简单场景: check_output，此 API 较老（不推荐）

```python
subprocess.check_output(
    args,
    *,
    stdin=None,
    stderr=None,
    shell=False,
    cwd=None,
    encoding=None,
    errors=None,
    universal_newlines=None,
    timeout=None,
    text=None,
    **other_popen_kwargs,
)
```

**参数**:
- `args`: 要执行的命令，可以是一个**字符串**或**字符串列表**。
    - shell=True: args 为字符串，更方便
    - shell=False: args 为列表，推荐，更安全
- `stdin`: 标准输入，默认为 None。
- `stderr`: 标准错误输出，默认为 None。可以设置为 subprocess.STDOUT 以将标准错误输出合并到标准输出。
- `shell`: 是否通过 shell 执行命令，默认为 False。如果为 True，则通过系统的 shell 执行命令。
- `cwd`: 设置命令执行的工作目录。
- `encoding`: 输出的编码格式，默认为 None。如果指定，则返回字符串而不是字节。
- `errors`: 编码错误处理方式，默认为 None。
- `universal_newlines`: 是否将输出中的换行符转换为 \n，默认为 None。已弃用，建议使用 text。
- `timeout`: 命令执行的超时时间（秒）。如果超时，会抛出 subprocess.TimeoutExpired 异常。
- `text`: 如果为 True，则返回字符串而不是字节。等同于 universal_newlines=True。
- `**other_popen_kwargs`: 其他传递给 subprocess.Popen 的参数。

**返回**：返回命令的标准输出（stdout）内容。如果 encoding 或 text 参数指定，则返回字符串；否则返回字节。

**异常**：
- `subprocess.CalledProcessError`: 命令返回非零退出状态
- `subprocess.TimeoutExpired`: 命令执行超时

```py
output = subprocess.check_output("sleep 1 && echo Hello, World!", shell=True)
print(output.decode("utf-8"))  # 输出: Hello, World!
```

注意，check_output 是**同步**的，会阻塞。

## 异步子进程：使用 asyncio.create_subprocess_exec

```py
import asyncio

async def async_check_output(*args, **kwargs):
    # 创建子进程
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs
    )
    
    # 异步读取 stdout 和 stderr
    # 等待子进程结束，此协程调度，让出权限
    stdout, stderr = await process.communicate()
    
    # 检查返回码
    if process.returncode != 0:
        error = stderr.decode().strip()
        raise subprocess.CalledProcessError(
            process.returncode, args, stdout, stderr, error
        )
    
    return stdout

async def main():
    try:
        output = await async_check_output("ls", "-l")
        print(output.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# 运行异步代码
asyncio.run(main())
```

等待子进程结束的方式有：

```py
stdout, stderr = await process.communicate()
await process.wait()
```

## subprocess.run

```py
subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, shell=False, timeout=None, check=False, encoding=None, errors=None)
```

参数：
- `args`：表示要执行的命令，可以是一个字符串（如果 shell=True）或一个列表（推荐）。例如：
    - shell=True: 字符串形式（需要 shell=True，但是需要注意注入攻击）：'ls -l'
    - shell=False: 列表形式：['ls', '-l']
- `stdin、stdout 和 stderr`：子进程的标准输入、输出和错误。其值可以是:
    - `subprocess.PIPE`：创建一个管道，用于子进程的输入和输出。
    - `subprocess.DEVNULL`：将子进程的输出重定向到 os.devnull。
    - `None`：不重定向。
- `timeout`：设置命令超时时间。如果命令执行时间超时，子进程将被杀死，并弹出 TimeoutExpired 异常。
- `check`：如果该参数设置为 True，并且进程退出状态码不是 0，则触发CalledProcessError异常。
- `encoding`：如果指定了该参数，则 stdin、stdout 和 stderr 可以接收字符串数据，并以该编码方式编码。否则只接收 bytes 类型的数据。
- `shell`：如果该参数为 True，将通过操作系统的 shell 执行指定的命令。
- `text`：布尔值，默认为 False。如果为 True，表示以文本模式处理输入和输出（相当于 universal_newlines=True）。
    
返回：一个 `CompletedProcess` 对象。可以通过以下属性处理输入、输出和错误：
- `stdout`：子进程的输出。
- `stderr`：子进程的错误输出。
- `returncode`：子进程的退出状态码。

`run()` 是一个同步函数，会阻塞直到命令执行完成。

典型场景​​：
- 执行简单命令，无需实时交互。
- 需要直接获取命令的输出或错误信息。
- 希望代码简洁、易读。

```py
import subprocess

result = subprocess.run(
    ["ls", "-l"],        # 命令参数（推荐列表形式，避免 shell 注入风险）
    capture_output=True, # 捕获 stdout/stderr
    text=True,           # 输出转为字符串（否则是字节流）
    check=True           # 非零返回码时抛出异常
)
print(result.stdout)
```

## 更精细的控制：subprocess.Popen

特点​​：
- ​​底层接口​​：提供更精细的控制（如实时交互、后台执行）。
- ​​非阻塞​​：可以启动进程后继续执行其他代码。
- ​​手动管理​​：需要自行处理输入/输出管道（如 stdout.read()）。
- ​​灵活性​​：支持流式读取输出、动态发送输入、超时控制等。

​​典型场景​​：
- 需要实时读取子进程的输出（例如逐行处理日志）。
- 需要与子进程交互（如发送输入并获取响应）。
- 启动后台进程并异步管理。

构造器：

```py
class subprocess.Popen(args, bufsize=-1, executable=None, stdin=None, stdout=None, stderr=None, preexec_fn=None, close_fds=True, shell=False, cwd=None, env=None, universal_newlines=None, startupinfo=None, creationflags=0, restore_signals=True, start_new_session=False, pass_fds=(), *, group=None, extra_groups=None, user=None, umask=-1, encoding=None, errors=None, text=None, pipesize=-1, process_group=None)
```

如果 stdin、stdout 和 stderr 参数设置为 PIPE，则 communicate() 方法会自动处理输入和输出。

方法：
- `communicate(input=None, timeout=None)`：发送数据到子进程的 stdin，返回 (stdout, stderr)，分别是字节流。会阻塞到子进程结束。
- `poll(timeout=None)`：检查子进程是否已经结束，返回状态码或 None。
- `wait(timeout=None)`：等待指定秒数，子进程结束返回状态码，否则抛出 TimeoutExpired 异常。None 代表阻塞。
- `send_signal(signal)`：向子进程发送信号。
- `terminate()`：终止子进程。
- `kill()`：强制终止子进程。

Popen 实例的常用属性：
- `stdin`: 当构造器传入 stdin=PIPE，此属性是函数 `open()` 返回的可写的流对象，可以使用 write(), flush() 等方法。是 text 还是 bytes 流取决如下：如果构造器传入 encoding 指定了比如 "utf-8"，而非默认的 None；或指定了 errors；或 text=True；或 universal_newlines=True，stdin 是文本流；否则是字节流
- `stdout`: 当构造器传入 stdout=PIPE，此属性是函数 `open()` 返回的可读的流对象。是 text 还是 bytes 流，规则同 `stdin`。可以使用 str() 类型转换获取内容，或 read() 方法获取。
- `stderr`: 当构造器传入 stdout=PIPE，此属性是函数 `open()` 返回的可读的流对象。是 text 还是 bytes 流，规则同 `stdin`
- `pid`: 子进程的进程 ID。
- `returncode`: 子进程的退出状态码。

```py
import subprocess

# 启动进程并实时读取输出
process = subprocess.Popen(
    ["ping", "baidu.com"],
    stdout=subprocess.PIPE,
    text=True
)

# 逐行读取输出
while True:
    # readline() 方法会阻塞，直到有数据可用
    # 读到文件末尾，返回 None
    line = process.stdout.readline()
    if not line:
        break
    print(line.strip())
```

注意，设置 process=None 不代表杀死进程和等待结束。需要判断进程是否结束，使用两个方法：
- `poll()`：检查子进程是否已经结束，返回状态码或 None。
- `wait()`：等待子进程结束，返回状态码。

## Ref and Tag