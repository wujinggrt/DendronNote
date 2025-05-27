---
id: 6jmn71jemd9d7rz9fp41fzl
title: ZeroMQ
desc: ''
updated: 1748154553994
created: 1743325170137
---

## 消息队列

安装 `pip install pyzmq`，通过 zmq.send/recv 的 copy=False 参数实现零拷贝，直接映射整块 chunk 的数据到内核，适合大数组传输。


| 模式          | 特点                       | 适用场景                                                       |
| ------------- | -------------------------- | -------------------------------------------------------------- |
| REQ-REP       | 严格同步，一问一答         | 需要严格交替顺序的场景。不能打破顺序，连续请求和连续接收消息。 |
| PUB-SUB       | 一对多广播，异步通信       | 实时数据分发（如日志）                                         |
| PUSH-PULL     | 单向流水线，多对多通信     | 并行任务分发                                                   |
| ROUTER-DEALER | 支持异步多路复用，复杂路由 | 高并发服务端                                                   |

选择建议
- 简单同步任务：REQ/REP。
- 广播通知：PUB/SUB。
- 并行任务分发：PUSH/PULL。
- 高并发服务端：ROUTER/DEALER。
- 设备直连或线程通信：PAIR。

### zmq.REQ / zmq.REP（严格同步）

```py
# 服务端（REP）
rep_socket = context.socket(zmq.REP)
rep_socket.bind("tcp://*:5555")
request = rep_socket.recv()
rep_socket.send(b"Response")

# 客户端（REQ）
req_socket = context.socket(zmq.REQ)
req_socket.connect("tcp://localhost:5555")
req_socket.send(b"Request")
response = req_socket.recv()
```

### zmq.PUB / zmq.SUB（广播订阅）

```py
# 发布者（PUB）
pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://*:6000")
pub_socket.send(b"news: Today's headlines...")

# 订阅者（SUB）
sub_socket = context.socket(zmq.SUB)
sub_socket.connect("tcp://localhost:6000")
sub_socket.setsockopt(zmq.SUBSCRIBE, b"news")  # 订阅以 "news" 开头的消息
message = sub_socket.recv()
```

### zmq.PUSH / zmq.PULL（负载均衡）

```py
# 生产者（PUSH）
push_socket = context.socket(zmq.PUSH)
push_socket.bind("tcp://*:7000")
push_socket.send(b"Task 1")  # 自动分发给所有 PULL 节点

# 消费者（PULL）
pull_socket = context.socket(zmq.PULL)
pull_socket.connect("tcp://localhost:7000")
task = pull_socket.recv()
```

### zmq.ROUTER / zmq.DEALER（异步路由）

```py
# 服务端（ROUTER）
router = context.socket(zmq.ROUTER)
router.bind("tcp://*:8000")
client_id, _, request = router.recv_multipart()
router.send_multipart([client_id, b"", b"Response"])

# 客户端（DEALER）
dealer = context.socket(zmq.DEALER)
dealer.connect("tcp://localhost:8000")
dealer.send(b"Request")
response = dealer.recv()
```

### zmq.PAIR（双向对等通信）

```py
context = zmq.Context()
# 进程 A
pair_a = context.socket(zmq.PAIR)
# 还可以绑定网络通信，比如
# pair_a.bind("tcp://*:15555")
pair_a.bind("ipc:///tmp/pair.ipc")
pair_a.send(b"Hello from A")
message = pair_a.recv()

# 进程 B
pair_b = context.socket(zmq.PAIR)
# pair_b.connect("tcp://pair_a_ip:15555")
pair_b.connect("ipc:///tmp/pair.ipc")
message = pair_b.recv()
pair_b.send(b"Hello from B")
```

此模式可以不严格遵循发送和接收的顺序，双发随时可以发送和接收。recv() 时，如果对方没有 send()，则会阻塞。只能点对点，不像 REQ/REP 模式可以一对多。但仅推荐用于简单、固定的点对点连接需求。

### ZeroMQ：（REQ-REP 模式） + 零拷贝传输

REQ-REP 模式。

实现架构：
- 服务端（进程A）：启动 ZeroMQ 服务端，调用 socket.recv() 等待客户端请求，随后 socket.send() 返回结果。注意，必须严格遵循 recv() -> send() -> recv() -> ... 顺序。否则报错。
- 客户端（进程B）：连接到服务端，socket.send() 发送数据，socket.recv() 等待数据。必须严格遵循 send() -> recv() -> send() ... 顺序，打乱顺序会报错。
- 交替处理：通过 REQ-REP 模式强制请求-响应顺序，确保双方交替操作。

简短示例如下，对于服务端的进程 A，类比实现 sam2 的求掩码：

```py
import zmq
import numpy as np

def process_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP 模式（服务端）
    socket.bind("tcp://*:15555")  # 监听 15555 端口
    print("Server started")
    # 2 bytes for points
    IMAGE_CHUNK_SIZE = 480 * 640 * 4
    BOUNDING_BOX_CHUNK_SIZE = 4 * 2
    chunk_size = IMAGE_CHUNK_SIZE + BOUNDING_BOX_CHUNK_SIZE
    while True:
        # 接收客户端发送的二进制数据（零拷贝）
        msg = socket.recv(copy=False)
        print(f"recerved msg size: {len(msg)}")
        assert len(msg) == chunk_size
        arr = np.frombuffer(
            msg.buffer[:-BOUNDING_BOX_CHUNK_SIZE], dtype=np.uint8
        ).reshape(480, 640, 4)
        bbox_2d = np.frombuffer(
            msg.buffer[-BOUNDING_BOX_CHUNK_SIZE:], dtype=np.uint16
        ).reshape(4,)
        # 处理数据（示例：简单反转数值）
        processed_arr = 255 - arr
        print(f"Received and processed data: {arr.shape} {bbox_2d.shape}\n{bbox_2d}...")
        # 发送处理后的数据（零拷贝）
        socket.send(processed_arr.tobytes(), copy=False)


if __name__ == "__main__":
    process_server()

```

客户端：

```py
import zmq
import time
import numpy as np

def process_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # REQ 模式（客户端）
    socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
    socket.connect("tcp://localhost:15555")  # 连接到服务端
    # 初始化数据（示例：全零数组）
    data = np.zeros((480, 640, 4), dtype=np.uint8)
    bbox_2d = np.array([1, 2, 3, 4], dtype=np.uint16)
    max_try = 5
    tried = 0
    IMAGE_CHUNK_SIZE = 480 * 640 * 4
    BOUNDING_BOX_CHUNK_SIZE = 4 * 2
    chunk_size = IMAGE_CHUNK_SIZE + BOUNDING_BOX_CHUNK_SIZE
    while True:
        # 发送数据（零拷贝）
        socket.send(data.tobytes() + bbox_2d.tobytes(), copy=False)
        # 接收服务端处理后的数据（零拷贝）
        try:
            msg = socket.recv(copy=False)
            tried = 0
            data = np.frombuffer(msg.buffer, dtype=np.uint8).reshape(480, 640, 4)
            # 客户端进一步处理（示例：加 1）
            # 注意，与整数操作后，可能会从 uint8 变为 int16，会影响 buffer 大小
            data = (data + 1) % 256
            data = data.astype(np.uint8)
        except zmq.Again:
            # REQ 模式下，两次连续的 send 会报错，所以要重新初始化。
            context = zmq.Context()
            socket = context.socket(zmq.REQ)  # REQ 模式（客户端）
            socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
            socket.connect("tcp://localhost:15555")  # 连接到服务端
            if tried >= max_try:
                print("max_try count reached, stop trying")
                break
            print("Timeout, retrying...")
            tried += 1
            socket.send(data.tobytes(), copy=False)  # 重发请求
if __name__ == "__main__":
    process_client()
```


## 资源释放和重连

ZeroMQ的套接字（Socket）和上下文（Context）是单次性资源。一旦调用socket.close()，底层资源会被完全释放，套接字将进入不可用状态，无法通过重新调用connect()或其他方法恢复。

​上下文依赖：套接字对象依赖于其关联的上下文（zmq.Context()）。如果上下文已被销毁（如调用context.term()或程序退出），所有关联的套接字也会失效。

正确关闭方式：

```py
def close(self):
    self.socket.close()
    self.context.term()  # 销毁上下文
    self.connected = False
```

恢复重连应该用：

```py
def reconnect(self, port: int):
    self.context = zmq.Context()  # 新建上下文
    self.socket = self.context.socket(zmq.REQ)  # 新建套接字
    self.socket.connect(f"tcp://localhost:{port}")
    self.connected = True
```

## 特定场景

我的服务器在与办公室的路由器同级。分别是 192.168.19.204 和 192.168.19.54。办公室的主机和笔记本都由路由器连接，构成局域网。局域网内的主机能够直接访问服务器，可以 ssh 登录和 ping 通。但是服务器不能穿透路由器，ping 局域网的主机。所以，服务器只能充当 ZeroMQ 服务器的角色，即用服务器调用 socket.bind()。若是由局域网主机调用 socket.bind()，服务器不能够找到它，socket.connect() 则无法执行。

## Ref and Tag