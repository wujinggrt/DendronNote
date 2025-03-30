---
id: 6jmn71jemd9d7rz9fp41fzl
title: ZeroMQ
desc: ''
updated: 1743325263769
created: 1743325170137
---


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

## Ref and Tag