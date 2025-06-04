---
id: gn1lbfm8x8khkazr37aiik4
title: Io_uring
desc: ''
updated: 1748605740249
created: 1748594624171
---

io_uring 提供优雅的内核/用户空间接口，通过特殊的轮询模式，完全取消了从内核到用户空间获取数据的系统调用，提供了卓越的性能。比如，read() 通常要进行系统调用，把内容从内核缓冲区读取，再返回给用户空间。但是 io_uring 不需要这个流程，用户空间可以获取内核传来的数据。

liburing 库隐藏了很多 io_uring 相关的繁琐代码，提供了简单 API。

## cat 例程

使用 API `readv` 读取，在文件大小更大时，比 `read` 减少系统调用次数，提高性能。读取后，输出到控制台。其中，`readv` 会阻塞，直到所有 iovec 缓冲区填满。

## 使用 io_uring 实现 cat

io_uring 有一个提交队列，一个完成队列。

提交队列指出操作类型。比如需要用 `readv()` 读取文件，可以把它作为提交队列条目（SQE）的部分。我们可以提出多个请求到队列，甚至混合读、写请求，最后使用 `io_uring_enter()` 系统调用告诉内核，交付它来执行。

一旦完成，完成队列条目（CQE）内容保存了 SQE 对应的结果。我们可以在用户空间直接访问 CQE，无需再次系统调用。

整合多个 I/O 请求后，只需要一次系统调用，任务更加有效。io_uring 还提供了一种模式，内核直接轮询 SQE，我们甚至不需要调用 `io_uring_enter()`。现在的 OS 和架构，系统调用开销比以前开销更大，特别是在 Spectre 和 Meltdown 硬件漏洞被发现后，OS 解决的同时增加了昂贵的系统调用开销。对于高性能应用程序，减少系统调用影响很大。

首先，调用 `io_uring_setup()` 设置队列，指定特定大小的环形缓冲区。SQE 会添加到环形缓冲区。从完成队列取出 CQE 处理成果。

### 完成队列条目（CQE）

提交队列更加复杂，从完成队列条目入手是更好的切入点。

```c
    struct io_uring_cqe {
  __u64  user_data;  /* sqe->user_data submission passed back */
  __s32  res;        /* result code for this event */
  __u32  flags;
    };
```

`user_data` 字段从 SQE 实例对应部分传来。CQE 不一定按照 SQE 顺序到达。比如，SSD 与 HDD 的读取任务耗时存在差异，显然请求 SSD 对应的 CQE 先可用。其余字段十分直观，res 即返回值。

### 提交队列条目（SQE）

稍微复杂，但更通用了。复杂与通用的折中。

```c
struct io_uring_sqe {
  __u8  opcode;    /* type of operation for this sqe */
  __u8  flags;    /* IOSQE_ flags */
  __u16  ioprio;    /* ioprio for the request */
  __s32  fd;    /* file descriptor to do IO on */
  __u64  off;    /* offset into file */
  __u64  addr;    /* pointer to buffer or iovecs */
  __u32  len;    /* buffer size or number of iovecs */
  union {
    __kernel_rwf_t  rw_flags;
    __u32    fsync_flags;
    __u16    poll_events;
    __u32    sync_range_flags;
    __u32    msg_flags;
  };
  __u64  user_data;  /* data to be passed back at completion time */
  union {
    __u16  buf_index;  /* index into fixed buffers, if used */
    __u64  __pad2[3];
  };
};
```

关注几个关键字：
- opcode: 制定操作，比如 `readv()` 对应 IORING_OP_READV
- fd: 文件描述符
- addr: 根据 opcode，可以是指向 buffer 或 iovecs 的 ptr
- len: 根据 opcode，可以是 buffer 大小或 iovecs 数量

最终使用 `io_uring_enter()` 注册 SQE。

`user_data` 通常存放指针，对应元信息等。

## Ref and Tag