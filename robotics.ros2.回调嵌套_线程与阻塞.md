---
id: bmrw6bg20dzhrju0e9i72sz
title: 回调嵌套_线程与阻塞
desc: ''
updated: 1753548663121
created: 1753506597402
---

在 ROS2 中，如果想要在一个 callback 中 sleep，那么会阻塞当前主线程，其他 callback 不能够有机会调用。可以使用事件循环，asyncio.sleep() 之后，马上切出去。在 callback 中封装协程，提交到事件循环中后，马上返回，由后台的事件循环来执行实际的逻辑，前台的 callback 只需要马上返回即可：

```py
class SomeNode(Node):
    def __init__(...):
        ...
        self.sub = self.create_subscription(
            ...
            self.ros_callback
        )
        # 创建新事件循环
        self.loop = asyncio.new_event_loop()

    # 事件循环线程函数
    def run_event_loop(self):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def ros_callback(self, msg):
        """ROS回调函数 - 在ROS线程中执行"""
        # 安全提交协程到事件循环线程
        future = asyncio.run_coroutine_threadsafe(
            self.async_operation(msg.data),
            self.loop
        )
        
        # 添加完成回调处理结果
        future.add_done_callback(self.handle_async_result)
```

## ROS2 中的阻塞

在 callback 中，不指定 callback_group，那么在一个 callback 中 sleep，或者 rclpy.spin_once()，阻塞会导致其他 callback 排队，或者饿死，没有机会执行。必须串行地等待它完成。

本质上，执行器维护任务队列和回调状态。

## 服务的回调中调用另一个服务：合理让出控制权

https://answers.ros.org/question/342302/

如果在 callback 中，直接使用 spin_node_until_future_complete()，检查异步调用的 future 是否完成，那么永远无法给到其他 callback 机会设置 future 完成状态，最终会死循环等待，拿不到结果。原因是此 callback 执行完成后，才有机会给其他 callback 运行，并没有多线程的方案。使用 while 判断 future.done() 和 spin_oncec() 同理。有两个解决方案，都需要多线程，分别是直接让出控制权，或是在不同的 callback group 运行。

这是因为 spin 系列接口用在回调中，会破坏执行器内部状态。具体表现在：
-   尝试在已经运行的执行器内部再次处理事件循环
-   干扰执行器的回调调度机制
-   可能导致执行器线程进入不可恢复的状态
-   破坏定时器和其他回调的调度队列

所以，在官网 service 的例子中，看到 spin 的例子仅仅用在外部函数，而非节点的 callback。

### 使用线程的 Event.wait() 让出控制权

https://gist.github.com/driftregion/14f6da05a71a57ef0804b68e17b06de5

使用 Event 的 wait 方案让出权限等待。所以用 Event.wait() 让出控制权，给其他线程执行 callback，检查 future 状态。注意，main() 函数中需要多线程行器。

```py
import time
from threading import Event

from example_interfaces.srv import AddTwoInts

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor


class ServiceFromService(Node):

    def __init__(self):
        super().__init__('action_from_service')
        self.service_done_event = Event()

        self.callback_group = ReentrantCallbackGroup()

        self.client = self.create_client(
            AddTwoInts,
            '/add_two_ints',
            callback_group=self.callback_group
        )

        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints_proxy',
            self.add_two_ints_proxy_callback,
            callback_group=self.callback_group
            )

        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback,
            )

    def add_two_ints_callback(self, request, response):
        self.get_logger().info('Request received: {} + {}'.format(request.a, request.b))
        response.sum = request.a + request.b
        return response

    def add_two_ints_proxy_callback(self, request, response):
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('No action server available')
            return response

        self.service_done_event.clear()

        event=Event()
        def done_callback(future):
            nonlocal event
            event.set()

        future = self.client.call_async(request)
        future.add_done_callback(done_callback)

        # Wait for action to be done
        # self.service_done_event.wait()
        event.wait()

        return future.result()

    def get_result_callback(self, future):
        # Signal that action is done
        self.service_done_event.set()


def main(args=None):
    rclpy.init(args=args)

    service_from_service = ServiceFromService()

    executor = MultiThreadedExecutor()
    rclpy.spin(service_from_service, executor)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 使用 callback group + busy waiting（推荐）

或者是使用不同的 callback_group，**创建新的 callback group，把请求任务提交到其他 group**，放在当前 group 会阻塞。

在 C++ 的构造函数创建新的 cg，单独给客户端请求使用。Python 同理，在构造函数创建。

```cpp
callback_group_input_ = this->create_callback_group(rclcpp::callback_group::CallbackGroupType::MutuallyExclusive);
get_input_client_ = this->create_client<petra_core::srv::GetInput>("GetInput", rmw_qos_profile_services_default, callback_group_input_);
```

在外部的 callback 中，请求并注册响应的 callback：

```cpp
int choice = -1;
auto inner_client_callback = [&,this](rclcpp::Client<petra_core::srv::GetInput>::SharedFuture inner_future)
    { 
        auto result = inner_future.get();
        choice = stoi(result->input);
        RCLCPP_INFO(this->get_logger(), "[inner service] callback executed");
    };
auto inner_future_result = get_input_client_->async_send_request(inner_request, inner_client_callback);
// 如需等待则 busy waiting
while (choice < 0 && rclcpp::ok())
{
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
```

注意，Group 有多种，可以直接实例化，比如 ReentrantCallbackGroup 和 MutuallyExclusiveCallbackGroup，两种方式都可以使用，建议互斥的版本，线程安全。对于 Python 版本：

```py
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class ProxyServiceServer(Node):
    def __init__(self):
        ...
        self._proxy_srv = self.create_service(Trigger, "main_service", self.proxy_callback)
        self._cli_cg = MutuallyExclusiveCallbackGroup()
        self._cli = self.create_client(Trigger, "nested_service", callback_group=self._cli_cg)
    
    def proxy_callback(self, request, response):
        done = False
        def done_callback(request_future):
            nonlocal done
            nonlocal response
            r = request_future.result()
            response.success = r.success
            response.message = r.message
            done = True
        f = self._cli.call_async(request)
        f.add_done_callback(done_callback)
        while not done and rclpy.ok():
            time.sleep(0.5)
        return response
```

使用 rclpy.spin_until_future_complete(self, future) 会导致整个 callback 阻塞，只能服务一次。

```py
    def proxy_callback(self, request, response):
        f = self._cli.call_async(request)
        # 注意，可能只能调用一次，之后所有内容都阻塞了。
        # rclpy.spin_until_future_complete(self, f)
        # 应该
        while not f.done() and rclpy.ok():
            time.sleep(0.1)
        return f.result()
```

## Ref and Tag