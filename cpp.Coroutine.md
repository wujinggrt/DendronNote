---
id: 2qmsrkddst66r6o0g8as9ft
title: Coroutine
desc: ''
updated: 1733158667210
created: 1733128673861
---

本文为《C++20 the Complete Guide》协程部分笔记，涉及基本概念、promise_type 实现、co_await、co_yield、co_return、awaiter等方面内容。
# 基本概念
协程是可以暂停执行 (suspend) 的函数，暂停执行的函数会等待恢复 (resume)。这些函数不会并行地执行。函数决定开始和恢复，协程决定暂停执行和终止。一般情况，协程和其调用者在同一线程交替执行，但也可以在别的线程恢复，操作较为复杂。


一个协程必须有 co_await，co_yield，co_return 关键字其一。如都不需要，需要 co_return 区别普通函数。协程通常返回一个对象作为协程接口 (the coroutine interface) ，调用者用来控制协程。

协程是无栈的，暂停执行时，用于恢复的相关数据存储在独立于栈的地方，导致协程生命周期可以长于调用者，即调用者的栈空间回收后，不影响协程的状态。因此，传递给协程的参数尽量不要有引用。例子：
```cpp
CoroTask callCoro(const int& 3) { ... }
void foo() {
    // 此时，3是一个prvalue，是右值，当前场景下，
    // 没有任何const int&或int&&延长其lifetime，马上被destroy，造成dangling ref.
    auto coro = callCoro(3); 
    ...
}
```
通常定义 coroutine 为 lazy，创建后不立即执行并马上返回，直到调用 resume 后才开始。
```cpp
CoroTask coro() { ... }
CoroTask callCoro() { 
    coro(); // CALL sub-coroutine 
    std::cout << "callCoro(): coro() done\n"; 
    co_await std::suspend_always{}; 
    ...
} 
```
嵌套的协程中，调用者直接操作最外层的协程接口，一般只会影响最外层协程。如果需要调度内部协程，应当做必要的设置。所以一般都会将自定义的coroutine interface class 设置为`[[nodiscard]]`，编译器能够识别，提醒没有使用协程接口的部分，比如例子中 sub-coroutine，要求必须使用 coroutine interface。
```cpp
CoroTask coro() { ... }
CoroTask callCoro() { 
    coro(); // CALL sub-coroutine 
    while (sub.resume()) { ... } 
    co_await std::suspend_always{}; 
    ...
} 
```
协程可以是静态函数，成员函数（除了构造器和析构器），lambda。使用 lambda 时要注意捕获对象的生命周期。

# 实现 promise_type：处理 co_await
我们通过一个例子，查看协程需要实现的内容。以处理 co_await 为例。
```cpp
#include <coroutine>
class [[nodiscard]] CoroTask {
 public:
  struct promise_type; 
  using CoroHdl = std::coroutine_handle<promise_type>;
 private:
  CoroHdl hdl;  // native coroutine handle
 public:
  CoroTask(auto h) : hdl{h} {  // store coroutine handle in interface
  }
  ~CoroTask() {
    if (hdl) {
      hdl.destroy();  // destroy coroutine handle
    }
  }
  CoroTask(const CoroTask&) = delete;
  CoroTask& operator=(const CoroTask&) = delete;
  bool resume() const {
    if (!hdl || hdl.done()) {
      return false;  // nothing (more) to process
    }
    hdl.resume();  // RESUME (blocks until suspended again or the end)
    return !hdl.done();
  }
};

struct CoroTask::promise_type {
  CoroTask get_return_object() {  // init and return the coroutine interface
    return CoroTask{CoroHdl::from_promise(*this)};
  }
  auto initial_suspend() { // initial suspend point
    return std::suspend_always{};  // - suspend immediately
  }
  void unhandled_exception() {  // deal with exceptions
    std::terminate();           // - terminate the program
  }
  void return_void() {  // deal with the end or co_return;
  }
  auto final_suspend() noexcept {  // final suspend point
    return std::suspend_always{};  // - suspend immediately
  }
};
```

需要实现一个 public 的 struct promise_type。具体包含如下：
- get_return_object：创建协程时，返回协程接口对象给调用者。创建协程时，首先自动创建 promise_type 的对象，再从 promise 对象中，调用静态成员函数 static coroutine_handle from_promise( Promise& p )，创建协程接口，用于给调用者控制协程。
- initial_suspend：创建协程后，首先执行 co_await promise.inital_suspend()，返回的 awaiter 通常由两种：coro的启动是lazy的方式还是eager的方式：1）std::suspend_always：创建协程后将控制权立即返回调用者，下次resume再从头执行coro；一般用它，避免首次运行丢失了值。2）std::suspend_never：代表立即执行。
- return_void：处理没有 co_return 或者 co_return; 的情况；
- unhandled_exception：处理抛出的异常；
- final_suspend：最后一次暂停执行必须返回 std::suspend_always{} 或者自定义的 awaiter，并且要求 noexcept。代表协程永远暂停执行。先执行 co_return 对应的 return_value 或 return_void 后，再执行 final_suspend。返回后，调用者掌控控制权或者是恢复下一个协程。协程接口和 promise_type 先后析构和回收空间。dtor 先后调用。用于做一些清扫工作。

promise_type 的实现在协程接口类内部，或在外部提供一个 class template 并用协程接口作为模板参数。推荐使用内部实现。通常，构造协程接口对象时，首先设置 promise 内部状态，用作协程和调用者交互的桥梁。

coroutine_handle 管理着协程，生命周期和拷贝方面类似智能指针，但是拷贝和创建代价相对较低，所以传递 coroutine_handle 为参数时，一般使用传值的方式。

从 static coroutine_handle from_promise( Promise& p ) 接口可以看到，结合 get_return_obejct()，只要传入参数`*this`指向同一个 promise_type 对象，那么返回的 handle 关联对应的协程。另一个类似的接口 from_address(void* p) 同理，只是传入指针，返回对应的 coroutine_handle。

coroutine_handle 有成员函数 promise()，获取内部的promise_type 的对象，可以操作协程保存的相关数据。所有 coroutine_handle<Promise> 都可以隐式转换为特化版本 coroutine_handle<void>，void 可以简写省略，是默认的模板参数。但是这个特化版本不允许我们访问 promise()。如不需要，我们可以使用它作为接口。

注意，指向同一个协程的 coroutine_handle，它们使用 operator== 比较各自 handle 的 address()，相同才返回 true。

# 处理异常的三种方式
在 promise_type 中实现 unhandled_exception() 中护理，体现在 catch 部分：
```cpp
void unhandled_exception() { 
    try { 
        throw; // rethrow caught exception 
    } 
    catch (const std::exception& e) { 
        std::cerr << "EXCEPTION: " << e.what() << std::endl; 
    } 
    catch (...) { 
        std::cerr << "UNKNOWN EXCEPTION" << std::endl; 
    } 
} 
```

选择 aborting：
```cpp
void unhandled_exception() { 
    ... 
    std::terminate(); 
}
```

选择向调用者抛出，由调用者在每次恢复执行前置空异常指针，调度回来后执行检查：
```cpp
struct promise_type { 
    std::exception_ptr ePtr; 
    ... 
    void unhandled_exception() { 
        ePtr = std::current_exception(); 
    } 
}; 
class [[nodiscard]] CoroTask { 
    ... 
    bool resume() const { 
    if (!hdl || hdl.done()) { 
        return false; 
    } 
    hdl.promise().ePtr = nullptr; // no exception yet 
    hdl.resume(); // RESUME 
    if (hdl.promise().ePtr) { 
        // RETHROW any exception from the coroutine 
        std::rethrow_exception(hdl.promise().ePtr); 
    } 
    return !hdl.done(); 
    } 
}; 
```

# 处理 co_yield
一般保存状态到 promise_type 的 data member，由其维护并作为协程与调用者交互。

使用 co_yield 通常带有一个值，传递给 promise_type 的 yield_value() 成员函数，并随后 suspend：
```cpp
struct promise_type { 
    int coroValue = 0; // last value from co_yield 
    auto yield_value(int val) -> std::suspend_always {  // reaction to co_yield
        coroValue = val;           // - store value locally
        return {};  // - suspend coroutine
    }
    ... /* 同co_await */
};

class [[nodiscard]] CoroGen {
...
    int getValue() const { // - yield value from co_yield: 
        return hdl.promise().coroValue; } 
};
```
yield_value 总是响应最近的 co_yield，因此只能获取最新的值。 yield_value 可以重载或者设置为泛型，使得一个 coro 可以 co_yield 不同类型的值。

yield_value 的返回值使用自定义的awaiter，控制 suspend 的工作。这对嵌套的协程有时是高效的。

# 处理 co_return
协程不能使用 return，需要使用 co_return。需要进一步改造 promise_type，使用一个data member 存储结果，一般使用 std::optional<T> result{} 保存，使用return_value() 成员函数存储协程返回的结果到 result。改造如下：
```cpp
template<typename T> 
class [[nodiscard]] ResultTask { 
    struct promise_type { 
        std::optional<T> result{}; // value from co_return 
        void return_value(const auto& value) { // reaction to co_return 
            result = value; // - store value locally 
        }
        ...
    };
    ...
    // - getResult() to get the last value from co_yield 
    std::optional<T> getResult() const { 
        return hdl.promise().result; 
    } 
}; 
```

return_value 可以重载或者设置为泛型，于是 co_return 可返回不同的类型。

# Awaiter
co_await 接收的对象是 awaitable 的，或者是1）可以通过 promise_type 中await_transform 转换为 awaitable 的，或者是2）接收实现了重载操作符 co_await 并返回 awaitable 的。实现了 awaitable 的 class 称为 awaiter。需要实现如下成员函数：

auto await_ready() -> bool 暂停执行前立即调用。返回 true，代表资源 ready，不会暂停执行，继续执行执行协程；返回 false，需要暂停执行；此函数调用时，协程正在运行，不能对 coroutine_handle 调用 destroy 或 resume。

auto await_suspend(CoroHdl hdl) -> <void|bool|std::coroutine_handle<>> 在协程确认暂停执行后之后立即调用，并且将此 coroutine_handle 传递为参数。此时可调用 destroy，前提是确保后续不再使用相关内容。
* 返回 void 代表立即suspend；
* 返回 bool 时，true 继续暂停，false 不再暂停。返回值的含义与 await_read 相反。为了高效，尽量在 await_ready 中设置取消推迟。但是 await_read 没有 coroutine_handle，拿不到 promise_type 信息交互，没有更多信息来确定是否suspend。
* 返回 coroutine_handle 代表 symmetric transfer。此情况下返回 std::noop_coroutine() 代表停止恢复新的协程，将控制权交给调用者。

auto await_resume() -> T 在协程恢复后立即返回值。这个值是 co_await 或 co_yield 的返回值，从 caller 中传递给协程的值。如果 T 选择 void，代表此操作符不返回值。

典型例子为 std::suspend_always:
```cpp
namespace std { 
    struct suspend_always { 
    constexpr bool await_ready() const noexcept { return false; } 
    constexpr void await_suspend(coroutine_handle<>) const noexcept { } 
    constexpr void await_resume() const noexcept { } 
}; 
```

Awaiter 的成员函数应该尽量设计为 constexpr，const， noexcept 版本。

# Symmetric Transfer
在 await_suspend() 返回 coroutine_handle 时，可以马上 resume 对应的协程。此技术可以提升性能，避免重新调用导致的栈溢出。在 symmetric transfer 中，我们使用 std::noop_coroutine() 表示停止，将控制权转交调用者。注意，不要使用 std::noop_coroutine() == coro_handle，不会得到我们想要的结果，想要设置一个 coroutine_handle 为空，设置为 nullptr，不要设置为 std::noop_coroutine()。
```cpp
handle{std::exchange(other.handle, nullptr)}; // good
handle{std::exchange(other.handle, std::noop_coroutine())}; // bad
```

# co_await 一个非 awaitable 对象
有两种方法处理非 awaitable 对象 x 情况的 co_await x：
## 在 promise_type 中定义 await_transform()
```cpp
class CoroTask { 
    struct promise_type { 
        ... 
        auto await_transform(int val) { 
            return MyAwaiter{val}; 
        } 
    }; 
    ... 
};
```
## 重载 co_await 操作符
```cpp
class MyType { 
    auto operator co_await() { 
        return std::suspend_always{}; 
    } 
};
```
调用如 co_await MyType{}，等价于调用 suspend_always。

# 控制 co_yield 时上下文切换策略
在 promise_type 的 yield_value() 成员函数中，返回自定义的 awaiter，可以控制协程暂停执行（返回 std::suspend_always）或恢复另一个协程（自定义 awaiter 的 await_suspend() 返回另外的 coroutine_handle）。

# The Coroutine Frame：协程相关数据的存储
一个协程 frame 保存了协程必要的数据，通常保存在堆，因此生命周期比调用者更长。当然，编译器可以选择保存 frame 到栈上，这需要重载 promise_type 的 new/delete 操作符和使用 pmr 等方式，或是编译器有足够信息推断 frame 大小，因此可以放到栈上，此时生命周期短于调用者。通常，对 coroutine_handle 调用 destroy() 会释放 promise_type 对象占用的内存。

比如，协程中 co_await 一个对象，此对象保存在协程 frame 中，因此可能生命周期长于调用者。因此可以使用引用。