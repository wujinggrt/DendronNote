---
id: tlutv61oazq3u89aeis7fql
title: Range
desc: ''
updated: 1733076835880
created: 1732967433669
---

# Ranges and Views

定义和使用 ranges 和 subranges 为单独对象，我们可以不再使用 begin() 和 end() 两个迭代器操作范围。新的特性和工具：
1. algorithms 方面出现新的重载和变种，使用单独的参数；
2. 一些处理 range 对象的工具，比如辅助函数用于创建和处理 range 对象，辅助类型处理 range 对象，concepts 处理 ranges；
3. views，轻量级 ranges，用于处理 range 中的转换；
4. 管道 (pipelines) 提供了灵活处理 ranges 和 views 的方式。

## 以 ranges 的方式传递容器到 algorithms

```cpp
std::vector<int> coll{25, 42, 2, 0, 122, 5, 7};
std::ranges::sort(coll); // sort all elements of the collection
```

众多标准的算法支持传递单个 range 作为参数，这些新的算法通常在 std::ranges，部分在 std::views 中，有着对应的版本。应尽可能用 std::ranges 版本，新的工具通常会纠正某些错误。

新的特性通常在`<ranges>`头文件中提供，对于既存的头文件，比如`<algorithm>`，也提供了 ranges 的算法。我们应该总是引入`<ranges>`库，以使用更完全的特性。

ranges 的限制和工具中，引入了 concepts 等一系列工具。比如 std::ranges::sort 中，要求 std::ranges::random_access_range，std::sortable。
```cpp
template<std::ranges::random_access_range R,
typename Comp = std::ranges::less>
requires std::sortable<std::ranges::iterator_t<R>, Comp>
... sort(R&& r, Comp comp = {});
```

在 std::ranges 命名空间中，ranges 的基础 concepts 有 range, output_range, input_range, forward_range, bidirectional_range, random_access_range, contiguous_range, sized_range, view, viewable_range, borrowed_range, common_range。
```cpp
template< class T >
concept range = requires( T& t ) {
    ranges::begin(t); // equality-preserving for forward iterators
    ranges::end (t);
};
```

## Views
Views 是轻量级 ranges，创建、拷贝和移动开销低、速度快。Views 可以引用 ranges 和 subranges，拥有临时 ranges，筛选和转换 ranges 的元素，生成元素。

比如，使用 range adaptor 创建一个 view，操作前 5 个元素：
```cpp
for (const auto& elem : std::views::take(coll, 5)) {
    ...
}
```

可以使用管道的语法操作 range，这样便可创建一连串的 views 来操作 range：
```cpp
auto v = coll
        | std::views::filter([] (auto elem) { return elem % 3 == 0; })
        | std::views::transform([] (auto elem) { return elem * elem; })
        | std::views::take(3);
```

由于 view 也是 range，在接收 range 的地方我们也传入 view。
```cpp
auto v = std::views::take(
    std::views::transform(
        std::views::filter(coll,
                           [] (auto elem) { return elem % 3 == 0; }),
        [] (auto elem) { return elem * elem; }),
    3);
```

Views 可以生成元素，比如：
```cpp
for (int val : std::views::iota(1, 11)) { // iterate from 1 to 10
    ...
}
```

多数情况，views 作用于左值 (lvalues) 时，拥有引用语义，因此需要关注操作 views 期间引用的生命周期。

通常情况，我们使用 adaptors 和 factories 来创建 views，它们有着自己的类型并且方便，比如 std::ranges::take_view 等。

关于修改和写入，可以传递 views 给 ranges 版本算法，比如：
```cpp
std::ranges::sort(coll | std::views::take(5)); // sort the first five elements of coll
```

views 是惰性求值的，它只会在调用 begin() 返回迭代器时，执行 ++ 或访问对应元素的值时，才开始处理。

一些 views 会使用缓存，比如访问第一个元素时，调用 begin() 会执行一些计算，那么可能会缓存这个结果。

### Sentinels
哨兵，sentinels，代表 range 的末尾。通常是一个特殊值，比如字符串结尾的'\0'，链表的 nullptr，非负数的列表中的 -1。在 STL 中，使用 end() 的迭代器作为 sentinel，但在 ranges 中，可以使用不同类型的值作为 sentinels。

要求 end 迭代器与普通迭代器为同类型有一些缺点，比如创建一个 end 迭代器需要开销。比如对一个 C 字符串，获取 end 迭代器需要迭代到字符串末尾，是 O(n) 耗时。有时迭代两次是不可能的，比如输入中的 EOF，再次读取可能得到不同结果。

比如，输入流迭代器，我们会使用 std::istream_iterator<>{} 表示 end 迭代器。
```cpp
std::for_each(std::istream_iterator<int>{std::cin}, // read ints from cin
    std::istream_iterator<int>{}, // end is an end-of-file iterator
    [] (int val) {
        std::cout << val << '\n';
});
```
如果使用 sentinels 作为迭代器，它的类型可以不同于普通迭代器：
1. 一开始确定 begin 迭代器时，不用确定 end 迭代器，而是在迭代过程中，处理数据并查找迭代器；
2. 对于 end 迭代器，我们使用新的类型。此类型不支持迭代器的相关操作，因此在编译期可以检测出对 end 迭代器操作的问题；
3. 定义容易。
```cpp
struct NullTerm {
    bool operator== (auto pos) const {
        return *pos == '\0'; // end is where iterator points to ’\0’
    }
};
int main()
{
    const char* rawString = "hello world";
    for (auto pos = rawString; pos != NullTerm{}; ++pos) {
        std::cout << ' ' << *pos;
    }
    std::cout << '\n';
    // call range algorithm with iterator and sentinel:
    std::ranges::for_each(rawString, // begin of rangeNullTerm{}, // end is null terminator
        NullTerm{}, // end is null terminator
        [] (char c) {
        std::cout << ' ' << c;
    });
    std::cout << '\n';
}
```

其中，重载操作符 == 的参数使用了 auto ，相当于成员函数模板。这是 C++20 特性，与模板有类似的特点和要求，比如我们不能在不同的翻译单元 (translation units) 中实现此模板的内容。同样地，不能在一个函数的作用域中定义如此的模板。

只定义 ==，尽管大多算法库都用到了 !=，这是因为 C++20 开始，编译器会自动地基于 == 来将 a != b 的表达式重写 (rewrites) 为 !(a==b)，或者是 !(b==a)，于是只实现 == 足够。

### 使用 Sentinels 和 Counts 定义 Range
Ranges 可以是容器，也可以是迭代器对。定义如下方式之一：
1. 一个 begin 和一个 end 迭代器，要求类型相同；
2. 一个 begin 迭代器和一个 sentinel，类型可以不同；
3. 一个 begin 迭代器和一个计数 (count)；
4. 数组。比如原生的数组。

#### subrange
另外，有一些工具可以定义 ranges，涉及上述方式。比如创建 subrange 使用 std::ranges::subrange，返回一个 view，于是拷贝开销小：
```cpp
std::ranges::subrange rawStringRange{rawString, NullTerm{}};
// use the range in an algorithm:
std::ranges::for_each(rawStringRange,
    [] (char c) {
    std::cout << ' ' << c;
});
std::cout << '\n';
```

注意，subrange 不是常规 range，调用 begin() 和 end() 可能得到不同类型结果。有了 range 或 view，我们可以使用 range-for，更方便和可读。另一个例子：
```cpp
template<auto End>
struct EndValue {
    bool operator== (auto pos) const {
        return *pos == End; // end is where iterator points to End
    }
};
int main()
{
    std::vector coll = {42, 8, 0, 15, 7, -1};
    // define a range referring to coll with the value 7 as end:
    std::ranges::subrange range{coll.begin(), EndValue<7>{}};
    // sort the elements of this range:
    std::ranges::sort(range);
    // print the elements of the range:
    std::ranges::for_each(range,
        [] (auto val) {
        std::cout << ' ' << val;
        });
    std::cout << '\n';
    // print all elements of coll up to -1:
    std::ranges::for_each(coll.begin(), EndValue<-1>{},
        [] (auto val) {
        std::cout << ' ' << val;
        });
    std::cout << '\n';
}
// 输出
// 0 8 15 42
// 0 8 15 42 7
```

std::unreachable_sentinel 也是一个例子，作为无终止的 range。

#### 处理 Begin 和 Count 定义的 Range
使用 range adaptor std::views::counted() 能够更方便和高效地创建 views，由原 range 的前 n 元素组成。
```cpp
std::vector<int> coll{1, 2, 3, 4, 5, 6, 7, 8, 9};
auto pos5 = std::ranges::find(coll, 5);
if (std::ranges::distance(pos5, coll.end()) >= 3) {
    for (int val : std::views::counted(pos5, 3)) {
        std::cout << val << ' ';
    }
}
```
std::views::counted(pos5, 3) 创建 view，由 pos5 起始，紧跟 3 个元素。counted 不做边界检查，我们用 std::ranges::distance() 检查是否有足够元素。

对比 std::views::take()：当我们仅有一个迭代器和 count，用 counted() 更方便。比如，我们只有 pos5，没有 end 迭代器。当我们有一个 range，需要处理前 n 个元素，take 更合适。

### Projections
sort() 和众多算法中，还可以接受一个额外的模板参数，projection。在执行算法时，关于每个元素，先变换 (projection) 元素，再执行具体算法。
```cpp
template<std::ranges::random_access_range R,
    typename Comp = std::ranges::less,
    typename Proj = std::identity>
requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
... sort(R&& r, Comp comp = {}, Proj proj = {});

std::ranges::sort(coll,
    std::ranges::less{}, // still compare with <
    [] (auto val) { // but use the absolute value
        return std::abs(val);
    });
```