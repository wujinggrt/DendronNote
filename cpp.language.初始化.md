---
id: vm8s235zpyh0sczlau2ltt9
title: 初始化
desc: ''
updated: 1747222025715
created: 1747207216410
---

## 花括号初始化

花括号初始化时，构造函数中，以初始化列表为参数的版本有优于其他版本的优先级。以 `std::vector<int>` 为例，会出现如此行为：

```cpp
std::vector<int> vec1 = {2, 3};
std::vector<int> vec2(2, 3);
```

上面两者出现不同效果。分别匹配如下的 ctor：

```cpp
constexpr vector( std::initializer_list<T> init,
                  const Allocator& alloc = Allocator() );
constexpr vector( size_type count,
                  const T& value,
                  const Allocator& alloc = Allocator() );
```

vec1 的数据为 {2, 3}，而 vec2 的数据为 {3, 3, 3}。

### 花括号中包含花括号

初始化列表中，也是初始化列表的元素。

```cpp
std::unordered_map<int,std::string> m2 = {{1,"foo"},{3,"bar"},{2,"baz"}};
```

上述情况下，ctor 匹配到构造函数：

```cpp
unordered_map( std::initializer_list<value_type> init,
               size_type bucket_count = /* implementation-defined */,
               const Hash& hash = Hash(),
               const key_equal& equal = key_equal(),
               const Allocator& alloc = Allocator() );
```

外层花括号 {{...}, {...}, {...}} 会被识别为 `std::initializer_list<std::pair<const Key, Value>>`，内层花括号 {key, value} 对应一个键值对（`std::pair<const Key, Value>` 的构造参数）。例如，{1, "foo"} 会隐式构造一个 `pair<const int, std::string>`。

如果需要指定桶的数量，可以调用如下：

```cpp
std::unordered_map<int, std::string> m2({{1,"foo"}, {3,"bar"}}, 100);
```

初始化列表为参数 ctor 逐元素遍历，再调用 value_type 的初始化列表 ctor 版本。

#### 参考源码

参考 [gcc 的 unordered_map](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/unordered_map.h)。最终调用如下的版本：

```cpp
  template<typename _Key,
       typename _Tp,
       typename _Hash = hash<_Key>,
       typename _Pred = std::equal_to<_Key>,
       typename _Alloc = std::allocator<std::pair<const _Key, _Tp> >,
       typename _Tr = __umap_traits<__cache_default<_Key, _Hash>::value>>
    using __umap_hashtable = _Hashtable<_Key, std::pair<const _Key, _Tp>,
                                        _Alloc, __detail::_Select1st,
                        _Pred, _Hash,
                        __detail::_Mod_range_hashing,
                        __detail::_Default_ranged_hash,
                        __detail::_Prime_rehash_policy, _Tr>;
  ...
  /**
   ...
   *  Base is _Hashtable, dispatched at compile time via template
   *  alias __umap_hashtable.
   */
  template<typename _Key, typename _Tp,
       typename _Hash = hash<_Key>,
       typename _Pred = equal_to<_Key>,
       typename _Alloc = allocator<std::pair<const _Key, _Tp>>>
    class unordered_map
    {
      typedef __umap_hashtable<_Key, _Tp, _Hash, _Pred, _Alloc>  _Hashtable;
      _Hashtable _M_h;
      typedef typename _Hashtable::key_type	key_type;
      typedef typename _Hashtable::value_type	value_type;
      ...

      unordered_map(initializer_list<value_type> __l,
            size_type __n = 0,
            const hasher& __hf = hasher(),
            const key_equal& __eql = key_equal(),
            const allocator_type& __a = allocator_type())
      : _M_h(__l, __n, __hf, __eql, __a)
      { }
      ...
    };
```

key_type 类型是 `Key`; 而 value_type 的类型是保存的节点类型，`std::pair<const Key, T>`。所以初始化列表的每个元素，都会是 std::pair 类型。

类型萃取 libstdc%2B%2B-v3/include/bits/hashtable_policy.h：

```cpp
  template<bool _Cache_hash_code, bool _Constant_iterators, bool _Unique_keys>
    struct _Hashtable_traits
    {
      using __hash_cached = __bool_constant<_Cache_hash_code>;
      using __constant_iterators = __bool_constant<_Constant_iterators>;
      using __unique_keys = __bool_constant<_Unique_keys>;
    };
```

可以猜测 __unique_keys 只可能是 true_type 和 false_type 类型。

__umap_hashtable 在文件 libstdc%2B%2B-v3/include/bits/hashtable.h：

```cpp
  template<typename _Key, typename _Value, typename _Alloc,
       typename _ExtractKey, typename _Equal,
       typename _Hash, typename _RangeHash, typename _Unused,
       typename _RehashPolicy, typename _Traits>
    class _Hashtable
    : public __detail::_Hashtable_base<_Key, _Value, _ExtractKey, _Equal,
                       _Hash, _RangeHash, _Unused, _Traits>,
      public __detail::_Map_base<_Key, _Value, _Alloc, _ExtractKey, _Equal,
                 _Hash, _RangeHash, _Unused,
                 _RehashPolicy, _Traits>,
      public __detail::_Rehash_base<_Key, _Value, _Alloc, _ExtractKey, _Equal,
                    _Hash, _RangeHash, _Unused,
                    _RehashPolicy, _Traits>,
      private __detail::_Hashtable_alloc<
    __alloc_rebind<_Alloc,
               __detail::_Hash_node<_Value,
                        _Traits::__hash_cached::value>>>,
      private _Hashtable_enable_default_ctor<_Equal, _Hash, _Alloc>
    {
        using __traits_type = _Traits;
        using __unique_keys = typename __traits_type::__unique_keys;

        _Hashtable(initializer_list<value_type> __l,
            size_type __bkt_count_hint = 0,
            const _Hash& __hf = _Hash(),
            const key_equal& __eql = key_equal(),
            const allocator_type& __a = allocator_type())
        : _Hashtable(__l.begin(), __l.end(), __bkt_count_hint,
            __hf, __eql, __a, __unique_keys{})
        { }

        template<typename _InputIterator>
        _Hashtable(_InputIterator __f, _InputIterator __l,
            size_type __bkt_count_hint = 0,
            const _Hash& __hf = _Hash(),
            const key_equal& __eql = key_equal(),
            const allocator_type& __a = allocator_type())
        : _Hashtable(__f, __l, __bkt_count_hint, __hf, __eql, __a,
                __unique_keys{})
        { }
        ...

        _Hashtable(const _Hash& __h, const _Equal& __eq,
            const allocator_type& __a)
        : __hashtable_base(__h, __eq),
        __hashtable_alloc(__node_alloc_type(__a)),
        __enable_default_ctor(_Enable_default_constructor_tag{})
        { }
    };

  template<typename _Key, typename _Value, typename _Alloc,
           typename _ExtractKey, typename _Equal,
           typename _Hash, typename _RangeHash, typename _Unused,
           typename _RehashPolicy, typename _Traits>
    template<typename _InputIterator>
      inline
      _Hashtable<_Key, _Value, _Alloc, _ExtractKey, _Equal,
         _Hash, _RangeHash, _Unused, _RehashPolicy, _Traits>::
      _Hashtable(_InputIterator __f, _InputIterator __l,
         size_type __bkt_count_hint,
         const _Hash& __h, const _Equal& __eq,
         const allocator_type& __a, true_type /* __uks */)
      : _Hashtable(__bkt_count_hint, __h, __eq, __a)
      { this->insert(__f, __l); }

  // Definitions of class template _Hashtable's out-of-line member functions.
  template<typename _Key, typename _Value, typename _Alloc,
           typename _ExtractKey, typename _Equal,
           typename _Hash, typename _RangeHash, typename _Unused,
           typename _RehashPolicy, typename _Traits>
    _Hashtable<_Key, _Value, _Alloc, _ExtractKey, _Equal,
               _Hash, _RangeHash, _Unused, _RehashPolicy, _Traits>::
    _Hashtable(size_type __bkt_count_hint,
               const _Hash& __h, const _Equal& __eq, const allocator_type& __a)
    : _Hashtable(__h, __eq, __a) // 仅仅初始化基类
    {
      auto __bkt_count = _M_rehash_policy._M_next_bkt(__bkt_count_hint);
      if (__bkt_count > _M_bucket_count)
        {
          _M_buckets = _M_allocate_buckets(__bkt_count);
          _M_bucket_count = __bkt_count;
        }
    }
```

初始化基类后，使用 insert() 成员函数插入初始化列表 `std::initializer_list<std::pair<const K, T>>` 的各个元素。比如 {1,"foo"}，会被认为是 std::pair 类型，传递给 insert()。有 sizeof...(_Args) 为 1。参考如下：

```cpp
    template<typename _InputIterator>
    void
    insert(_InputIterator __first, _InputIterator __last)
    {
      if constexpr (__unique_keys::value)
        for (; __first != __last; ++__first)
          _M_emplace_uniq(*__first);
      else
        return _M_insert_range_multi(__first, __last);
    }

  template<typename _Key, typename _Value, typename _Alloc,
       typename _ExtractKey, typename _Equal,
       typename _Hash, typename _RangeHash, typename _Unused,
       typename _RehashPolicy, typename _Traits>
    template<typename... _Args>
      auto
      _Hashtable<_Key, _Value, _Alloc, _ExtractKey, _Equal,
         _Hash, _RangeHash, _Unused, _RehashPolicy, _Traits>::
      _M_emplace_uniq(_Args&&... __args)
      -> pair<iterator, bool>
      {
        const key_type* __kp = nullptr;

        if constexpr (sizeof...(_Args) == 1)
        {
            if constexpr (__is_key_type<_Args...>)
            {
                const auto& __key = _ExtractKey{}(__args...);
                __kp = std::__addressof(__key);
            }
        }
        else if constexpr (sizeof...(_Args) == 2)
        {
            if constexpr (__is_key_type<pair<const _Args&...>>)
            {
                pair<const _Args&...> __refs(__args...);
                const auto& __key = _ExtractKey{}(__refs);
                __kp = std::__addressof(__key);
            }
        }

        _Scoped_node __node { __node_ptr(), this }; // Do not create node yet.
        __hash_code __code = 0;
        size_type __bkt = 0;

        if (__kp == nullptr)
        {
            // Didn't extract a key from the args, so build the node.
            __node._M_node
            = this->_M_allocate_node(std::forward<_Args>(__args)...);
            const key_type& __key = _ExtractKey{}(__node._M_node->_M_v());
            __kp = std::__addressof(__key);
        }

        if (auto __loc = _M_locate(*__kp))
            // There is already an equivalent node, no insertion.
            return { iterator(__loc), false };
        else
        {
            __code = __loc._M_hash_code;
            __bkt = __loc._M_bucket_index;
        }

        if (!__node._M_node)
            __node._M_node
                = this->_M_allocate_node(std::forward<_Args>(__args)...);

        // Insert the node
        auto __pos = _M_insert_unique_node(__bkt, __code, __node._M_node);
        __node._M_node = nullptr;
        return { __pos, true };
      }
```

## Ref and Tag