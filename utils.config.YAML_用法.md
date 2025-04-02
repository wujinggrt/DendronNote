---
id: swykb3cramun062tsnjetsr
title: YAML_用法
desc: ''
updated: 1743586356865
created: 1743585843505
---

大小写敏感; 使用缩进表示层级关系; 不能用 Tab 只能空格，空格对其即可,不要求数量。通常选择 2 或者 4。# 开头代表注释.

## 数据类型:
- 纯量 (scalars)。纯量包含: boolean, float, int, null (使用 ~ 表示 null), string, date, datetime.
- 数组
- 对象


## 对象

对象是键值对集合, 也称为 mapping, hashes, dictionary。键值对使用 `key: value` 表示，冒号之后必须有空格。对象的形式可以为:

```yaml
key: {key1: value1, key2: value2, ...}
```

可以是

```yaml
key: 
    child-key: value
    child-key2: value2
```

可以是
```yaml
?  
    - complexkey1
    - complexkey2
:
    - complexvalue1
    - complexvalue2
```

## 数组

以 `-` 开头的内容代表数组：

```yaml
- A
- B
- C
```

子成员可以是数组。

```yaml
-
 - A
 - B
 - C
```

第一个 `-` 是数组成员，紧跟着数组成员。随后，代表此成员是一个数组，成员有 `A, B, C`。等价于: `[[A, B, C]]`

比如下面几个都是等价的

```yaml
companies:
    -
        id: 1
        name: company1
        price: 200W
    -
        id: 2
        name: company2
        price: 500W
```

```yaml
companies: [{id: 1,name: company1,price: 200W},{id: 2,name: company2,price: 500W}]
```

## 复合的例子

```yaml
languages:
  - Ruby
  - Perl
  - Python 
websites:
  YAML: yaml.org 
  Ruby: ruby-lang.org 
  Python: python.org 
  Perl: use.perl.org
```

对应JSON为:

```json
{ 
  languages: [ 'Ruby', 'Perl', 'Python'],
  websites: {
    YAML: 'yaml.org',
    Ruby: 'ruby-lang.org',
    Python: 'python.org',
    Perl: 'use.perl.org' 
  } 
}
```

例子launch属性是一个数组,有三个值:

```yaml
launch:
- node:
    pkg: "turtlesim"
    exec: "turtlesim_node"
    name: "sim"
    namespace: "turtlesim1"

- node:
    pkg: "turtlesim"
    exec: "turtlesim_node"
    name: "sim"
    namespace: "turtlesim2"

- node:
    pkg: "turtlesim"
    exec: "mimic"
    name: "mimic"
    remap:
    -
        from: "/input/pose"
        to: "/turtlesim1/turtle1/pose"
    -
        from: "/output/cmd_vel"
        to: "/turtlesim2/turtle1/cmd_vel"
```

使用 & 声明锚点,使用 * 引用。

## Ref and Tag