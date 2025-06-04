---
id: 5a3xsgk01h72w4ezxpchfxj
title: Re_正则
desc: ''
updated: 1748881003715
created: 1742456931966
---

## 正则的 metacharacters

```
^ $ . [ ] { } - ? * + ( ) | \
```

分为两种类型：
- basic regular expressions (BRE):  `^ $ . [ ] *`
- extended regular expressions (ERE): `()|{}?+`，使用 grep 时，需要使用 `-E` 选项

vim, find 仅支持 BRE，使用 ERE 时，需要在前面用 `\` 转义为 ERE 才行。比如 `()|{}?+`。

## 多行匹配

当字符串有换行符时，正则的 `.` 不会匹配换行符。有以下方法处理：
- 使用 `re.DOTALL` 标志（推荐）。
- 显示匹配换行符，在 pattern 添加 `\n` 或 `\s`。
- 显示匹配所有字符。比如 `[\s\S]*` 匹配所有空白（包括换行、制表符、空白字符）和非空白字符。

注意，re.M 影响的是 `^$` 这两个元字符。

## 搜索

匹配成功都返回 group。group(0) 代表匹配到的内容，group(i) 代表第 i 组，每组由括号指定。

### 返回对象 re.Match

| 方法          | 描述                                                             |
| ------------- | ---------------------------------------------------------------- |
| group(num=0)  | num=0 代表匹配的整个表达式的字符串，其他值则是对应的元组。       |
| groups()      | 返回包含所有小组字符串的元组。                                   |
| span([group]) | 0 代表整个匹配的字符串的区间下标，其他值对应小组字符串起始下标。 |

### match()

re.match(pattern, string, flags=0)

从字符串的起始位置开始匹配，要求 string 起始处就满足 pattern 的匹配（不一定匹配到末尾，可以仅仅匹配前面部分），否则返回 None。如果需要从指定位置开始匹配，参考后面的 re.compile()。

### search()

re.search(pattern, string, flags=0)

扫描整个字符串并返回**第一个**成功的匹配，否则返回 None。

### split(pattern, string, maxsplit=0, flags=0)

根据出现的 pattern 拆分字符串。如果 pattern 包含括号，也会返回 pattern，方便查看是哪些内容分割了字符串。

```py
>>> re.split(r'\W+', 'Words, words, words.') # \W 代表非 word，比如 ", " 和 "."
['Words', 'words', 'words', '']
>>> re.split(r'(\W+)', 'Words, words, words.')
['Words', ', ', 'words', ', ', 'words', '.', ''] # 包含了分割的部分
>>> re.split(r'\W+', 'Words, words, words.', maxsplit=1)
['Words', 'words, words.']
>>> re.split('[a-f]+', '0a3B9', flags=re.IGNORECASE)
['0', '3', '9']
```

### findall(pattern, string, flags=0)

与 split() 相反，split() 以 pattern 拆分，保留非 pattern 的内容。而 findall() 保留 pattern 匹配内容。如果 pattern 包含括号指定的分组，则返回的列表只包含分组。

```py
>>> re.findall(r'\bf[a-z]*', 'which foot or hand fell fastest')
['foot', 'fell', 'fastest']
>>> re.findall(r'(\w+)=(\d+)', 'set width=20 and height=10')
[('width', '20'), ('height', '10')]
```

### finditer(pattern, string, flags=0)

返回迭代器，能够生成 re.Match 对象。

```py
it = re.finditer(r"\d+","12a32bc43jf3") 
for match in it: 
    print (match.group() )
```

### re.compile(pattern[, flags]) -> re.Pattern

返回的 class re.Pattern 实例，有如下方法：

| 方法                                        | 描述                                                 |
| ------------------------------------------- | ---------------------------------------------------- |
| search(string[, pos[, endpos]]) -> re.Match | 相当于 re.search(pattern, string[pos:endpos], flags) |
| match(string[, pos[, endpos]])              | 类似                                                 |
| split(string, maxsplit=0)                   | 同 re.split()                                        |

## re.sub(pattern, repl, string, count=0, flags=0) 类似 sed 替换功能

将 string 最左侧匹配到 pattern 的内容，使用 repl 替换为指定内容，并返回替换后的 repl。如果没有匹配，返回原字符串。repl 可以是字符串，也可以是函数。

pattern 如果部分匹配 string，返回的字符串中，pattern 匹配到的部分替换为按 repl 处理后部分，其余部分保持不变。

### repl 为字符串

repl 中，任何 `\` 内容都会被转义，比如 `\n` 转义为代表换行。未在 ASCII 中定义的，则保留。如果反斜杠带着数字，则引用 pattern 匹配到的组，比如 `\6` 匹配第六组。

```py
>>> re.sub(r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):',
...        r'static PyObject*\npy_\1(void)\n{',
...        'def myfunc():')
'static PyObject*\npy_myfunc(void)\n{'
```

例子中，\1 代表第一个 group，指代 myfunc，于是返回 repl 中的内容。最后返回 repl 中替换 `\1` 后的内容。

### repl 为函数

函数接受 re.Match 为参数，处理后，返回字符串，作为 re.sub() 返回的字符串。

```py
>>> def dashrepl(matchobj):
...     if matchobj.group(0) == '-': return ' '
...     else: return '-'
...
>>> re.sub('-{1,2}', dashrepl, 'pro----gram-files')
'pro--gram files'
>>> re.sub(r'\sAND\s', ' & ', 'Baked Beans And Spam', flags=re.IGNORECASE)
'Baked Beans & Spam'
```

由于默认贪婪匹配，`-{1,2}` 匹配到了 `----` 中的 `--` 两次，所以返回了 dashrepl 的 else 分支，两个 `-` 合为一个，得到最终结果的两个 `-` 部分。另一方面，gram 和 files 之间的 `-`，只有一个，所以返回空格。

## re.escape(pattern)

```py
>>> print(re.escape('https://www.python.org'))
https://www\.python\.org
```

将 pattern 中的特殊字符转义，比如 `.` 转义为 `\.`。在我们有一段字符串时，并且其中有一些比如 `.` 或 `*` 的元字符，那么 re.escape() 可以帮我们更好地匹配。

```py
def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification
        
    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    ...
    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)    
        ...
```

比如，从模型回答的文本中，我们想用期待的名字来匹配，但是，万一名字中有 `.` 等元字符，可能匹配不到我们期待的名字。比如，期待匹配类似 `Dr. Marting is a knight`。

## 断言的语法结构

在匹配字符和元组时，指定匹配的内容满足某些条件。

### 正向先行断言（positive  lookahead）

正向先行断言（positive lookahead），查找某个位置后面是否跟随特定的模式，而不消耗实际的字符。

形式：`(?=pattern)`，比如 `\d+(?=PM)` 能够匹配 `12PM`，`3PM` 等，我们的 `\d+` 除了匹配数字，还要求后面紧跟的内容不能为 `PM`，且 `PM` 不消耗匹配的字符，不会在模式中有任何处理。

```py
import re
res = re.findall(r'\d+(?=PM)', '12PM 3PM 4AM')
print(res)
# ['12', '3']
```

### 负向先行断言（negative lookahead）

匹配字符串时，查找某个位置之后是否**不跟着**特定的模式。

形式：`(?!pattern)`，比如 `\d+(?!PM)` 能够匹配 `12AM`，`3AM` 等。

```bash
$ echo "123AW"  | perl -wnle 'm@\d+(?!AW)@ and print $&;'
# 12
```

### 正向后行断言（positive lookbehind）

位置之前是否跟着特定模式。

形式：`(?<=pattern)`，比如 `(?<=abc)def` 能够匹配 `abcdef`，结果为 `def`。

### 负向后行断言（negative lookbehind）

位置之前方是否不跟着特定模式。

形式：`(?<!pattern)`，比如 `(?<!abc)def` 能够匹配 `xyzdef`，结果为 `def`。

### 应用

在 tar 命令打包项目时，需要排除 `build` 和 `tmp` 等目录，可以使用通配符的方案：

```bash
find . -maxdepth 1 \( -not -name '*/build' -and -not -name '*/tmp' \)
```

正则的方案：

```bash
find . -maxdepth 1 -not -regex '.*/\(build\|tmp\)'
```

注意，find 命令中的 regex 使用元字符 `()|` 等需要使用 `\` 转义。

使用断言来排除多个单词，正则如下：

```regex
^(?!.*build)(?!.*tmp).*$
```

```bash
find . -maxdepth 1 | perl -wnle 'm@^(?!.*(build|tmp)$)@ and print;'
```

负向先行断言，匹配如此的开头（对应 `^` 部分）；此开头后续不能跟任意以 `build` 或 `tmp` 结尾的字符串，其中 `$` 代表字符串限定在最末尾了。

## Ref and Tag

[菜鸟教程](https://m.runoob.com/python/python-reg-expressions.html)