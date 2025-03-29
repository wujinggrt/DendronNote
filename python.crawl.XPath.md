---
id: lyr2i1azrkyi8fossrsnpwd
title: XPath
desc: ''
updated: 1740075376646
created: 1740073452503
---

## 查找元素
XPath (XML Path Language) 用于在 XML 和 HTML 文档中查找信息。

### 节点有七类
元素、属性、文本、命名空间、处理指令、注释以及文档（根）节点。XML 文档是被作为节点树来对待的。树的根被称为文档节点或者根节点。

### 语法
| 表达式   | 描述                                                 | 示例                | 结果                                                                 |
| -------- | ---------------------------------------------------- | ------------------- | -------------------------------------------------------------------- |
| nodename | 当前节点的所有节点                                   | bookstore           | 选取 bookstore 下所有子节点                                          |
| /        | 当前节点下的某节点 | /bookstore          | 当前节点即根节点，查找其下孩子中所有 bookstore 节点，孙子节点匹配不到                        |
| //       | 全局选节点                                           | //book              | 当前节点下的全局查找 book 节点，可以与当前节点有间隔，是孙子节点都可 |
| @        | 选取节点属性                                         | //title[@lang='en'] | 节点 title 的 lang 属性为 en                                         |

`..`获取父节点，比如`//a[@href="link4.html]/../@class`。

#### 选取节点属性
使用`[]`填写谓语提取节点：谓语可以是常见函数、表达式、属性判别等。

| 路径表达式                           | 结果                                                                                    |
| ------------------------------------ | --------------------------------------------------------------------------------------- |
| `/bookstore/book[1]`                 | 选取属于 bookstore 子元素的第一个 book 素。                                             |
| `/bookstore/book[last()]`            | 选取属于 bookstore 子元素的最后一个 book 素。                                           |
| `/bookstore/book[last()-1]`          | 选取属于 bookstore 元素的倒数第二个 book 素。                                           |
| `/bookstore/book[position()<3]`      | 选取最前面的两个属于 bookstore 素的子元素的 book 素。                                   |
| `//title[@lang]`                     | 选取所有拥有名为 lang 的属性的 title 素。                                               |
| `//title[@lang='eng']`               | 选取所有 title 素，且这些元素拥有值为 eng 的 lang 属性。                                |
| `/bookstore/book[price>35.00]/title` | 选取 bookstore 素中的 book 元素的所有 title 元素，且其中的 price 元素的值须大于 35.00。 |
| `//a[text()="first item"]/text()`    | 匹配满足 text() 为 first item 的节点，                                                  |

#### 谓词
谓词支持通配符。比如 `//book[@*]`匹配。

| 运算符 | 描述           | 实例                        | 返回值                                                              |
| ------ | -------------- | --------------------------- | ------------------------------------------------------------------- |
| `      | `              | 计算两个节点集              | `//book                                                             | //cd` | 返回所有拥有 book 和 cd 元素的节点集 |
| `+`    | 加法           | `6 + 4`                     | 10                                                                  |
| `-`    | 减法           | `6 - 4`                     | 2                                                                   |
| `*`    | 乘法           | `6 * 4`                     | 24                                                                  |
| `div`  | 除法           | `8 div 4`                   | 2                                                                   |
| `=`    | 等于           | `price=9.80`                | 如果 price 是 9.80，则返回 true。如果 price 是 9.90，则返回 false。 |
| `!=`   | 不等于         | `price!=9.80`               | 如果 price 是 9.90，则返回 true。如果 price 是 9.80，则返回 false。 |
| `<`    | 小于           | `price<9.80`                | 如果 price 是 9.00，则返回 true。如果 price 是 9.90，则返回 false。 |
| `<=`   | 小于或等于     | `price<=9.80`               | 如果 price 是 9.00，则返回 true。如果 price 是 9.90，则返回 false。 |
| `>`    | 大于           | `price>9.80`                | 如果 price 是 9.90，则返回 true。如果 price 是 9.80，则返回 false。 |
| `>=`   | 大于或等于     | `price>=9.80`               | 如果 price 是 9.90，则返回 true。如果 price 是 9.70，则返回 false。 |
| `or`   | 或             | `price=9.80 or price=9.70`  | 如果 price 是 9.80，则返回 true。如果 price 是 9.50，则返回 false。 |
| `and`  | 与             | `price>9.00 and price<9.90` | 如果 price 是 9.80，则返回 true。如果 price 是 8.50，则返回 false。 |
| `mod`  | 计算除法的余数 | `5 mod 2`                   | 1                                                                   |

有时候某个属性中包含了多个值，那么可以使用contains函数，比如`//title[contains(@lang,'en')]`和`//title[contains(text(), "XX") and contains(@name, "item")]`。

注意，谓词下标从 1 起始，而非 0。

获取

#### lxml
lxml 是一个 HTML/XML 解析器。[官方文档](http://lxml.de/index.html)。安装`pip install lxml`即可。可以用 HTML 字符串解析，也可以从文件解析。lxml 的 Element 对象是节点：
```py
from lxml import etree

text = """
<html><body><div><ul>
    <li class="item-0"><a href="link1.html">first item</a></li>
    <li class="item-1"><a href="link2.html">second item</a></li>
    <li class="item-inactive"><a href="link3.html">third item</a></li>
    <li class="item-1"><a href="link4.html">fourth item</a></li>
    <li class="item-0"><a href="link5.html">fifth item</a>
</li></ul></div>
</body></html>
"""
# 将字符串解析为html文档
html = etree.HTML(text)
result = etree.tostring(html).decode("utf-8")
print(result)  # 得到一个完整的页面

# 得到所有节点元素对象
print(html.xpath("//*"))
# ["first item"]
print(html.xpath("//*[contains(text(), 'first item') and contains(@href, 'link1.html')]/text()"))
# text() 放置最后，便索引标签内部的值。下面查找 href 为对应值的节点，并输出其标签内的文本，得 ['fifth item']
# 否则返回 [Element 对象, ...]
print(html.xpath("//li//*[@href='link5.html']/text()"))
# 现实所有 a 节点各自的属性值 ['link1.html', 'link2.html', 'link3.html', 'link4.html', 'link5.html'
print(html.xpath("//a/@href"))

# 所有祖先节点
print(html.xpath("//li[1]/ancestor::*"))
# 所有为 div 的祖先节点
print(html.xpath("//li[1]/ancestor::div"))
# 所有属性
print(html.xpath("//li[1]/attribute::*"))
# 所有直接联系的孩子，即 first item 的 a
print(html.xpath("//li[1]/child::*/text()"))
# 子孙，只有一个，即 first item 的 a
print(html.xpath("//li[1]/descendant::*/text()"))
```

在 lxml 中使用 XPath。注意，xpath() 方法返回 list 对象。


```py
from lxml import etree
html =etree.parse('hello.html')
# 获取所有li标签：
result =html.xpath('//li')
print(result) # list
for i in result:
    print(etree.tostring(i))
# 获取所有li元素下的所有class属性的值：
result =html.xpath('//li/@class')
print(result)# 获取li标签下href为www.baidu.com的a标签：
result =html.xpath('//li/a[@href="www.baidu.com"]')
print(result)# 获取li标签下所有span标签：
result =html.xpath('//li//span')
print(result)
获取li标签下的a标签里的所有class：
result =html.xpath('//li/a//@class')
print(result)
获取最后一个li的a的href属性对应的值：
result =html.xpath('//li[last()]/a/@href')# print(result)
获取倒数第二个li元素的内容：
result =html.xpath('//li[last()-1]/a')
print(result)
print(result[0].text)
获取倒数第二个li元素的内容的第二种方式：
result =html.xpath('//li[last()-1]/a/text()')
print(result)
```

其他例子：
* `x1 = "//span[contains(./descendant::a/text(), '赖明星')]"`，contains 内使用节点轴
* `doc.xpath(r'//*[re:match(@id, "postmessage_\d+")]', namespace={"re": "http://exslt.org/regular-expressions"})` 正则
