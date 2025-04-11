---
id: lj8j2g0qy6jssv0o858y531
title: Mermaid_画图工具和语法
desc: ''
updated: 1744394007003
created: 1744387527662
---

## Graph, Flowchat

```mermaid
graph TD
    classDef param stroke:#916fdb,stroke-width:2px;
    classDef op fill:#b0dfe5,stroke:#4a85b1,stroke-width:2px;
    W((W)):::param --> OpM((\*)):::op
    X((X)):::param --> OpM

    b((b)):::param --> OpPlus((\+)):::op
    OpM --> OpPlus
    OpPlus --> OpMinus((\-)):::op
    y((y)):::param --> OpMinus
    OpMinus --> OpSq((\^2)):::op
    OpSq --> L((L)):::param
    style L fill:green,stroke:blue,stroke-width:2px
```

设置颜色：fill 填充颜色，stroke 和 stroke-width 指定边界颜色和宽度。有两种方式应用到节点上：
- 使用 style 指出，仅对单个节点起作用。
- 定义为 class 后，使用 ::: 符号指定类型，可以复用颜色配置。

```mermaid
flowchart LR
D[renderer] <--> A[Dev Server] <--ws--> B[service]
B <--mcp--> m(MCP Server)
```

```mermaid
flowchart TB
    c1-->a2
    subgraph one
    a1-->a2
    end
    subgraph two
    b1-->b2
    end
    subgraph three
    c1-->c2
    end
```

## 时序图：Sequence diagram

```mermaid
sequenceDiagram
    participant Alice
    participant Bob
    Alice->>John: Hello John, how are you?
    loop HealthCheck
        John->>John: Fight against hypochondria
    end
    Note right of John: Rational thoughts <br/>prevail!
    John-->>Alice: Great!
    John->>Bob: How about you?
    Bob-->>John: Jolly good!
```

```mermaid
sequenceDiagram
    autonumber
    Student->>Admin: Can I enrol this semester?
    loop enrolmentCheck
        Admin->>Admin: Check previous results
    end
    Note right of Admin: Exam results may <br> be delayed
    Admin-->>Student: Enrolment success
    Admin->>Professor: Assign student to tutor
    Professor-->>Admin: Student is assigned
```

## 类表（Class Diagram），UML 图

```mermaid
classDiagram
Class01 <|-- AveryLongClass : Cool
Class03 *-- Class04
Class05 o-- Class06
Class07 .. Class08
Class09 --> C2 : Where am i?
Class09 --* C3
Class09 --|> Class07
Class07 : equals()
Class07 : Object[] elementData

Class01 : size()
Class01 : int chimp
Class01 : int gorilla

Class08 <--> C2: Cool label
```

```mermaid
---
title: Animal example
---
classDiagram
    note "From Duck till Zebra"
    Animal <|-- Duck

    note for Duck "can fly<br>can swim
    can dive
    can help in debugging"

    Animal <|-- Fish
    Animal <|-- Zebra
    Animal : +int age
    Animal : +String gender
    Animal: +isMammal() -> bool
    Animal: +mate() -> Animal

    class Duck{
        +beakColor: str
        +swim()
        +quack()
    }
    class Fish{
        -sizeInFeet: int
        -canEat() -> bool
    }
    class Zebra{
        +is_wild: bool
        +run()
    }
```

Tips: `<br>` 可以换行。

## Ref and Tag

官网教程有很多示例
https://mermaid.js.org/intro/