---
id: 7o2b9g6gjspfptoaw8mq28j
title: Quick_start
desc: ''
updated: 1745780956224
created: 1745779126636
---

## 环境配置

需要 npm（Node Package Manager），一个 JavaScript 包管理工具，也是 Node.js 的默认包管理器。Ubuntu 安装如下：

```bash
sudo apt install -y nodejs npm
# 有时当前不是最新的，更新如下
sudo npm install npm -g
```

首先设置国内镜像，随后安装，完成后测试：

```bash
npm config set registry https://registry.npmmirror.com
npm install -g typescript
tsc -v
```

Hello World 例子：

```ts
// app.ts
var message: string = "Hello World" 
console.log(message)
```

编译并转换为 JS 代码：

```bash
tsc app.ts
node app.js
```

在同级目录生成 app.js，可以运行。

## 特点

TypeScript 是 JavaScript 的超集，可以编译成 JavaScript。有静态类型检查，类型推断，接口和类型定义，类和模块支持，工具和编辑器支持，兼容 JavaScript等优点。

基本类型：


| 类型      | 描述                             | 示例                                 |
| --------- | -------------------------------- | ------------------------------------ |
| string    | 表示文本数据                     | let name: string = "Alice";          |
| number    | 表示数字，包括整数和浮点数       | let age: number = 30;                |
| boolean   | 表示布尔值 true 或 false         | let isDone: boolean = true;          |
| array     | 表示相同类型的元素数组           | let list: number[] = [1, 2, 3];      |
| tuple     | 表示已知类型和长度的数组         | let person: [string, number] = ["Alice", 30];|
| enum      | 定义一组命名常量                 | enum Color { Red, Green, Blue };     |
| any       | 任意类型，不进行类型检查         | let value: any = 42;                 |
| void      | 无返回值（常用于函数）           | function log(): void {}              |
| null      | 表示空值                         | let empty: null = null;              |
| undefined | 表示未定义                       | let undef: undefined = undefined;    |
| never     | 表示不会有返回值                 | function error(): never { throw new Error("error"); } |
| object    | 表示非原始类型                   | let obj: object = { name: "Alice" }; |
| union     | 联合类型，表示可以是多种类型之一 | `let id: string                      |
| unknown   | 不确定类型，需类型检查后再使用   | let value: unknown = "Hello";          |

## Ref and Tag

https://www.runoob.com/typescript/ts-tutorial.html