---
id: 2dei298qbfeobl9krl61vnp
title: VSCode_配置clangd与头文件包含地址等
desc: ''
updated: 1750817151620
created: 1750313789535
---

## 学习特性

在 AnthonyCalandra/modern-cpp-features 学习新特性。

## 安装

```bash
sudo apt install clangd

```

## 配置 clangd

clangd 在版本 14 后，提供变量自动类型提示，auto 也能看到类型。

### 项目级（推荐）

在项目根目录创建 .clangd 文件，比如：

```yaml
CompileFlags:
  Add: [-std=c++17, -I./include]  # 添加编译选项和头文件路径
Diagnostics:
  ClangTidy: true  # 启用静态检查
```

### 全局

在 VS Code 的 settings.json 设置：

```json
{
  // clangd 自动类型提示，auto 也能看到类型
  "editor.inlayHints.enabled": "on",
  "clangd.path": "clangd",  // 若 clangd 不在 PATH 中，改为绝对路径
  "clangd.arguments": [
    "--background-index",     // 后台索引
    "--clang-tidy",          // 启用 clang-tidy
    "--header-insertion=never", // 禁止自动插入头文件
    "--query-driver=/usr/bin/g++"  // 指定编译器路径（适配你的项目）
  ]
}
```

在 VS Code 中，需要禁用 C/C++ 扩展，它们有冲突。

### 生成编译数据库（compile_commands.json）

CMake 项目配置时时，指定 `-DCMAKE_EXPORT_COMPILE_COMMANDS=1` 即可。

```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1
```

ament_cmake：

```bash
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

可以用 bear 工具：

```bash
bear -- colcon build
```

如果 CMake 文件不完整，可以用 bear：

```bash
bear -- cmake --build build
```

## 配置头文件地址

目录 /usr/include 下的头文件，编译器和 clangd 自动查找。

使用 compile_commands.json 最全，也可以手动配置。

### ROS2 的头文件

ROS2 项目头文件通常在 `/opt/ros/humble/include` 路径。

## Ref and Tag

[官网](https://clangd.llvm.org/config.html)