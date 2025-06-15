---
id: yv3a0r2jcb9n0zxx1vjdyo8
title: CMake_quick_start
desc: ''
updated: 1749998395797
created: 1749877457406
---

## 编译

```cmake
add_executable(<name> <source1> [source2...] [source3...])
add_library(<name> [STATIC|SHARED|MODULE] <source1> [source2...] [source3...])
```

最终生成动态库文件：`lib<name>.so`。

使用别名，添加命名空间：

```cmake
add_library(<alias_name> ALIAS <target>)
```

例子：

```cmake
add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
```

### 包含头文件

target_include_directories() 命令用于为特定目标（如可执行文件或库）指定头文件的搜索路径，告诉编译器和 IDE 在哪里查找头文件。用法：

```CMake
target_include_directories(<target>
    <INTERFACE|PUBLIC|PRIVATE> [directory1...]
    [<INTERFACE|PUBLIC|PRIVATE> [directory2...] ...]
)
```

- `<target>` 目标名称（通过 add_executable() 或 add_library() 创建的目标）
- `PRIVATE` 仅当前 `<target>` 使用，不传递给依赖它的目标。适用于头文件用于实现当前目标场景，如 .cpp 需要的内部头文件
- `INTERFACE` 不用于当前目标 `<target>`，仅传递给依赖它的目标。适用场景：头文件仅用于依赖者，当前项目并不使用，如纯头文件库
- `PUBLIC` 当前目标使用 且传递给依赖它的目标（PRIVATE + INTERFACE）。适用于头文件用于目标接口的场景，如公共 API 头文件中，.h 文件既被当前目标使用，也被依赖者使用
- `[directory]` 头文件路径（绝对路径或相对路径，推荐使用 ${CMAKE_CURRENT_SOURCE_DIR} 作为起始）

### 链接库

```cmake
target_link_libraries(<target> [PUBLIC | PRIVATE | INTERFACE] [items])
```

- `PUBLIC`：将库加入当前目标，并传递给依赖它的目标。
- `PRIVATE`：将库加入当前目标，但不传递给依赖它的目标。
- `INTERFACE`：将库加入当前目标，但当前目标不能使用，依赖于当前目标的能使用。常用于接口库。

## 子目录

```cmake
add_subdirectory(<name>)
```

将子目录加入构建列表。会进入此目录，并执行 CMakeLists.txt 文件，根据子目录构建规则整合到主项目。

比如，子目录构建了一个库，主项目可以使用它。

```cmake
# 主 CMakeLists.txt
add_subdirectory(libs/mylib)  # 构建 mylib 库
target_link_libraries(my_app PRIVATE mylib)  # 主程序链接 mylib
```

用于支持外部项目集成，通过 add_subdirectory() 命令，嵌入外部项目到当前项目。子目录的 add_library() 或 add_executable() 命令，是全局可见的。

在子目录是 test 场景下，通常把 add_subdirectory(test) 放到末尾，编译好主目录的目标文件后，再编译子目录的测试用例。

由于 add_library() 在主目录中先于子目录执行，所以子目录的 CMakeLists.txt 可以引用主目录的库，比如使用 target_link_libraries() 引用主目录的库，指导生成对应编译文件。

注意要避免循环依赖。

可以指定子目录构建位置：

```cmake
add_subdirectory(src src_build)  # 子目录构建产物放在 src_build 中
```


## 导入外部库

### 本地库

根据库的精确路径导入：

```cmake
add_library(my_library SHARED IMPORTED)
set_target_properties(my_library PROPERTIES
    IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/libs/libmy_library.so
)
```

导入静态库时，只需把 SHARED 替换为 STATIC。此方法推荐在项目中，明确有第三方库时使用。

或者使用如下方式，更简便直观：

```cmake
find_library(RELAXED_IK_LIB
    NAMES librelaxed_ik_lib.so
    PATHS ${PROJECT_SOURCE_DIR}/libs /usr/local/lib
)
```

此方法适合在库的位置不固定时查找。

NAMES 参数可以强制查找 .so 库。如果不指定后缀，则会查找匹配的。比如使用 NAMES relaxed_ik_lib，遇到 librelaxed_ik_lib.so 时，会匹配。

### 编译

不同模块，可以 add_library() 之后，在 target_link_libraries() 来指定连接，最后合并输出。

### 使用动态库时，注意路径

移动编译后的文件，可能因为路径改变而导致找不到动态库。需要给连接器信息查找库。

#### 设置 RPATH（最佳实践）

```cmake
# 设置RPATH
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
```

设置 CMAKE_INSTALL_RPATH，设置安装后的运行时库搜索路径（RPATH）。

- `$ORIGIN`：特殊变量，表示安装后的可执行文件自身所在的目录
    - 在 Linux 上等同于 `$ORIGIN`
    - 在 macOS 上等同于 `@loader_path`
    - 在 Windows 上不直接支持（需要其他处理）
- `$ORIGIN/../lib`：表示可执行文件所在目录的上一级目录中的 lib 子目录
- `分号(;)`：分隔多个搜索路径

当程序运行时，动态链接器会按顺序查找：
1. 可执行文件所在目录（`$ORIGIN`）
2. 可执行文件所在目录的上级目录中的 lib 目录（`$ORIGIN/../lib`）

设置 CMAKE_BUILD_WITH_INSTALL_RPATH 为 TRUE，在构建阶段使用安装后的 RPATH。在 build 目录可以使用 RPATH，验证安装后是否正确。

设置 CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE 之后，在 RPATH 中包含链接时使用的库路径，自动将链接器找到库的路径添加到 RPATH，使得 link_directories() 和 find_library() 能够找得到。

#### 设置 LD_LIBRARY_PATH （测试时使用）

假设需要链接的库在 ../lib 目录。

```bash
export LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH
```

### FetchContent 模块

#### 拉取远程库

引入模块

```cmake
include(FetchContent)
```

FetchContent_Declare 声明拉取的内容，FetchContent_MakeAvailable 自动处理获取，调用 add_subdirectory() 和处理声明中的依赖。

```cmake
find_package(yaml-cpp REQUIRED)
if (yaml-cpp_FOUND)
    message(STATUS "Found yaml-cpp")
else()
    message(STATUS "yaml-cpp not found")
    include(FetchContent)
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(yaml-cpp)
endif()

target_link_libraries(my_app PRIVATE yaml-cpp::yaml-cpp)
```

#### 使用本地副本

可以使用本地副本源码：

```cmake
FetchContent_Declare(
  local_lib
  SOURCE_DIR /path/to/local/source  # 关键参数
)
```

压缩包下载（URL/HTTP）：

```cmake
FetchContent_Declare(
  local_lib
  SOURCE_DIR /path/to/local/source  # 关键参数
)
```

## find_package: 查找外部依赖包

```cmake
find_package(<PackageName> [version] [REQUIRED] [COMPONENTS <components>])
```

引用的项目如需指定路径，可以配置：

```cmake
set(CMAKE_PREFIX_PATH /path/to/my_package)
```

### 组织项目为 package

核心步骤：
1. 创建包配置文件模板（`<PackageName>Config.cmake.in`）
2. 生成版本配置文件（`<PackageName>ConfigVersion.cmake`）
3. 配置并安装包配置文件
4. 安装目标文件和头文件
5. 导出目标信息

参考 [yaml-cpp](https://github.com/jbeder/yaml-cpp/blob/master/CMakeLists.txt)。

创建包配置模板 yaml-cpp-config.cmake.in：

```cmake
@PACKAGE_INIT@  # CMake 自动生成的初始化代码
# 设置并验证头文件目录
# @PACKAGE_CMAKE_INSTALL_INCLUDEDIR@ 被替换为实际安装路径（如 /usr/include
# 不存在会报错
set_and_check(YAML_CPP_INCLUDE_DIR "@PACKAGE_CMAKE_INSTALL_INCLUDEDIR@")
# 验证库目录
set_and_check(YAML_CPP_LIBRARY_DIR "@PACKAGE_CMAKE_INSTALL_LIBDIR@")

# 替换为构建时的值 ON/OFF
set(YAML_CPP_SHARED_LIBS_BUILT @YAML_BUILD_SHARED_LIBS@)

# 核心，目标定义文件，比如 yaml-cpp::yaml-cpp
include("${CMAKE_CURRENT_LIST_DIR}/yaml-cpp-targets.cmake")

# These are IMPORTED targets created by yaml-cpp-targets.cmake
# 设置导出来的目标名，如 yaml-cpp::yaml-cpp
set(YAML_CPP_LIBRARIES "@EXPORT_TARGETS@")

# 解决多次包含问题
if(NOT TARGET yaml-cpp)
  add_library(yaml-cpp INTERFACE IMPORTED) 
  # 向后兼容，连接到 yaml-cpp，使得旧式风格 yaml-cpp 也能正常使用
  target_link_libraries(yaml-cpp INTERFACE yaml-cpp::yaml-cpp) 
  # 弃警告
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    set_target_properties(yaml-cpp PROPERTIES 
      DEPRECATION "The target yaml-cpp is deprecated and will be removed in version 0.10.0. Use the yaml-cpp::yaml-cpp target instead."
    )
  endif()
endif()

# 检查 find_package() 是否请求有效组件
check_required_components(yaml-cpp)
```

先引入模块：

```cmake
include(CMakePackageConfigHelpers)
```

指定版本配置文件：

```cmake
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/yaml-cpp-config-version.cmake"
  COMPATIBILITY AnyNewerVersion)
```

包文置文件：

```cmake
set(EXPORT_TARGETS yaml-cpp::yaml-cpp)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/yaml-cpp-config.cmake.in"
  "${PROJECT_BINARY_DIR}/yaml-cpp-config.cmake"
  INSTALL_DESTINATION "${YAML_CPP_INSTALL_CMAKEDIR}"
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR)
unset(EXPORT_TARGETS)
```

安装：

```cmake
set(YAML_CPP_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/yaml-cpp"
  CACHE STRING "Path to install the CMake package to")

if (YAML_CPP_INSTALL)
  # 库文件 yaml-cpp
  install(TARGETS yaml-cpp
    EXPORT yaml-cpp-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  # 安装头文件
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
                FILES_MATCHING PATTERN "*.h")
  # 导出目文件
  install(EXPORT yaml-cpp-targets
    NAMESPACE yaml-cpp::
    DESTINATION "${YAML_CPP_INSTALL_CMAKEDIR}")
  # 安装配置文件
  install(FILES
      "${PROJECT_BINARY_DIR}/yaml-cpp-config.cmake"
      "${PROJECT_BINARY_DIR}/yaml-cpp-config-version.cmake"
    DESTINATION "${YAML_CPP_INSTALL_CMAKEDIR}")
  # 
  install(FILES "${PROJECT_BINARY_DIR}/yaml-cpp.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()
```

安装库：

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ..
cmake --build . --target install
```

注意：
1. 命名规范：配置文件必须命名为 `<PackageName>Config.cmake` 或 `<package-name>-config.cmake`
2. 目标命名空间：通过 NAMESPACE MyLibrary:: 确保目标被正确命名
3. 路径结构：安装到 lib/cmake/MyLibrary 是标准做法
4. 依赖传递：在 Config.cmake.in 中使用 find_dependency() 处理项目依赖

提示：使用 cmake --install . --prefix=/install/path 进行安装，确保所有组件被正确部署。测试时可通过设置 CMAKE_PREFIX_PATH 指向安装目录验证查找功能。



## file: 文件操作

### 复制文件

运行 cmake 命令时复制。

```cmake
file(COPY <file> [<file>...] DESTINATION <dir> [<options>])
```

```cmake
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/file
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}}
    PATTERN ".gitkeep" EXCLUDE
)
```

### 通配符获取文件

复制特定文件类型：

```cmake
file(GLOB CONFIG_FILES
    "${CMAKE_SOURCE_DIR}/configs/*.yaml"
    "{CMAKE_SOURCE_DIR}/configs/*.json"
)

foreach(CONFIG_FILE ${CONFIG_FILES})
    file(COPY ${CONFIG_FILE}
         DESTINATION ${CMAKE_BINARY_DIR}/configs)
endforeach()
```

## 安装

### 安装编译文件

```cmake
install(TARGETS my_app
    RUNTIME DESTINATION bin          # 安装可执行文件到 bin 目录
    LIBRARY DESTINATION lib          # 安装共享库到 lib 目录
    ARCHIVE DESTINATION lib/static   # 安装静态库到 lib/static 目录
)
```

| 关键字   | 安装内容              | 典型目标位置 |
| -------- | --------------------- | ------------ |
| RUNTIME  | 可执行文件、动态库    | bin          |
| LIBRARY  | 共享库（.so, .dylib） | lib          |
| ARCHIVE  | 静态库（.lib, .a）    | lib/static   |
| INCLUDES | 头文件                | include      |

#### COMPONENT - 指定安装组件分组

COMPONENT 关键字用于将安装内容分组，允许用户选择性安装项目中的不同部分。主要用途：
- 选择性安装：用户可以只安装需要的部分
- 模块化打包：为不同组件创建单独的分发包
- 依赖管理：定义组件间的依赖关系
- 定制化部署：针对不同用户提供不同组件组合

```cmake
# 定义可执行文件属于 runtime 组件
install(TARGETS relaxed_ik_app
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    COMPONENT runtime
)

# 定义配置文件属于 configs 组件
install(DIRECTORY configs/
    DESTINATION .
    COMPONENT configs
)

# 定义文档属于 docs 组件
install(FILES README.md LICENSE.txt
    DESTINATION docs
    COMPONENT documentation
)
```

构建时，指定安装目录，比如：

```bash
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -S . -D build
```

进入构建目录，即可最小化安装到指定位置，比如 /usr/local/bin 目录：

```bash
cd ./build
cmake --install . --component runtime,configs
```

定制化分包：

```bash
cpack -G DEB --component runtime   # 只打包运行时组件
cpack -G NSIS -C "runtime;configs" # 打包运行时和配置组件
```

依赖关系管理：

```cmake
set(CPACK_COMPONENT_RUNTIME_DEPENDS configs)
```

### 安装时复制资源

make install 时复制资源到指定目录。比如把 configs/ 目录复制到可执行文件同级目录。

```CMake
# 可执行文件和库等
install(TARGETS relaxed_ik_app
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/
    COMPONENT runtime
)
# 目录资源
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/configs/ 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/
    PATTERN ".gitignore" EXCLUDE
)
```

安装 TARGET 部分，用于编译不同文件的场景：
- RUNTIME 代表

```bash
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -S . -B build
cmake -DCMAKE_INSTALL_PREFIX=./install -S . -B build
```

如果需要复制文件，还可使用 FILES，类似 DIRECTORY 用法。

## 变量含义

- `${CMAKE_INSTALL_PREFIX}`: 安装目录，默认为 /usr/local
- `${PROJECT_SOURCE_DIR}`: 项目源代码目录
- `${PROJECT_BINARY_DIR}`: 项目编译目录
- `${PROJECT_NAME}`: 项目名
- `${CMAKE_CURRENT_LIST_DIR}`: 当前 .cmake 文件所在目录
- `${CMAKE_CURRENT_SOURCE_DIR}`: 当前 CMakeLists.txt 文件所在目录。在 add_subdirectory 中，${CMAKE_CURRENT_SOURCE_DIR} 对应子目录的 CMakeLists.txt 文件所在目录
- `${CMAKE_CURRENT_BINARY_DIR}`: 构建当前 CMakeLists.txt 文件所在目录，比如 build 目录
- `${CMAKE_MODULE_PAHT}`: find_package 为 module 模式时，Find{{PackageName}}.cmake 文件所在路径
- `${CMAKE_PREFIX_PATH}`: 搜索路径，find_package(), find_library(), find_path() 和 find_file() 依据此路径查找

## Ref and Tag