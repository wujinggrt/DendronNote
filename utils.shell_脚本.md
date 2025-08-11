---
id: srkdle9fm56heprkeiuuqri
title: Shell_脚本
desc: ''
updated: 1754065310000
created: 1754060661732
---

## 执行命令

### source 与 .

在当前 Shell 进程，执行脚本。典型的场景为当前的终端，还有新开的进程。

最佳实践：在交互式命令行中使用 source，在脚本中使用 `.`。

### bash 和 sh

`bash ./run.sh` 或直接执行 `./run.sh`，系统会创建新的 Shell 进程，不会影响当前环境。

使用 source 或 . 运行脚本时，`$0` 会解为 bash；使用 bash 或直接运行时，则解析为脚本的相对或绝对路径名，通常可以配合 dirname $0 等方式获取绝对路径。

## 变量

惯例，赋值是不用 `$`，访问值时用 `$VAR`。

### 参数

`$0` 代表本脚本名称，用于决定绝对路径。

注意，运行脚本会启动新的进程，脚本中的 cd 切换工作目录，不会影响调用的 shell 场景，仅影响实际运脚本的进程。所以有时候会使用 `cd $(dirname) && pwd` 等方式获取绝对路径。

### 绝对路径

```bash
# 可能要安装包
realpath {{相对路径等}}
# 兼容性更好的方法
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
# 或者
SCRIPT_DIR=$(dirname $(readlink -f $0))
```

### 数组

访问某个目录下的所有内容：

```bash
# 使用 ($(commands)) 获取数组是常见的技巧
FILES=($(find . -name "*yaml"))
for yaml in ${FILES[@]}; do
  ...
done
```

```bash
declare -a arr
arr=(12 45)
echo ${arr[0]}
echo ${arr[1]}
```

获取数组长度 `${#array[@]}`。

#### subscripts * 和 @ 处理展开数组的所有元素

`${array[*]}`，`{$array[@]}` 将所有内容通过 IFS 分割成各元素。`"${array[*]}"` 将所有内容拓展为一行，`"{$array[@]}"` 将所有内容按照原传入的，通过 IFS 分割成各元素。

```bash
animals=("a dog" "a cat" "a fish") 
for i in ${animals[*]}; do echo $i; done 
```

结果如下：

```
a 
dog 
a 
cat 
a 
fish 
```

结果同上：

```bash
for i in ${animals[@]}; do echo $i; done
```

拓展，展平到一行

```bash
for i in "${animals[*]}"; do echo $i; done
a dog a cat a fish 
```

分开：

```bash
for i in "${animals[@]}"; do echo $i; done 
a dog 
a cat 
a fish 
```

### 关联数组（map）

声明与赋值不能合并：

```bash
declare -A fruits
fruits=([apple]="red" [banana]="yellow" [grape]="purple")
```

访问不存在的键返回空字符串（不会报错）：

```bash
echo "${user[address]}"  # 输出空行
```

### Parameter Expansion

#### 处理不存在的变量

默认值，在函数和脚本传参时使用比较频繁：

```bash
DEFAULT_VAL=hello
hello=${1:-$DEFAULT_VAL}
world=${2:-world}
```

花括号内可以是环境变量，如果当前没有，使用默认值。

### envsubst：替换对应值为环境变量值

```bash
sudo apt install -y gettext-base
```

创建模板文件 app.conf.template：

```bash
# app.conf.template
server_ip = ${SERVER_IP}
port = ${PORT}
debug = ${DEBUG_MODE:-false}  # 设置默认值
```

设置环境变量：

临时设置：

```bash
export SERVER_IP="192.168.1.100"
export PORT=8080
export DEBUG_MODE="true"
```

从 .env 文件加载（推荐）：

```bash
# .env
SERVER_IP=192.168.1.100
PORT=8080
DEBUG_MODE=true
```

加载到环境：

```bash
set -a; source .env; set +a
```


```bash
# 文件替换（输入文件，输出到标准输出），也可以重定向
envsubst < app.conf.template
```

### 特殊变量

```bash
${PWD} # 当前目录
```

`$0`: 当前脚本的文件名。
`$n（n≥1）`: 传递给脚本或函数的参数。n 是一个数字，表示第几个参数。
`$#`: 传递给脚本或函数的参数个数。
`$*`: 传递给脚本或函数的所有参数。当被双引号包围时，所有的参数被扩展并放到一起。f "word" "words with spaces"，在函数f中，`"$*"` 为 "word words with spaces", `"$@"` 为 "word" "words with spaces"
`$@`: 传递给脚本或函数的所有参数。当被双引号" "包含时，`$@` 与 `$*` 稍有不同，`$@` 保留了原有传入的语义，`$*` 则全部扩展并只有一个 string 的变量
`$?`: 上个命令的退出状态，或函数的返回值。
`$$`: 当前 Shell 进程 ID。对于 Shell 脚本，就是这些脚本所在的进程 ID。

## 工作 job 管理

```py
{{any_command}} &         # 在后台运行某命令，也可用 CTRL+Z 将当前进程挂到后台
jobs                      # 查看所有后台进程（jobs）
bg                        # 查看后台进程，并切换过去
fg                        # 切换后台进程到前台
fg {job}                  # 切换特定后台进程到前台
```

bash 终端通过 () & 语法就能开启一个线程，所以对于上面的例子，可以归并到如下一个脚本中：

```bash
(cd renderer && npm run serve | while read -r line; do echo -e "[renderer] $line"; done) &

(cd service && npm run serve | while read -r line; do echo -e "[service] $line"; done) &

wait # 等待所有当前 shell 启动的进程结束
```

## 条件判断

### test 表达式

```bash
statement1 && statement2  # and 操作符
statement1 || statement2  # or 操作符

exp1 -a exp2              # exp1 和 exp2 同时为真时返回真（POSIX XSI扩展）
exp1 -o exp2              # exp1 和 exp2 有一个为真就返回真（POSIX XSI扩展）
( expression )            # 如果 expression 为真时返回真，输入注意括号前反斜杆
! expression              # 如果 expression 为假那返回真

# String
str # str is not null
-n str # len > 0
-z str # len = 0
str1 = str2               # 判断字符串相等，如 [ "$x" = "$y" ] && echo yes
str1 == str2            # 不是POSIX的
str1 != str2              # 判断字符串不等，如 [ "$x" != "$y" ] && echo yes
str1 < str2               # 字符串小于，如 [ "$x" \< "$y" ] && echo yes
str2 > str2               # 字符串大于，注意 < 或 > 是字面量，输入时要加反斜杆
-n str1                     # 判断字符串不为空（长度大于零）
-z str1                     # 判断字符串为空（长度等于零）

# File
-<a|e> file            # 判断文件存在，如 [ -a /tmp/abc ] && echo "exists"
-d file                   # 判断文件存在，且该文件是一个目录
-f file                   # 判断文件存在，且该文件是一个普通文件（非目录等）
-L file                   # 判断文件存在，且该文件是一个Symbolic link
-r file                   # 判断文件存在，且可读
-s file                   # 判断文件存在，且尺寸大于0
-w file                   # 判断文件存在，且可写
-x file                   # 判断文件存在，且执行
-N file                   # 文件上次修改过后还没有读取过
-O file                   # 文件存在且属于当前用户
-G file                   # 文件存在且匹配你的用户组
file1 -eq file2 # same inode
file1 -nt file2           # 文件1 比 文件2 新
file1 -ot file2           # 文件1 比 文件2 旧

# Integer
num1 -eq num2             # 数字判断：num1 == num2
num1 -ne num2             # 数字判断：num1 != num2
num1 -lt num2             # 数字判断：num1 < num2
num1 -le num2             # 数字判断：num1 <= num2
num1 -gt num2             # 数字判断：num1 > num2
num1 -ge num2             # 数字判断：num1 >= num2

# 分支控制：if 和经典 test，兼容 posix sh 的条件判断语句
test {expression}         # 判断条件为真的话 test 程序返回0 否则非零
[ expression ]            # 判断条件为真的话返回0 否则非零

test "abc" = "def"        # 查看返回值 echo $? 显示 1，因为条件为假
test "abc" != "def"       # 查看返回值 echo $? 显示 0，因为条件为真
```

## 函数

### 局部变量

```bash
myfunc() {
  local local_var="仅在函数内可见"
  echo $local_var  # 正常输出
}
myfunc
echo $local_var    # 无输出（变量已销毁）
```

### 返回值作为判断

```bash
abc() {
    return 1
}
if abc; then
    echo abc
fi
```

如果想要赋值，应当使用 echo，并使用 $(()) 的形式从 stdio 获取输出。

## 循环

```bash
until [[ "$count" -gt 5 ]]; do 
    echo "$count" 
    count=$((count + 1)) 
done
```

```bash
while read distro version release; do 
    printf "Distro: %s\tVersion: %s\tReleased: %s\n" \ 
        "$distro" \ 
        "$version" \ 
        "$release" 
done < distros.txt
```

until 和 while 可以处理从 stdio 输入的数据，因此可以读文本内容。

```bash
get_panes_info() {
    tmux list-panes -s -t "$SESSION" -F "#{window_index}.#{pane_index} #{pane_current_command} #{pane_tty}"
}
while IFS=' ' read -r pane_id process tty; do
    pane_status["$pane_id"]=0  # 0=未启动, 1=已启动
    pane_process["$pane_id"]="$process"
    pane_tty["$pane_id"]="$tty"
    ((total_panes++))
done < <(get_panes_info)
```

for 循环的两种形式：

```bash
for variable [in words]; do 
    commands 
done 
# C语言风格，注意等号后没空格
# 注意，[[ expr ]] 和 (( expr )) 中的表达式，与括号前后空格分隔
for (( i=0; i+1<5; i+=2 )); do 
    echo $i 
done
```

## IO 重定向

### Here Documents

用法：

```bash
cat << EOF
<html> 
  <head> 
    <title>$TITLE</title> 
  </head> 
  <body> 
    <h1>$TITLE</h1> 
    <p>$TIMESTAMP</p>
  </body> 
</html>
EOF
```

有时候，为了美观，不需要前置空白字符（包括制表符）时，可以使用 `<<-` 的形式：

```bash
ftp -n <<- _EOF_ 
  open $FTP_SERVER 
  user anonymous me@linuxbox 
  cd $FTP_PATH 
  hash 
  get $REMOTE_FILE 
  bye 
_EOF_
```

### < 标准重定向

从文件读取数据给标准输入。

```bash
wc -l < myfile.txt
```

### << Here Document，文档内嵌入

```bash
command << DELIMITER
多行文本...
DELIMITER
```

DELIMITER 常用 EOF。默认会进行变量替换和命令替换（可用<<'DELIMITER'禁用）。

```bash
cat << 'EOF'
这里$HOME不会被展开: $HOME
EOF
```

### <<< Here String，字符串即时输入

将单个字符串作为命令的输入，比 echo "string" | command 更高效，避免创建子进程。字符串部分会替换变量和命令。

```bash
grep "hello" <<< "hello world"
name="Alice"
wc -w <<< "Hello $name"
```

### 读取 IO 和 IFS

```bash
# read-multiple.sh
#!/bin/bash
echo -n "Enter one or more values > "
read var1 var2 var3 var4 var5
```

执行脚本后，比如：

```bash
read-multiple
a b c d e
```

输入会赋值给各个 var。

IFS（Internal Field Separator），指导分割的分隔符，默认为 space，tab，newline character。可以临时修改 IFS，方便读取。比如：

```bash
FILE=/etc/passwd
file_info="$(grep "^user1:" $FILE)"
IFS=":" read user pw uid gid name home shell <<< "$file_info" 
```

如果复杂一点：

```bash
OLD_IFS="$IFS" 
IFS=":" 
read user pw uid gid name home shell <<< "$file_info" 
IFS="$OLD_IFS"
```

## Ref and Tag