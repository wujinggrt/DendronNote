---
id: 7d8vhioct7wag76b0fqz8me
title: Systemd_服务配置和启动等设置
desc: ''
updated: 1746522819699
created: 1744559529402
---

以前版本使用 System V 作为 init program，现在，systemd 已经完全取代了 System V。优势：
- 配置更直观和方便。
- 依赖是显示而清晰的。
- 设置权限和资源限制更方便，系统更安全。
- 可检测，按要求自动重启。
- 支持并行启动，加快开机速度。

## Targets, Services, and Units

有三个重要概念：
- **unit**: systemd 的基本配置单元，描述一个服务、套接字、设备、挂载点等。每个 unit 是一个配置文件，通常位于 `/etc/systemd/system/` 或 `/lib/systemd/system/` 目录下。
- **service**: 一种特殊类型的 unit，表示一个系统服务。通常是一个可启动和停止的 daemon。服务可以是守护进程、后台任务或其他需要在系统启动时运行的程序。
- **target**: 一组 unit 的集合，也是 unit，类似于 System V 中的 runlevel。可以将多个 unit 组合在一起，以便在启动时同时启动它们。可以在 `/etc/systemd/system/` 和 `/usr/lib/systemd/system/` 中查看例子，target 文件几乎不会包含 [Service] 部分，而是组织多个 unit，启动多个服务。

对比 service 和 target：

功能:
- service 文件用于定义和管理具体的服务。
- target 文件用于组织和管理一组单元，表示系统的某个状态或功能。

依赖关系:
- service 文件可以依赖于target文件。例如，一个服务可能需要在network.target之后启动。
- target 文件可以包含多个service文件。例如，multi-user.target包含多个服务，表示系统处于多用户模式。

启动顺序:
- service文件通过After、Before等指令定义启动顺序。
- target文件通过包含和依赖关系定义系统的启动顺序和状态。

### Units

Unit 文件通常在三个目录中：
- `/etc/systemd/system/`：用户自定义的 unit 文件，优先级最高。
- `/run/systemd/system/`：运行时生成的 unit 文件，优先级次之。
- `/lib/systemd/system/`：系统默认的 unit 文件，优先级最低。有时会在 `/usr/lib/systemd/system` 目录。

当 systemd 启动一个 unit 时，它会读取这些目录中的 unit 文件，并根据它们的配置启动相应的服务。根据优先级查找，查到优先级最高的执行便停止，优先级低的不会考虑。

最佳实践：如果想要覆盖系统默认 unit 的行为，在 /etc/systemd/system 目录下替换同名文件即可。如果禁用某个 unit，在 /etc/systemd/system 目录下创建一个同名的空文件，或者链接到 /dev/null。

所有 unit 文件都以 `.service`、`.socket`、`.mount` 等后缀名结尾，表示它们的类型。每个 unit 文件包含多个部分，每个部分以方括号开头，表示该部分的配置项。

所有 unit 文件以 `[Unit]` 开头，包含 unit 的描述信息和依赖关系。接下来是 `[Service]` 部分，包含服务的启动和停止配置。最后是 `[Install]` 部分，包含安装和启用该 unit 的配置。

以 D-Bus 服务为例，参考 /lib/systemd/system/dbus.service 文件：

```ini
[Unit]
Description=D-Bus System Message Bus
Documentation=man:dbus-daemon(1)
Requires=dbus.socket
```

可以看出，Requires 关键字指出了依赖 dbus.socket unit。告诉 systemd，启动此服务时，要创建一个本地 socket。

#### 常见关键字

如果需要启动多个服务，并且要求启动顺序，通过如下关键字表达：
- `Requires=`: 如果依赖的 unit 启动失败，则当前 unit 也会失败。
- `Wants=`: 如果依赖的 unit 启动失败，则当前 unit 不会失败。
- `Conflicts=`: 指定当前 unit 与依赖的 unit 互斥，不能同时运行。

以上三个最为常用。称为 output dependencies。

- `Before=`: 指定当前 unit 在依赖的 unit 之前启动。
- `After=`: 指定当前 unit 在依赖的 unit 之后启动。

以上两个是 incoming dependencies。

上面三个可以与下面两个组合使用。

### Services

服务可以开启一个 daemon，通常是一个 unit 文件，以 `.service` 结尾。服务的配置文件通常包含以下部分除了 Unit 文件需要的 `[Unit]` 部分外，还包含 `[Service]` 部分。以 lighttpd.service 为例：

```ini
[Service]
ExecStart=/usr/sbin/lighttpd -f /etc/lighttpd/lighttpd.conf -D
ExecReload=/bin/kill -HUP $MAINPID
```

#### 常见关键字

`Type={{Option}}`，默认有如下选择：
- `simple`: 默认值，服务在启动时不会 fork 进程，直接执行 ExecStart 指令作为主进程，systemd 会等待服务进程结束。适用于无需后台化的前台进程。若退出则代表服务结束。
- `forking`: 服务是一个**守护进程**（不指出代表在前台工作），ExecStart 指定了可执行文件的路径，通过 fork() 创建进程来执行。systemd 会追踪子进程作为服务的主进程。比如，启动的服务会执行命令挂到后台，然后马上退出。如果使用 simple 则不合适。

关键区别​​：
- simple 适用于前台运行的服务（如 nginx -g 'daemon off;'）。
- forking 适用于传统守护进程（如默认配置的 nginx 或 mysqld）。

`ExecStart=/path/to/executable`，指定服务的可执行文件路径。可以有多个 ExecStart 指令，systemd 会依次执行。

`Restart={{Option}}`，指定服务的重启策略。选项：no, always, on-failure、on-abort、on-success、on-watchdog。


### Targets

target 是一组 unit 的集合，可以将多个 unit 组合在一起，以便在启动时同时启动它们。目标通常以 `.target` 结尾。以 `multi-user.target` 为例：

```ini
[Unit]
Description=Multi-User System
Documentation=man:systemd.special(7)
Requires=basic.target
Conflicts=rescue.service rescue.target
After=basic.target rescue.service rescue.target
AllowIsolate=yes
```

#### 常见关键字

`WantedBy=`，指定当前 unit 的安装目标。表示在启动时自动启动此 unit。

### systemd 管理各个部分

内核启动 /sbin/init 后，启动 systemd。systemd 程序是一个符号链接，指向 /lib/systemd/systemd。systemd 读取 `/usr/lib/systemd/system/default.target` 文件，启动默认的 target。默认的 target 是 `graphical.target`，表示图形界面。可以使用 `systemctl get-default` 命令查看当前的默认 target。

```bash
systemctl get-default
```

可以使用 `systemctl set-default` 命令设置默认 target，例如：

```bash
systemctl set-default multi-user.target
```

如果是终端登录，那么启动的是 `multi-user.target`，如果是图形界面登录，则启动 `graphical.target`。注意，其中 graphical.target 依赖于 multi-user.target。graphical.target 内容如下：

```ini
[Unit]
Description=Graphical Interface
Documentation=man:systemd.special(7)
Requires=multi-user.target
Wants=display-manager.service
Conflicts=rescue.service rescue.target
After=multi-user.target rescue.service rescue.target display-manager.service
AllowIsolate=yes
```

可以看到是在 multi-user.target 等之后启动。

查看所有服务：

```bash
systemctl list-units --type=service
```
查看所有目标：

```bash
systemctl list-units --type=target
```

### 添加自己的服务

创建一个新的 unit 文件，通常位于 `/etc/systemd/system/` 目录下。文件名以 `.service` 结尾，例如 `simpleserver.service`。文件内容如下：

```ini
[Unit]
Description=Simple server

[Service]
Type=forking
ExecStart=/usr/bin/simpleserver
Restart=on-abort

[Install]
WantedBy=multi-user.target
```

[Unit] 部分使得 systemctl 可以列出此服务。

[Install] 部分指定了服务的安装目标，表示在 multi-user.target 启动时自动启动此服务。WantedBy 关键字避免了在 multi-user.target 中写出 After 来启动此服务，如此更加灵活。

可以使用如下命令控制：
- `systemctl start simpleserver` 命令启用此服务
- `systemctl stop simpleserver` 命令停止服务，通常是杀死服务的进程
- `systemctl enable simpleserver` 命令添加服务到启动项中，使其开机启动。如果有 WantedBy 字段，会在对应的 unit 的 wants 目录下创建一个符号链接，指向此 service 文件，方便启动
- `systemctl disable simpleserver` 命令删除服务

注意，如果需要在 `systemctl stop` 运行某些程序，比如 `docker stop webdav`，那么需要在 service 文件中使用:
```ini
ExecStop=/usr/bin/docker stop webdav
```

## 比较 After 和 WantedBy

它们分别用于控制服务的​​启动关联性​​和​​启动顺序​​。以下是两者的核心区别：

**After** 位于 `[Unit]` 部分，定义服务**启动顺序**，指出当前服务在哪些服务之后启动。比如：
```ini
After=network.target
```

表示在网络服务初始化完成后启动。注意，不会处理依赖关系。

**WantedBy** 位于 `[Install]` 部分，定义服务**启动关联性**，指明当前服务跟随哪个系统目标（target）**自动启动**。比如：

```ini
WantedBy=multi-user.target
```

表示当系统进入多用户模式后，服务会自动激活，通过其 wants 目录来自动加载。比如，使用 `systemctl enable myservice` 命令后，系统会在 `/etc/systemd/system/multi-user.target.wants` 目录下创建符号链接，指向到 myservice 这个 unit 的 service 文件，方便加载。

After 的常见用途​​：
- 确保网络就绪后启动网络相关服务（如 After=network.target）。
- 控制服务间启动顺序（如数据库服务先于 Web 服务启动）。

​​WantedBy 的常见值​​：
- multi-user.target：多用户命令行模式。
- graphical.target：图形界面模式。
- default.target：系统默认目标

通过合理组合两者，可实现服务按需启动并确保依赖顺序，例如在系统进入特定运行级别时，按顺序加载关键服务。

| ​​配置项​​ | ​​作用范围​​ | ​​主要功能​​                     | ​​生效阶段​​            |
| ---------- | ------------ | -------------------------------- | ----------------------- |
| After      | [Unit]       | 定义启动顺序（时序控制）         | 服务启动时生效          |
| WantedBy   | [Install]    | 定义自动启用的目标（关联性控制） | systemctl enable 时生效 |

## systemctl 命令

```bash
# 有时候服务停止没有响应，服务停不下来，这时候就需要将进程kill掉了
systemctl kill service_name
# 查看配置文件
systemctl cat service_name
```

## 例子

```ini
[Unit]
Description=My Service
After=postgresql.service  # 在 PostgreSQL 服务之后启动

[Service]
ExecStart=/usr/bin/my-service

[Install]
WantedBy=multi-user.target  # 随多用户模式自动启用
```

在此配置中，服务会在 postgresql.service 启动后运行，且通过 systemctl enable 后，会关联到 multi-user.target 的启动流程

### frpc 服务配置

```ini
[Unit]
# 服务名称，可自定义
Description = frp server
After = network.target syslog.target
Wants = network.target

[Service]
TimeoutSec=300
Restart=always
#ExecStartPre=/bin/sleep 90
Type = simple
# 启动frps的命令，需修改为您的frps的安装路径
ExecStart = /usr/soft/frp_0.51.3_linux_386/frpc -c /usr/soft/frp_0.51.3_linux_386/frpc.ini

[Install]
WantedBy = multi-user.target
```

## Ref and Tag

https://www.cnblogs.com/yanjingge888/articles/18501514