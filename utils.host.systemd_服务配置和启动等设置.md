---
id: 7d8vhioct7wag76b0fqz8me
title: Systemd_服务配置和启动等设置
desc: ''
updated: 1744736860062
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
- **target**: 一组 unit 的集合，类似于 System V 中的 runlevel。可以将多个 unit 组合在一起，以便在启动时同时启动它们。

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

依赖通过如下关键字表达：
- `Requires=`: 如果依赖的 unit 启动失败，则当前 unit 也会失败。
- `Wants=`: 如果依赖的 unit 启动失败，则当前 unit 不会失败。
- `Conflicts=`: 指定当前 unit 与依赖的 unit 互斥，不能同时运行。

以上三个最为常用。称为 output dependencies。

- `Before=`: 指定当前 unit 在依赖的 unit 之前启动。
- `After=`: 指定当前 unit 在依赖的 unit 之后启动。

以上两个是 incoming dependencies。

### Services

服务时一个 daemon，通常是一个 unit 文件，以 `.service` 结尾。服务的配置文件通常包含以下部分除了 Unit 文件需要的 `[Unit]` 部分外，还包含 `[Service]` 部分。以 lighttpd.service 为例：

```ini
[Service]
ExecStart=/usr/sbin/lighttpd -f /etc/lighttpd/lighttpd.conf -D
ExecReload=/bin/kill -HUP $MAINPID
```

#### 常见关键字

`Type={{Option}}`，默认有如下选择：
- simple: 默认值，服务在启动时不会 fork 进程，直接执行 ExecStart 指令作为主进程，systemd 会等待服务进程结束。适用于无需后台化的前台进程。若退出则代表服务结束。
- forking: 服务是一个守护进程（不指出代表在前台工作），ExecStart 指定了可执行文件的路径，通过 fork() 创建进程来执行。systemd 会追踪子进程作为服务的主进程。

`ExecStart=/path/to/executable`，指定服务的可执行文件路径。可以有多个 ExecStart 指令，systemd 会依次执行。

`Restart={{Option}}`，指定服务的重启策略。选项：on-failure、on-abort、on-success、on-watchdog。


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

内核启动 /sbin/init 后，启动 systemd。systemd 程序是一个符号链接，指向 /lib/systemd/systemd。systemd 读取 `/etc/systemd/system/default.target` 文件，启动默认的 target。默认的 target 是 `graphical.target`，表示图形界面。可以使用 `systemctl get-default` 命令查看当前的默认 target。

```bash
systemctl get-default
```

可以使用 `systemctl set-default` 命令设置默认 target，例如：

```bash
systemctl set-default multi-user.target
```

如果是终端登录，那么启动的是 `multi-user.target`，如果是图形界面登录，则启动 `graphical.target`。注意，其中 graphical.target 依赖于 multi-user.target。

查看所有服务：

```bash
systemctl list-units --type=service
```
查看所有目标：

```bash
systemctl list-units --type=target
```

### 添加自己的服务

创建一个新的 unit 文件，通常位于 `/etc/systemd/system/` 目录下。文件名以 `.service` 结尾，例如 `sympleserver.service`。文件内容如下：

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

随后使用 `systemctl start simpleserver` 命令启用此服务。停止服务使用 `systemctl stop simpleserver` 命令。最后，使用 `systemctl enable simpleserver` 命令将服务添加到启动项中，使其开机启动。可以使用 `systemctl disable simpleserver` 命令删除服务。


## Ref and Tag