---
id: yp5eibjkz0etviz1sy8kae7
title: 在插入模式返回普通模式时的输入法中英切换问题
desc: ''
updated: 1740421773573
created: 1740419087154
---

我使用了 VsCode，引入了 vim 插件。经常使用 markdown 文档做笔记，在中文与英文之间切换有些许不便。比如，在 vim 的 insert 模式下，我需要输入中文。但是返回 normal 模式时，输入法可能没有切换会英文状态，从而不能用 vim 的一些快捷键。我该如何配置来解决？

## 绑定快捷键

在返回 normal 模式时，通过 escape 起作用，但是不太好用。

```json
{
  "key": "escape",
  "command": "workbench.action.terminal.sendSequence",
  "args": { "text": "/usr/local/bin/im-select com.apple.keylayout.US\n" },
  "when": "editorTextFocus && vim.active && vim.mode == 'Insert'"
}
```

## im-select 方案

### Linux

使用命令 `fcitx-remote -s <输入法ID>` 切换输入法。

### Windows

[Github: im-select](https://github.com/daipeihust/im-select) 指出，windows 平台的 im-select.exe 不能在 cmd 和 powershell 运行。推荐使用 git-bash。在 Git Bash 查看 $PATH 环境变量，把 im-select.exe 放在能够运行的地方即可，比如放到了 `/c/User/10945` 用户目录。

```bash
# 输出类似 2052（中文）或 1033（英文）。
/c/User/10945/im-select.exe

# 在中英文切换，类似 shift
/c/User/10945/im-select.exe locale
```

```json
    // 在 vim 输入时，从 insert 模式切换回 normal 时，自动切换输入法为英文。
    // {im} 会被替换为 defaultIM 的值（即 1033），从插入模式返回普通模式时，会切换输入法到键盘。
    // 如果需要切换到其他窗口，比如浏览器，需要输入时，使用 Win + 空格 即可切换输入法。
    "vim.autoSwitchInputMethod.enable": true,
    "vim.autoSwitchInputMethod.defaultIM": "1033", // 2052 代表中文，1033 代表英文
    "vim.autoSwitchInputMethod.obtainIMCmd": "C:\\Users\\10945\\im-select.exe",
    "vim.autoSwitchInputMethod.switchIMCmd": "C:\\Users\\10945\\im-select.exe {im}",
```

## Ref and Tag

#Vim