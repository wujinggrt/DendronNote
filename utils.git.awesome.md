---
id: lrfk6147jm7jwvejhs6zha5
title: Awesome
desc: ''
updated: 1744462267639
created: 1744461965642
---

## 常见使用命令

使用 --no-ff 版本更常见。使用如下命令即可：

```bash
# 创建新的 branch
alias gcb="git checkout -b"

# 提交 commit
alias gcm="git commit --no-verify -m"

# 切换 branch
alias gc="git checkout"

# 推送本地新创建的 branch 到远端
alias gpso='git push --set-upstream origin "$(git symbolic-ref --short HEAD)"'

# 推送本地 branch 到远端
alias gp="git push"

# 查看本地 branch
alias gb="git branch"

# 拉取远端 branch
alias gpl="git pull --no-ff --no-edit"

# 添加所有文件
alias ga="git add -A"

# 设置远端 branch
alias gbst="git branch --set-upstream-to=origin/"

# 查看 commit 树
alias glg="git log --graph --oneline --decorate --abbrev-commit --all"

# 查看 commit 表格
alias gl="git log"
```

## Ref and Tag