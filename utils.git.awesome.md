---
id: lrfk6147jm7jwvejhs6zha5
title: Awesome
desc: ''
updated: 1748425254692
created: 1744461965642
---

## git pull

拉取远程仓库的特定分支：

```bash
git clone -b <分支名> <仓库URL>
```

例子：

```bash
git clone -b CVPR-Challenge-2025-Round1 git@github.com:TianxingChen/RoboTwin.git
```

    (
        model_args,
        data_args,
        training_args,
        action_head_args,
        model_config,
        bnb_model_from_pretrained_args,
    ) = parse_param()
    config = {
        "model_args": model_args,
        "data_args": data_args,
        "training_args": training_args,
        "action_head_args": action_head_args,
        "bnb_model_from_pretrained_args": bnb_model_from_pretrained_args,
    }

## git stash: 暂存修改

### 用法

保存**当前**分支下的修改：

```bash
git stash # 保存当前context，并将所有文件回退到repo的HEAD
git stash -u # 包含未追踪文件，默认不包含
git stash list # 列出暂存的内容
```

如果想要切换分支，且暂时不想提交工作，git stash 之后再切换是个最佳实践。此时修改存储在栈上，可以用 git stash list 查看。

```bash
git stash apply
git stash apply stash@{1} # 指定某个存储
```

重新应用 stash 的文件。可以在一个分支上保存一个贮藏，切换到另一个分支，然后尝试重新应用这些修改。 

git stash apply ​​不会自动恢复暂存状态​​。如果需要恢复存储的修改时，还原暂存区（stage/index）的状态。

git stash apply 不会删除栈上的存储，如果需要同时删除，使用 git stash pop 完成。

```bash
git stash pop
git stash pop stash@{1}
```

### 最佳实践

#### 临时切换分支

暂存当前修改，切换分支：

```bash
git stash
git switch other-branch
# 处理其他任务后返回原分支
git switch original-branch
git stash pop
```

#### 拉取远程仓库

本地有未提交的修改，但需要拉取远程最新代码时，可以解决：

```bash
git stash
# 此时仓库干净，可以拉取
git pull
git stash pop
```

如果恢复时有冲突，需要手动解决冲突。

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