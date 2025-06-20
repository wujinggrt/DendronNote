#!/bin/bash
# or zsh
# source ./zsh_setup.sh
# 注意，需要安装等宽字体，比如 NerdFont
# https://github.com/ryanoasis/nerd-fonts
# 下载后选一个双击图标，便可 install，在终端选择此字体即可。
# 在镜像中，默认 root 用户，并且没有 sudo 命令，所以需要删除此脚本的所有 sudo，操作如下：
# perl -i.bak -wple 's/sudo\s+//g' ./zsh_setup.sh
SUDO=''
if [[ $UID -ne 0 ]]; then
    SUDO='sudo'
fi
$SUDO apt update
$SUDO apt install -y zsh git tmux
# ohmyzsh
# sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
echo Y | sh -c "$(curl -fsSL https://gitee.com/pocmon/ohmyzsh/raw/master/tools/install.sh)"
# p10k configuration # 配置终端
git clone --depth=1 https://gitee.com/romkatv/powerlevel10k ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
git clone https://gitee.com/testbook/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
# 在 .bashrc 或 .zshrc 配置接受命令：bindkey '^ ' autosuggest-accept，功能是 ctrl+空格 接受
# 光标移动到到哪儿，建议接收到哪儿。使用 ctrl+e 或 end，接收全部；使用 alt+f，接收单词。
git clone https://gitee.com/qiushaocloud/zsh-autosuggestions ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
# 更加丰富的高亮
git clone https://gitee.com/wangl-cc/fast-syntax-highlighting ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/fast-syntax-highlighting
# 展示自动补全历史，提示参数信息，最近使用文件夹
# 
# ctrl+n/p可以上下选择，随后回车选中选择的
# tab
# ctrl+r或s，分别对应最近到以前和以前到最近的顺序，可以用两个词的fuzzy search
# autocomplete 目前是高危插件，不建议
perl -i.bak -wple 's/(ZSH_THEME=)(.*)$/$1"powerlevel10k\/powerlevel10k"/g;' ~/.zshrc
perl -i.bak -wple 's/(plugins=)(.*)$/$1\(git zsh-autosuggestions z zsh-syntax-highlighting fast-syntax-highlighting\)/g;' ~/.zshrc

# 设置 .zshrc
LOCAL_PROFILE="$HOME/.local_profile"
cat >> ~/.zshrc <<- EOF
if [[ -f $LOCAL_PROFILE ]]; then
        . $LOCAL_PROFILE
fi
EOF

$SUDO chsh -s $(which zsh)
# 下一次进终端便可配置 zsh 的 powerlevel10k 外观