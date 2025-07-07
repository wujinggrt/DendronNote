#!/bin/bash
# 执行 source ./zsh_setup.sh 即可安装
# 为了让终端正确显示 icon，建议安装等宽字体
# 下载后选一个，选一个字体即可，推荐安装 icon 现实最全的： CodeNewRoman Nerd Font
# https://github.com/ryanoasis/nerd-fonts/releases/download/v3.4.0/CodeNewRoman.zip
#
# 配置终端显示样式：成功安装后，p10k configuration 会进入配置脚本
# ctrl+r/f: 关键字搜索历史和前向命令
#
# 命令推荐和补全工具：终端的光标移动到哪儿，建议接收到哪儿
# 比如，使用右键，或 ctrl+e 或 end 键，光标移动到当前行的末尾，代表接收全部建议
# 使用 alt+f，光标前移一个单词，代表接受一个单词，下一个单词可以继续接受或继续提示
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
# ctrl+r 模糊搜索历史命令
git clone https://gitee.com/testbook/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
# 命令推荐和补全工具，用法是终端的光标移动到到哪儿，建议接收到哪儿
git clone https://gitee.com/qiushaocloud/zsh-autosuggestions ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
# 更加丰富的高亮
# 展示自动补全历史，提示参数信息，最近使用文件夹
git clone https://gitee.com/wangl-cc/fast-syntax-highlighting ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/fast-syntax-highlighting
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

chsh -s $(which zsh)
# 下一次进终端便可配置 zsh 的 powerlevel10k 外观