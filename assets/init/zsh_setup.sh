#!/usr/bin/bash or zsh
# source ./zsh_setup.sh
sudo apt update && sudo apt install -y zsh
# ohmyzsh
# sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
echo Y | sh -c "$(curl -fsSL https://gitee.com/pocmon/ohmyzsh/raw/master/tools/install.sh)"
# p10k configuration # 配置终端
git clone --depth=1 https://gitee.com/romkatv/powerlevel10k ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
git clone https://gitee.com/testbook/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
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

# 设置
cat >> ~/.zshrc <<- EOF
if [[ -f ~/.local_profile ]]; then
        source ~/.local_profile
fi
bindkey '^]' autosuggest-accept
EOF
# 使用英文终端
echo -e "export LANG=en_US.UTF-8\nexport LANGUAGE=en_US:en" >> ~/.zshrc
# PASSWORD= # required
# echo "$PASSWORD" | chsh -s $(which zsh)
# 下一次进终端便可配置 zsh 的 powerlevel10k 外观
