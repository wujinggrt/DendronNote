#!/usr/bin/bash or zsh
# source ./zsh_setup.sh
# 在镜像中，默认 root 用户，并且没有 sudo 命令，所以需要删除此脚本的所有 sudo，操作如下：
# perl -i.bak -wple 's/sudo\s+//g' ./zsh_setup.sh
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

# 设置 .zshrc
LOCAL_PROFILE="$HOME/.local_profile"
cat >> ~/.zshrc <<- EOF
if [[ -f $LOCAL_PROFILE ]]; then
        . $LOCAL_PROFILE
fi
EOF

cat >> $LOCAL_PROFILE <<- EOF
# ctrl+u clear cursor before
bindkey '^U' backward-kill-line
bindkey '^]' autosuggest-accept
# 使用英文终端
export LANG=en_US.UTF-8\nexport LANGUAGE=en_US:en

alias tnew='tmux new -s'
alias tls='tmux ls'
alias tat='tmux attach -t'

# docker
alias drr='docker run -it --rm'
alias de='docker exec -it'
alias ds='docker stop'
alias drm='docker rm'

alias pe='perl -wlne'

alias gS='git status'
alias gs='git stage'

# 开启系统代理
function proxy_on() {
        # proxy_on
        # proxy_on {{prot#}}
        local port=7890
        if [[ $# -eq 1 && $1 =~ ^[0-9]+$ ]]; then
                port=$1
        fi
        export http_proxy=http://127.0.0.1:$port
        export https_proxy=http://127.0.0.1:$port
        export no_proxy=127.0.0.1,localhost
        export HTTP_PROXY=http://127.0.0.1:$port
        export HTTPS_PROXY=http://127.0.0.1:$port
        export NO_PROXY=127.0.0.1,localhost
        echo -e "\033[32m[√] 已开启代理，端口 $port\033[0m"
}

# 关闭系统代理
function proxy_off(){
        unset http_proxy
        unset https_proxy
        unset no_proxy
        unset HTTP_PROXY
        unset HTTPS_PROXY
        unset NO_PROXY
        echo -e "\033[31m[×] 已关闭代理\033[0m"
}

export THU_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
export PATH=$PATH:/usr/local/lib
export HF_DATASETS_CACHE="/data1/wj_24/.cache/huggingface/datasets"
export HF_HOME="/data1/wj_24/.cache/huggingface"
# 个人路径根据常用状态设置
# export HOME="/data1/wj_24"

export HF_TOKEN="*******************"
#export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH


export WANDB_API_KEY="*********************"
EOF

sudo chsh -s $(which zsh)
# 下一次进终端便可配置 zsh 的 powerlevel10k 外观
