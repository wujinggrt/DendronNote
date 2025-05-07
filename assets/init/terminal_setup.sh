#!/usr/bin/bash or zsh
# 执行使用 . ./terminal_setup.sh，随后用到 sudo 的命令会提示输入 sudo 密码，输入即可
# 如果用户是 root，不会要求提醒输入密码，但会直接安装到 root 用户
# 在镜像中，默认 root 用户，并且没有 sudo 命令，所以需要删除此脚本的所有 sudo，操作如下：
# perl -i.bak -wple 's/sudo\s+//g' ./terminal_setup.sh
UBUNTU_VERSION=$(lsb_release -d | perl -wnlae '$F[2] =~ /(\d\d\.\d\d)\.\d+/ and print $1')
EMAIL_ADDRESS=wujinggrt@qq.com
USER=wj-24
HOME=/home/${USER}
# 配置
ZEROTIER= # 设置此变量为任意值，只要长度非 0，则安装 ZEROTIER
MINICONDA3=1 
DOCKER=1
NVIDIA_GPU=1 # 如果没有，与 NVIDIA 相关配置都不会执行
NVIDIA_CONTAINER_TOOLKIT=1 # 设置此变量为任意值，只要长度非 0，安装NVIDIA driver
# 应该先试用 ubuntu-drivers devices 查看建议安装版本，随后再修改
NVIDIA_DRIVER_VERSION=550
# authorization: root
if [[ $UBUNTU_VERSION == 24.04 ]]; then
    # 24.04换源
    cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.bak
    perl -i.bak -wple 's@(URIs:) http://(archive|security).ubuntu.com/ubuntu/@$1 http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g;' \
    /etc/apt/sources.list.d/ubuntu.sources
elif [[ $UBUNTU_VERSION == 22.04 ]]; then
    # 22.04 换源
    sudo sed -i "s@http://.*archive.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
    sudo sed -i "s@http://.*security.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
else
    echo "ERROR: Unsupported Ubuntu version: $UBUNTU_VERSION"
    exit 1
fi
sudo apt update
sudo apt install -y wget curl expect tmux
# Zerotier
if [[ -n $ZEROTIER ]]; then
    curl -s https://install.zerotier.com | bash
    sudo systemctl enable zerotier-one.service
    sudo systemctl start zerotier-one.service
fi
# SSH
sudo apt install -y openssh-server
# 开机启动
# 也可以用service sshd start，这是老一点的系统
sudo systemctl start ssh
sudo systemctl enable ssh
# 登录别名设置如下
# 在 .ssh/config 中配置（~/.ssh/config)
# Host 别名
#     Hostname 主机名
#     Port 端口 # 可选
#     User 用户名 
# 使用公钥让ssh登录免密
# 直接拷贝公钥到链接机器上，在将拷贝的 id_rsa.pub 内容添加到 .ssh/authorized_keys 里面（不存在则创建），比如：
# scp .ssh/id_rsa.pub username@hostname:SomeDir
# 随后登录到此机器，并：
# cat ~/id_rsa.pub >> .ssh/authorized_keys
# 修改服务器端的 sshd 配置，编辑 /etc/ssh/sshd_config 如下
# PubkeyAuthentication yes
# PasswordAuthentication no
# 重启服务器的 ssh 服务
# $ sudo systemctl restart sshd.service
# 或
# $ sudo service sshd stop
# $ sudo service sshd start
# 为了github能够在ssh传输：
mkdir -p $HOME/.ssh
# 需要按回车键
ssh-keygen -t rsa -f ${HOME}/.ssh/id_rsa -N ""
echo -e "yes\n" | ssh -T git@github.com
# utils
mkdir -p $HOME/.local/share && apt install -y tldr
# Asia/Shanghai, CMake需要此包
echo 1 70 | apt install -y tzdata
sudo apt install -y cppman iputils-ping net-tools iproute2 make cmake htop vim cmake unzip tree x11-apps
# git设置
sudo apt install -y git
# git config --global user.name $USER
# git config --global user.email $EMAIL_ADDRESS
# miniconda
# https://docs.anaconda.com/miniconda/install/
if [[ -n $MINICONDA3 ]]; then
    mkdir -p $HOME/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda3/miniconda.sh
    bash $HOME/miniconda3/miniconda.sh -b -u -p $HOME/miniconda3
    rm $HOME/miniconda3/miniconda.sh
    # After installing, close and reopen your terminal application or refresh it by running the following command:
    # source miniconda3/bin/activate
    # To initialize conda on all available shells, run the following command:
    # conda init --all
    # 也可以使用如下设置，一劳永逸
    # ~/miniconda3/bin/conda init zsh # 最后也可以是 bash
fi
# 安装Docker
if [[ -n $DOCKER ]]; then
    wget http://fishros.com/install -O fishros && echo 8 | . fishros
    # 开机启动
    sudo systemctl enable --now docker > /dev/null 2>&1
    # docker
    # nvidia runtime和Docker
    if [[ -n $NVIDIA_GPU ]]; then
        # Win下装过驱动后便可忽略driver
        # 应该先试用 ubuntu-drivers devices 查看建议安装版本，随后再修改
        sudo apt install -y nvidia-driver-${NVIDIA_DRIVER_VERSION} nvidia-settings nvidia-prime
        if [[ $NVIDIA_DRIVER_VERSION -ne 550 ]]; then 
            echo "ERROR 驱动版本可能不适合当前的 CUDA Toolkit"
            exit 1
        fi
        # CUDA Toolkit for 550
        wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run 
        sudo sh cuda_12.6.3_560.35.05_linux.run --silent --toolkit
        # cuDNN 需要先登录才能下载和安装，参考：
        # https://developer.nvidia.com/cudnn
        # 下面是安装
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        sudo apt-get -y install cudnn
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
        # 安装后注意修改环境变量
        if [[ -n $NVIDIA_CONTAINER_TOOLKIT ]]; then
            # 虚拟化套件
            # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
            # Installing the NVIDIA Container Toolkit — NVIDIA Container Toolkit 1.16.2 documentation
            # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
            && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            sudo apt-get update
            sudo apt-get install -y nvidia-container-toolkit
            sudo systemctl restart docker
            # docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi # 验证安装
            # 使用英伟达命令行工具
            # 执行以下命令会修改/etc/docker/daemon.json：
            # $ cat /etc/docker/daemon.json 
            # {
            #     "runtimes": {
            #         "nvidia": {
            #             "args": [],
            #             "path": "nvidia-container-runtime"
            #         }
            #     }
            # }
            sudo nvidia-ctk runtime configure --runtime=docker
            sudo systemctl restart docker
            sudo systemctl enable docker # 开机自启动
            # 使用所有GPU 
            # $ docker run --gpus all 【镜像名】 
            # 使用两个GPU 
            # $ docker run --gpus 2 【镜像名】 
            # 指定GPU运行 
            # $ docker run --gpus '"device=1,2"' 【镜像名】
            # ssh -L 6006:127.0.0.1:6666 username@serverIP
        fi
    fi
fi

# 初始假设默认是 Bash
shell_name='bash'
if shopt -u lastpipe 2> /dev/null; then
    # 当前 shell 是 Bash，下面的 : 相当于 Python pass
    :
else
    # 当前 shell 是 Zsh 或其他 shell
    if [[ -n $ZSH_VERSION ]]; then
        shell_name='zsh'
    else
        # 当前使用的 shell 不是 Bash 或 Zsh
        shell_name=''
    fi
fi

# 根据 shell 名称加载相应的配置文件
if [ "$shell_name" = "bash" ]; then
    if [ -f ~/.bashrc ]; then
        . ~/.bashrc
    fi
elif [ "$shell_name" = "zsh" ]; then
    if [ -f ~/.zshrc ]; then
        . ~/.zshrc
    fi
fi
