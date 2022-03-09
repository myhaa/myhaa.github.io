---
title: 深度学习之TensorFlow安装
author: Myhaa
top: false
cover: false
toc: true
mathjax: false
categories: 深度学习
tags:
  - 安装教程
date: 2021-03-20 11:56:00
img:
coverImg:
password:
summary: 深度学习之TF安装
---



# 参考

* [windows tensorflow-gpu的安装](https://zhuanlan.zhihu.com/p/35717544)
* [官网](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)

# windows安装

## 先更新显卡驱动

* 右键更新显卡驱动

  ![image-20210320115932959](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTensorFlow%E5%AE%89%E8%A3%85/image-20210320115932959.png)

## 查看显卡驱动对应`cuda`版本

* 桌面右键显卡控制面板

  ![image-20210320120246190](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTensorFlow%E5%AE%89%E8%A3%85/image-20210320120246190.png)

## 下载安装`cuda`

* [下载地址](https://developer.nvidia.com/cuda-toolkit-archive)：非常之慢

* [下载地址2](https://developer.nvidia.com/zh-cn/cuda-downloads)

* [安装参考地址](https://blog.csdn.net/XunCiy/article/details/89070315)

  ![image-20210320153609006](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTensorFlow%E5%AE%89%E8%A3%85/image-20210320153609006.png)

* 以下两个地方注意比选就行

  ![image-20210320122925046](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTensorFlow%E5%AE%89%E8%A3%85/image-20210320122925046.png)

  ![image-20210320122936325](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTensorFlow%E5%AE%89%E8%A3%85/image-20210320122936325.png)



## 查看`cuda`对应的`cudnn`版本

* [地址](https://www.tensorflow.org/install/source_windows#gpu)

  ![image-20210320121040837](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTensorFlow%E5%AE%89%E8%A3%85/image-20210320121040837.png)

## 下载安装`cudnn`

* [下载地址](https://developer.nvidia.com/zh-cn/cudnn)：需要登录

* [下载地址2](https://developer.nvidia.cn/rdp/cudnn-download)

  ![image-20210320162205564](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTensorFlow%E5%AE%89%E8%A3%85/image-20210320162205564.png)

* 安装

  ![image-20210320163021436](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTensorFlow%E5%AE%89%E8%A3%85/image-20210320163021436.png)

## 安装`tensorflow-gpu`

* 根据`cuda`对应版本安装`tf`的对应版本，使用`pycharm`安装

  ![image-20210320163946401](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTensorFlow%E5%AE%89%E8%A3%85/image-20210320163946401.png)

## 测试是否安装成功

```python
import tensorflow as tf
print(tf.__version__)
print('GPU', tf.test.is_gpu_available())

# output
# GPU [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

* 出现以下报错时：

  ```shell
  2021-03-20 16:51:38.885475: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found
  ```

* 将`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\bin`目录下的`cusolver64_11.dll`复制，并将副本改名为`cusolver64_10.dll`

## 注意

1. 显卡有对应的显卡驱动
2. 显卡驱动对应`cuda`的版本
3. 根据`cuda`的版本对应`cudnn`、`tf`版本

# Linux安装

## linux配置conda环境及搭建深度学习环境

### 安装minconda3

* [参考](https://docs.anaconda.com/anaconda/install/linux/)

```shell
# 安装miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 校验hash值
sha256sum Miniconda3-latest-Linux-x86_64.sh

# 更改执行权限
chmod +x Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh

# 安装完成后配置任意位置启动conda
source <path to conda>/bin/activate
conda init
conda deactivate

source ~/.bashrc
conda config --set auto_activate_base False

# 验证
conda --version

# 装好conda之后，确定一个你放软件的地方
mkdir ~/conda_software
conda create -p ~/conda_software

# 配置环境变量，可以加到.bashrc或者.zshrc文件，这样每次登录不用再设置
export PATH=$HOME/conda_software/bin${PATH:+:${PATH}}

# 安装软件
conda activate ~/softwares/conda_software
conda config --add channels conda-forge

# 以下举几个例子：
# conda install zsh
# conda install htop
# conda install tmux
# conda install openssl

# 查看安装了哪些包
conda list

# 查看当前存在哪些虚拟环境
conda env list
# conda info -e

# 检查更新
conda update conda

# 创建python虚拟环境
conda create -n python38 python=3.8

# 激活
conda activate python38

# 安装包
# 指定环境安装包
conda install -n python38 numpy
# 在环境内安装包
conda install numpy

# 关闭虚拟环境
conda deactivate

# 删除虚拟环境
conda remove -n python38 --all

# 删除环境中的包
conda remove -n python38 numpy

# 设置国内镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn

# 设置搜索时显示通道地址
conda config --set show_channel_urls yes

# 恢复默认镜像
conda config --remove-key channels
```

### 配置服务器jupyter环境

* [参考](https://blog.lihj.me/post/conda-jupyter-installation.html)

```shell
# 安装jupyter lab
conda install jupyterlab

# jupyter 配置

# 生成配置文件
jupyter notebook --generate-config

ipython

# ipython
from notebook.auth import passwd
passwd()

# 复制生成的密文


# 修改默认配置文件
vim ~/.jupyter/jupyter_notebook_config.py

# 修改以下内容
c.NotebookApp.allow_remote_access = True
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.password = u'sha1:...刚才复制的密文'
c.NotebookApp.port = 8888 # 指定一个访问端口

# 启动jupyter
# jupyter notebook
jupyter lab

# 远程访问
# 在本地浏览器输入 address_of_remote:8888 进入jupyter 的登陆界面


# 安装 jupyter 使用的 python 可直接被 jupyter 调用，不需额外配置。
# 为了使用其他语言或者新环境中的某种语言，需要单独安装该语言的 jupyter kernel 供 jupyter 调用
# 进入环境
conda activate python38

# 安装依赖包
conda install notebook ipykernel

# 安装python kernel
which ipython # should show ~/miniconda3/envs/python38/bin/ipython
ipython kernel install --user --name "python38" --display-name "Python38"

# kernel 文件保存在 ~/.local/share/python38/kernels/python37。

# 此时安装的是新环境中 ipython 所属的 python 3.8 (~/miniconda3/envs/python38/bin/python) 为 jupyter kernel。

# 其他位置或环境的 python 可用相同方法安装为 jupyter kernel
```

### 升级cuda

* [各显卡驱动下载地址](https://www.nvidia.cn/Download/index.aspx?lang=cn)
* [CUDA下载地址](https://developer.nvidia.com/zh-cn/cuda-toolkit)
* [cudnn下载地址](https://developer.nvidia.com/rdp/cudnn-archive)：会让登录和比较慢，耐心等待
* [tensorflow各版本对比](https://www.tensorflow.org/install/source#linux)

#### 删除历史cuda版本信息

```shell
# sudo rm /etc/apt/sources.list.d/cuda*
# sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" 
# sudo apt-get --purge remove "*nvidia*"
# sudo apt-get autoremove
# sudo apt-get autoclean
# sudo rm -rf /usr/local/cuda*
# 卸载所有N卡驱动
sudo apt-get remove --purge nvidia-\*
sudo apt-get remove --purge cuda-\*
sudo apt-get remove --purge *cudnn*
sudo apt autoremove
sudo apt-get autoclean


# 查看已安装的东西
sudo dpkg --list | grep nvidia*
ubuntu1604_1.0.0-1_amd64.deb  # 可安装的显卡驱动
lspci | grep -i nvidia  # 查看显卡
nvidia-smi  # 查看显卡
```



#### [Linux系统信息查看](https://blog.csdn.net/weixin_41010198/article/details/109166131)

```shell
# 查看内核版本
cat /proc/version
uname -a
uname -r

# 查看linux版本信息
lsb_release -a
cat /etc/issue

# 查看linux是64为还是32位
getconf LONG_BIT
file /bin/ls

# 直接查看系统的架构
dpkg --print-architecture
arch
file /lib/systemd/systemd

# 查看Mint系统对应的Ubuntu系统
cat /etc/os-release
cat /etc/upstream-release/lsb-release

gcc --version  # 查看gcc版本
```



#### [知乎教程](https://zhuanlan.zhihu.com/p/143429249)

* 安装教程来几乎没问题

```shell
# 安装显卡驱动
sudo add-apt-repository ppa:graphics-drivers  #添加NVIDA显卡驱动库
sudo apt update
ubuntu-drivers devices  #显示可安装驱动

#sudo ubuntu-drivers autoinstall  #让Ubuntu自动帮你选择版本并安装
sudo apt install nvidia-driver-450  #安装450驱动
sudo reboot
nvidia-smi  #查看GPU信息,需先重启

# 安装cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt list cuda* # 查看所有，名字以cuda开头的，可以用apt install安装的packages

sudo apt install cuda-toolkit-10-2  #只安装CUDA 10.2
#sudo apt install cuda-10-2  #安装CUDA 10.2。包含驱动，版本自动选择。

# 安装cudnn
# 官网下载符合版本的三个deb
cd ~/Downloads  #进入下载好的三个文件的路径
sudo dpkg -i libcudnn*  #同时安装
#sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb  #逐个安装
#sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb
#sudo dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.2_amd64.deb

# 测试
cp -r /usr/src/cudnn_samples_v7/ $HOME  #复制样本文件到$HOME文件夹下
cd  $HOME/cudnn_samples_v7/mnistCUDNN  #进入样本目录
make clean && make  #编译
./mnistCUDNN  #执行cuDNN测试
# 输出“Test Passed”说明cuDNN安装成功。

# 安装tensorflow
pip install tensorflow-gpu==2.4.0

# 测试
import tensorflow as tf
tf.test.is_gpu_avaiable()
```



#### [tensorflow官网教程：GPU支持](https://www.tensorflow.org/install/gpu)

* Ubuntu 16.04(CUDA 11.0)

```shell
# Add NVIDIA package repositories
# Add HTTPS support for apt-key
sudo apt-get install gnupg-curl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
# apt-get install software-properties-common
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
# apt-get install apt-transport-https ca-certificates
sudo apt-get update
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt-get update
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt-get update

# Install NVIDIA driver
# Issue with driver install requires creating /usr/lib/nvidia
sudo mkdir /usr/lib/nvidia
ubuntu-drivers devices  # 查看显卡和推荐驱动
sudo apt-get install --no-install-recommends nvidia-450
# sudo apt-get install --no-install-recommends nvidia-driver-450
# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install development and runtime libraries (~4GB)
# sudo apt install cuda-toolkit-11-0  #只安装CUDA 10.2
# sudo apt install cuda-10-2  #安装CUDA 10.2。包含驱动，版本自动选择。
sudo apt-get install --no-install-recommends \
    cuda-11-0 \
    libcudnn8=8.0.4.30-1+cuda11.0  \
    libcudnn8-dev=8.0.4.30-1+cuda11.0


# Install TensorRT. Requires that libcudnn7 is installed above.
sudo apt-get install -y --no-install-recommends \
    libnvinfer7=7.1.3-1+cuda11.0 \
    libnvinfer-dev=7.1.3-1+cuda11.0 \
    libnvinfer-plugin7=7.1.3-1+cuda11.0 \
    libnvinfer-plugin-dev=7.1.3-1+cuda11.0

```



#### 导师教程

```shell
1、安装CUDA及cuDNN
TENSORFLOW对CUDA的要求：https://www.tensorflow.org/install/install_linux?hl=zh-cn
nvdia的官方文档：http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#handle-uninstallation
（1）tensorflow1.8的版本要求为：
* CUDA9.0，不支持8.0
* cuDNN7.0.x，不支持7.1.x

（2）卸载 CUDA 8.0：如果原来没有安装CUDA8.0，则可以忽略这一步
/usr/local/cuda-X.Y/bin/uninstall_cuda_X.Y.pl
比如：
/usr/local/cuda-8.0/bin/uninstall_cuda_8.0.pl
【可选】驱动也一并卸掉，因为新版 CUDA 通常需要装一个新驱动。
$ sudo /usr/bin/nvidia-uninstall

（3）下载CUDA9.0 与 cuDNN7.0
下载 CUDA Toolkit 现在需要注册一个 NVIDIA 官方账号。注册完成后在 https://developer.nvidia.com/cuda-release-candidate-download 按照系统、版本选择要下载的包。cuDNN 的安装类似，地址在 https://developer.nvidia.com/rdp/cudnn-download 。不过官方文档表示 cuDNN 的升级不会冲突，直接安装就好。 

（4）安装CUDA9.0
sudo dpkg -i cuda-repo-ubuntu1704-9-0-local-rc_9.0.103-1_amd64.deb
sudo apt-key add /var/cuda-repo-9-0-local-rc/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda

在装新 CUDA 的时候系统会安装新版驱动。

（5）安装完后运行 nvidia-smi 试一下，如果提示 mismatch 就重启。 

（6）配置环境变量
$ export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

（7）安装cuDNN7.0.x
dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb

（8）安装cuda-command-line-tools
apt-get install cuda-command-line-tools

2、安装tensorflow
pip3 install tensorflow-gpu
```



#### 安装时遇到的问题

1. [nvidia-smi报错：NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver._missyoudaisy的博客-CSDN博客](https://blog.csdn.net/missyoudaisy/article/details/104432746)
2. [apt-get update ，upgarde 和dist-upgrade 的区别_wangyezi19930928的专栏-CSDN博客](https://blog.csdn.net/wangyezi19930928/article/details/54928201)
3. [`Error! Could not locate dkms.conf file` - Ask Ubuntu](https://askubuntu.com/questions/227258/error-could-not-locate-dkms-conf-file)
4. [更新Linux内核头文件(linux headers)_xiaoaide01的专栏-CSDN博客](https://blog.csdn.net/xiaoaid01/article/details/41862487)
5. [Error! Your kernel headers for kernel 4.4.0-210-generic cannot be found - Google 搜索](https://www.google.com.hk/search?q=Error!+Your+kernel+headers+for+kernel+4.4.0-210-generic+cannot+be+found&rlz=1C1GCEU_zh-CNCN866CN866&oq=Error!+Your+kernel+headers+for+kernel+4.4.0-210-generic+cannot+be+found&aqs=chrome..69i57.681j0j4&sourceid=chrome&ie=UTF-8)
6. [17.04 - Unable to install nvidia drivers - unable to locate package - Ask Ubuntu](https://askubuntu.com/questions/951046/unable-to-install-nvidia-drivers-unable-to-locate-package)
7. [Ubuntu报错software-properties-common : Depends: python3-software-properties](https://blog.csdn.net/qq_34168515/article/details/107410732)
8. [ubuntu-drivers: command not found](https://askubuntu.com/questions/361862/nvidia-drivers-synaptic)
9. [nvidia-smi返回错误信息‘Failed to initialize NVML: Driver/library version mismatch’](https://spring-quan.github.io/2019/03/29/nvidia-smi%E8%BF%94%E5%9B%9E%E9%94%99%E8%AF%AF%E4%BF%A1%E6%81%AF%E2%80%98Failed-to-initialize-NVML-Driver-library-version-mismatch%E2%80%99/)