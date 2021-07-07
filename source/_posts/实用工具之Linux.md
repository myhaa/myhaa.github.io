---
title: 实用工具之Linux
date: 2019-10-14 15:21:11
author: Myhaa
img:
top: false
cover: false
coverImg:
password:
toc: true
mathjax: false
summary: 有关Linux的笔记
categories: 实用工具
tags:
  - Linux
---

![image-20210621145304764](%E5%AE%9E%E7%94%A8%E5%B7%A5%E5%85%B7%E4%B9%8BLinux/image-20210621145304764.png)

# 一、Linux基础

## 常用命令

### rsync

* [参考](https://man.linuxde.net/rsync)

#### 语法

```shell
rsync [OPTION]... SRC DEST
rsync [OPTION]... SRC [USER@]host:DEST
rsync [OPTION]... [USER@]HOST:SRC DEST
rsync [OPTION]... [USER@]HOST::SRC DEST
rsync [OPTION]... SRC [USER@]HOST::DEST
rsync [OPTION]... rsync://[USER@]HOST[:PORT]/SRC [DEST]
```

对应于以上六种命令格式，rsync有六种不同的工作模式：

1. 拷贝本地文件。当SRC和DES路径信息都不包含有单个冒号":"分隔符时就启动这种工作模式。如：`rsync -a /data /backup`
2. 使用一个远程shell程序(如[rsh](http://man.linuxde.net/rsh)、[ssh](http://man.linuxde.net/ssh))来实现将本地机器的内容拷贝到远程机器。当DST路径地址包含单个冒号":"分隔符时启动该模式。如：`rsync -avz *.c foo:src`
3. 使用一个远程shell程序(如rsh、ssh)来实现将远程机器的内容拷贝到本地机器。当SRC地址路径包含单个冒号":"分隔符时启动该模式。如：`rsync -avz foo:src/bar /data`
4. 从远程rsync服务器中拷贝文件到本地机。当SRC路径信息包含"::"分隔符时启动该模式。如：`rsync -av root@192.168.78.192::www /databack`
5. 从本地机器拷贝文件到远程rsync服务器中。当DST路径信息包含"::"分隔符时启动该模式。如：`rsync -av /databack root@192.168.78.192::www`
6. 列远程机的文件列表。这类似于rsync传输，不过只要在命令中省略掉本地机信息即可。如：`rsync -v rsync://192.168.78.192/www`

#### 选项

```shell
-v, --verbose 详细模式输出。
-q, --quiet 精简输出模式。
-c, --checksum 打开校验开关，强制对文件传输进行校验。
-a, --archive 归档模式，表示以递归方式传输文件，并保持所有文件属性，等于-rlptgoD。
-r, --recursive 对子目录以递归模式处理。
-R, --relative 使用相对路径信息。
-b, --backup 创建备份，也就是对于目的已经存在有同样的文件名时，将老的文件重新命名为~filename。可以使用--suffix选项来指定不同的备份文件前缀。
--backup-dir 将备份文件(如~filename)存放在在目录下。
-suffix=SUFFIX 定义备份文件前缀。
-u, --update 仅仅进行更新，也就是跳过所有已经存在于DST，并且文件时间晚于要备份的文件，不覆盖更新的文件。
-l, --links 保留软链结。
-L, --copy-links 想对待常规文件一样处理软链结。
--copy-unsafe-links 仅仅拷贝指向SRC路径目录树以外的链结。
--safe-links 忽略指向SRC路径目录树以外的链结。
-H, --hard-links 保留硬链结。
-p, --perms 保持文件权限。
-o, --owner 保持文件属主信息。
-g, --group 保持文件属组信息。
-D, --devices 保持设备文件信息。
-t, --times 保持文件时间信息。
-S, --sparse 对稀疏文件进行特殊处理以节省DST的空间。
-n, --dry-run现实哪些文件将被传输。
-w, --whole-file 拷贝文件，不进行增量检测。
-x, --one-file-system 不要跨越文件系统边界。
-B, --block-size=SIZE 检验算法使用的块尺寸，默认是700字节。
-e, --rsh=command 指定使用rsh、ssh方式进行数据同步。
--rsync-path=PATH 指定远程服务器上的rsync命令所在路径信息。
-C, --cvs-exclude 使用和CVS一样的方法自动忽略文件，用来排除那些不希望传输的文件。
--existing 仅仅更新那些已经存在于DST的文件，而不备份那些新创建的文件。
--delete 删除那些DST中SRC没有的文件。
--delete-excluded 同样删除接收端那些被该选项指定排除的文件。
--delete-after 传输结束以后再删除。
--ignore-errors 及时出现IO错误也进行删除。
--max-delete=NUM 最多删除NUM个文件。
--partial 保留那些因故没有完全传输的文件，以是加快随后的再次传输。
--force 强制删除目录，即使不为空。
--numeric-ids 不将数字的用户和组id匹配为用户名和组名。
--timeout=time ip超时时间，单位为秒。
-I, --ignore-times 不跳过那些有同样的时间和长度的文件。
--size-only 当决定是否要备份文件时，仅仅察看文件大小而不考虑文件时间。
--modify-window=NUM 决定文件是否时间相同时使用的时间戳窗口，默认为0。
-T --temp-dir=DIR 在DIR中创建临时文件。
--compare-dest=DIR 同样比较DIR中的文件来决定是否需要备份。
-P 等同于 --partial。
--progress 显示备份过程。
-z, --compress 对备份的文件在传输时进行压缩处理。
--exclude=PATTERN 指定排除不需要传输的文件模式。
--include=PATTERN 指定不排除而需要传输的文件模式。
--exclude-from=FILE 排除FILE中指定模式的文件。
--include-from=FILE 不排除FILE指定模式匹配的文件。
--version 打印版本信息。
--address 绑定到特定的地址。
--config=FILE 指定其他的配置文件，不使用默认的rsyncd.conf文件。
--port=PORT 指定其他的rsync服务端口。
--blocking-io 对远程shell使用阻塞IO。
-stats 给出某些文件的传输状态。
--progress 在传输时现实传输过程。
--log-format=formAT 指定日志文件格式。
--password-file=FILE 从FILE中得到密码。
--bwlimit=KBPS 限制I/O带宽，KBytes per second。
-h, --help 显示帮助信息。
```



# 二、Linux进阶

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

​```ipython
from notebook.auth import passwd
passwd()
​```

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



# 三、参考书籍

# 四、疑难解答

## 1、修改文件或目录的权限

（1）语法

```shell
chmod [-cfvR] [--help] [--version] mode file...
```

（2）参数说明

**mode :** 权限设定字串，格式如下 :

```shell
[ugoa...][[+-=][rwxX]...][,...]
```

**其中：**

- u 表示该文件的拥有者，g 表示与该文件的拥有者属于同一个群体(group)者，o 表示其他以外的人，a 表示这三者皆是。
- \+ 表示增加权限、- 表示取消权限、= 表示唯一设定权限。
- r 表示可读取，w 表示可写入，x 表示可执行，X 表示只有当该文件是个子目录或者该文件已经被设定过为可执行。

（3）其他参数说明：

- -c : 若该文件权限确实已经更改，才显示其更改动作
- -f : 若该文件权限无法被更改也不要显示错误讯息
- -v : 显示权限变更的详细资料
- -R : 对目前目录下的所有文件与子目录进行相同的权限变更(即以递回的方式逐个变更)
- --help : 显示辅助说明
- --version : 显示版本

（4）实例

```shell
# 给个人目录的其他用户删除写权限
 hadoop fs -chmod -R o-w /user/name/dir
```

## 2、vim 文本搜索

### 问题：

* 在Linux环境中，一个大文本中搜索指定字符串应该怎么操作？

### 解决：

```linux
vi my.txt
```

* 键盘按`Esc`
* 输入`/search_string`
* 键盘按`n`或者`N`来进行向前或向后搜索

## 3、日期循环

```shell
#! /bin/bash

start=20200312
end=20200322

while [ ${start} -le ${end} ]
do
  echo ${start}
  start=`date -d "1 day ${start}" +%Y%m%d`	# 日期自增
done
```

* 参考[日期循环](https://sjq597.github.io/2015/11/03/Shell-按日期循环执行/)

## 4、将代码输出重定向到log文件-不覆盖的形式

```shell
0 11 * * * /usr/bin/python3 /home/user/adsp_new/orientation.py >> /home/user/adsp_new/logs1_ori.txt 2>&1
```

## 5、查看系统版本

```shell
lsb_release -a
```

## 6、查看显卡信息

```shell
lspci | grep -i nvidia
```

