---
title: HEXO创建个人博客
---

[TOC]

# 一、`Hexo`介绍

## 1、官网

[Hexo](<https://hexo.io/zh-cn/index.html>)

## 2、安装

### （1）Node.js（Windows）

1. [官网](<https://nodejs.org/zh-cn/>)下载对应版本

2. 安装选择全部默认（也可以自己设定安装位置）

3. 安装完，键盘上`Win+R`打开命令行，输入以下命令出现版本号即安装成功

   ```shell
   node -v
   npm -v
   ```

### （2）Git（Windows）

1. [官网](<https://git-scm.com/>)下载对应版本

2. 安装选择全部默认（也可以自己设定安装位置）

3. 最后一步也可以选择`Use Git from the Windows Command Prompt`，这样就可以命令行打开`git`

4. 安装完，命令行输入以下命令出现版本号即安装成功

   ```shell
   git --version
   ```

### （3）Hexo（Windows）

1. 在Windows上选定一个目录作为博客目录，在该目录下右键点击`Git Bash Here`，接下来使用git控制台进行Hexo的安装

   ```shell
   npm i hexo-cli -g
   hexo -v  # 验证安装是否成功
   ```

# 二、GitHub

## 1、官网

[GitHub](<https://github.com/>)

## 2、注册

官网跟着指引注册就OK

## 3、创建博客仓库

### （1）如图操作

![第一步](pictures\创建博客仓库1.png)

![第二步](pictures\创建博客仓库2.png)

### （2）注意事项

1. 第二张图中的第一步yourname一定要跟Owner的名字一样，且一定要加`.github.io`后缀（至于为什么，我也不得而知）

## 4、给仓库选择主题

### （1）如图操作

![第三步](pictures\创建博客仓库3.png)

![第四步](pictures\创建博客仓库4.png)

# 三、写博客并发布到GitHub

## 1、本地网站配置

### （1）命令如下（在博客目录下）

```shell
hexo init  # 初始化该目录
npm install  # 安装必备的组件
hexo g  # 生成静态网页
hexo s  # 打开本地服务器并复制地址到chrome打开
ctrl c  # 关闭本地服务器
```

### （2）chrome打开本地网站地址

1. [<http://localhost:4000/>](<http://localhost:4000/>)

## 2、 写新文章

### （1）命令如下（在博客目录下）

```shell
npm i hexo-deployer-git  # 安装拓展
hexo new post "new md file"  # 新建一篇文章
# 修改./source/_posts下的md文件
hexo g
hexo s
ctrl c
```

## 3、连接GitHub

### （1）如何配置连接

1. 详情请见另一篇文章：[Git](Git.md)

### （2）修改配置

1. 命令（在博客目录下）

   ```shell
   vi _config.yml
   a  # 修改
   # 如下图修改最后几行
   Esc  # 退出修改
   :wq!  # 保存
   ```

2. 如图

![第五步](pictures\创建博客仓库5.png)

3. 注意：修改图中标红的地方就OK，换成你自己的name

### （3）配置全局git name and email

```shell
git config --global user.name "your github name"
git config --global user.email "your github private email"
```

​	**注意：**`your github private email`怎么配置请看[Git](Git.md)

### （4）发布到GitHub

```shell
hexo d
```

### （5）同时拥有gitlab and github账号时

1. 发布完后，将全局git name and email改为gitlab账号

   ```shell
   git config --global user.name "your gitlab name"
   git config --global user.email "your gitlab private email"
   ```

2. 进入`.deploy_git`配置局部账号即github账号

   ```shell
   cd .deploy_git  # 根目录下进入.deploy_git
   # 配置
   git config --local user.name "your github name"
   git config --local user.email "your github private email"
   ```

3. 这样配置后，hexo d就是用github账号push，对gitlab push就是用gitlab账号

## 4、新的文章

### （1）添加新md

1. 在./source/_posts下添加新的md文件

2. 使用命令push

   ```shell
   hexo g
   hexo d
   ```



[回到顶部](#一、`Hexo`介绍)