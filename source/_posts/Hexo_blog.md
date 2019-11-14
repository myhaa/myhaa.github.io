---
title: Hexo创建个人博客
date: 2019-11-12 11:21:11
author: Myhaa
img:
top: false
cover: false
coverImg:
password:
toc: true
mathjax: false
summary: 运用Hexo+GitHub搭建个人博客详细教程
categories: GitHub
tags:
  - Hexo
  - GitHub
---

# 一、Hexo介绍

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

![图1](/Hexo_blog/create1.png)

![图2](/Hexo_blog/create2.png)

### （2）注意事项

1. 第二张图中的第一步yourname一定要跟Owner的名字一样，且一定要加`.github.io`后缀（至于为什么，我也不得而知）

## 4、给仓库选择主题

### （1）如图操作

![图3](/Hexo_blog/create3.png)

![图4](/Hexo_blog/create4.png)

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

1. 详情请见另一篇文章：{% post_link Git Git %}

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

![图5](/Hexo_blog/create5.png)

3. 注意：修改图中标红的地方就OK，换成你自己的name

### （3）配置全局git name and email

```shell
git config --global user.name "your github name"
git config --global user.email "your github private email"
```

​	**注意：**`your github private email`怎么配置请看{% post_link Git Git %}

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

### （2）插入图片

1. 安装插件

   ```shell
   npm install https://github.com/CodeFalling/hexo-asset-image --save
   ```


2. 修改`_config_yml`配置

   ```shell
   post_asset_folder: true  # 将false改为true
   ```

3. 在`./source/_posts`中新建一个md文件时，同时创建一个与md文件同名的文件夹，该文件夹用来存放该md文件所需图片
4. 接着在md文件中以`![](/md文件名/1.png)`的格式插入图片
5. 详细教程请见[ETRD博客](http://etrd.org/2017/01/23/hexo中完美插入本地图片/)

### （3）插入markdown

1. Hexo的[标签插件](<https://hexo.io/zh-cn/docs/tag-plugins.html>)

2. 引用站内文章

   ```shell
   {% post_path filename %}
   {% post_link filename [title] [escape] %}
   {% post_link hexo-3-8-released %} # 链接使用文章的标题
   ```

3. 在使用此标签时可以忽略文章文件所在的路径或者文章的永久链接信息、如语言、日期。

   例如，在文章中使用 `{% post_link how-to-bake-a-cake %}` 时，只需有一个名为 `how-to-bake-a-cake.md` 的文章文件即可。即使这个文件位于站点文件夹的 `source/posts/2015-02-my-family-holiday` 目录下、或者文章的永久链接是 `2018/en/how-to-bake-a-cake`，都没有影响。

   默认链接文字是文章的标题，你也可以自定义要显示的文本。此时不应该使用 Markdown 语法 `[]()`。

   默认对文章的标题和自定义标题里的特殊字符进行转义。可以使用`escape`选项，禁止对特殊字符进行转义。

4. [参考链接](<https://www.jibing57.com/2017/10/30/how-to-use-post-link-on-hexo/>)

## 5、更换主题

* [hexo-theme-matery](<https://github.com/blinkfox/hexo-theme-matery>)