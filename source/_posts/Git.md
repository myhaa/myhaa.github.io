---
title: Git笔记
date: 2019-11-12 16:21:11
author: Myhaa
img:
top: false
cover: false
coverImg:
password:
toc: true
mathjax: false
summary: 有关Git的笔记
categories: GitHub
tags:
  - Git
  - GitHub
  - SSH
---



# 一、Git基础

## 1、Git简介

### （1）什么是Git？

Git是迄今为止最先进的分布式版本控制系统

### （2）Git安装

* [安装教程](<https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>)

## 2、创建版本库

### （1）将已有文件夹变为版本库

```shell
cd /home/user/my_project  # 切换到想要初始化的文件夹
git init  # 初始化为版本库，文件夹会出现.git的隐藏文件夹
git add .  # 添加文件夹中所有文件到暂存区
git commit -m "your commit description"  # 提交暂存区所有文件到版本库并保存提交记录
```

### （2）从远处仓库（GitLab\GitHub）克隆

```shell
cd /home/user/my_project  # 切换到想要存放版本库的文件夹
git clone https://github.com/myhaa/How-To-Ask-Questions-The-Smart-Way.git  # clone https地址
git clone git@github.com:myhaa/How-To-Ask-Questions-The-Smart-Way.git  # clone ssh地址
cd How-To-Ask-Questions-The-Smart-Way  # 进入clone的版本库文件夹
```

## 3、操作版本库

### （1）版本库中文件的两种状态

1. 未追踪状态（*untracked*）：从未**add+commit**的文件（Untracked files:）
2. 追踪状态（*tracked*）：曾经**add+commit**过的文件
   - 未修改（unmodified）：在版本库中
   - 已修改（modified）：Changes not staged for commit:
   - 暂存（staged）：Changes to be committed:
3. 详情如下图（图来自：<https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository>）

![图1：版本库中的文件状态](/Git/lifecycle.png)

### （2）Tracking New Files(untracked)

1. 查看版本库状态发现README文件是`Untracked file`。

```shell
$ git status
On branch master
Your branch is up-to-date with 'origin/master'.
Untracked files:
  (use "git add <file>..." to include in what will be committed)

    README

nothing added to commit but untracked files present (use "git add" to track)
```

2. 将README添加到暂存区并提交到版本库

```shell
git add README  # 添加到暂存区（staged）发现README状态为（Changes to be committed: new file）
git commit -m "add README"  # 提交到版本库
```

### （3）Staging Modified Files(tracked)

1. 修改刚刚提交到版本库的README文件

```shell
vi README  # 用vim修改README文件
git status  # 查看文件状态发现README的状态为（Changes not staged for commit: modified）
```

2. 将README添加到暂存区并提交到版本库

```shell
git add README  # 添加到暂存区（staged）
git commit -m "update README"  # 提交到版本库
```

### （4）Viewing Your Staged and Unstaged Changes

1. 比较`Changes not staged for commit: `下的文件与本地最新版本库的差别：

```shell
git diff
```

2. 比较`Changes to be committed:`下的文件与本地最新版本库的差别：

```shell
git diff --staged
```

3. 比较`Changes not staged for commit:`和`Changes to be committed:`下同名文件的差别

```shell
git diff --cached
```





# 二、Git进阶

# 三、参考书籍

* [官方教程](<https://git-scm.com/book/en/v2>)
* [廖大神](https://www.liaoxuefeng.com/wiki/896043488029600)

# 四、疑难解答




