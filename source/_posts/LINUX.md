---
title: Linux笔记
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
categories: 编程与开发
tags:
  - Linux
  - 网络安全
---

# 一、Linux基础

# 二、Linux进阶

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
0 11 * * * /usr/bin/python3 /home/meiyunhe/adsp_new/orientation.py >> /home/meiyunhe/adsp_new/logs1_ori.txt 2>&1
```

