---
title: Python笔记
date: 2019-10-15 15:21:11
author: Myhaa
img:
top: false
cover: false
coverImg:
password:
toc: true
mathjax: false
summary: 有关Python的笔记
categories: 编程与开发
tags:
  - Python
  - 编程语言
---



# 一、Python基础



# 二、Python进阶



# 三、参考书籍



# 四、疑难解答

## 1、编码问题

### `Python2`编码问题

* [参考](<https://foofish.net/why-Python-encoding-is-tricky.html>)

### `Python3`编码问题

* 在`Python3`版本中，把`'xxx'`和`u'xxx'`统一成`Unicode`编码，即写不写前缀`u`都是一样的。
* 在`Python3`版本中，所有的字符串都是使用`Unicode`编码的字符串序列。
* [参考](<https://foofish.net/how-Python3-handle-charset-encoding.html>)

## 2、日期操作

* [参考菜鸟教程](<https://www.runoob.com/python/python-date-time.html>)

### 产生一段时间的日期

```python
from datetime import datetime, date, timedelta
import pandas as pd
date_id_list = [datetime.strftime(x, '%Y%m%d') for x in list(pd.date_range(start='20190701', end='20190928'))]
```

### 获取指定日期的前-后N天

```python
import datetime
n = 1
tomorrow = datetime.datetime(2015, 10, 28) + datetime.timedelta(days=1)	# 2015-10-29 00:00:00
tomorrow_format = tomorrow.strftime('%Y%m%d')	# '20151029'
```



## 3、mrjob运行参数详情

```shell
python xxx.py -r hadoop --local-tmp-dir 'xxx' --hadoop-tmp-dir 'hdfs:xxx' --file 'xxx.txt' --jobconf mapred.map.tasks=20 --jobconf mapred.reduce.tasks=2 input.txt -o output_dir
```

**参考官方文档：**[mrjob](<https://mrjob.readthedocs.io/en/latest/>)

## 4、list中排列组合

```python
from itertools import combinations

combine_2 = list(combinations([1,2,3,4], 2))
```

## 5、文件操作

### 获取指定目录下指定文件

```python
import os

L = []
for root, dirs, files in os.walk(os.getcwd()):
    for x in files:
        if os.path.splitext(x)[1] == '.txt':
            L.append(os.path.join(root, x))
file_path = L[0]
```

### pandas读取excel文件

```python
roc_data = pd.read_excel(file_path, sheet_name='20200323_10000')
roc_data
```



## 6、命令行参数`sys.argv[1:]`解析

* [python类库31--命令行解析](https://www.cnblogs.com/itech/archive/2010/12/31/1919017.html)

### 手动解析

### getopt解析

### optionparser解析【推荐】

## 7、字典排序

```python
result = {}
sorted(result.items(), key=lambda x: x[1], reverse=True)
```

## 8、python自动登录Linux等服务

[参考](https://pexpect.readthedocs.io/en/stable/overview.html)

```python
import pexpect
child = pexpect.spawn('ftp ftp.openbsd.org')
child.expect('Name .*: ')
child.sendline('anonymous')
child.expect('Password:')
child.sendline('noah@example.com')
child.expect('ftp> ')
child.sendline('lcd /tmp')
child.expect('ftp> ')
child.sendline('cd pub/OpenBSD')
child.expect('ftp> ')
child.sendline('get README')
child.expect('ftp> ')
child.sendline('bye')
```

## 9、 生成requirements.txt

### 第一种方法：太多太杂，对整个虚拟环境的

```python
# 生成
pip freeze > requirements.txt

# 安装
pip install -r requirements.txt
```

### 第二种方法：需要pip安装模块，但是可以对指定目录进行生成

```
# pip 安装模块
pip3 install pipreqs

# 对指定目录进行生成requirements.txt
cd 到指定目录
pipreqs ./ --encoding=utf8
# 这样在指定目录就会有requirements.txt的依赖文件
```



## 10、python取mysql中文乱码

1. mysql代码中将中文字段用`hex`函数转换
2. python代码中用`bytes.fromhex(取出的字段).decode('utf-8')`来转换

