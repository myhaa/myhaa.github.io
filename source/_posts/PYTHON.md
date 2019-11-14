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
categories: Programming
tags:
  - Python
---



# 一、Python基础



# 二、Python进阶



# 三、参考书籍



# 四、疑难解答

## 1、`Python2`编码问题

* [参考](<https://foofish.net/why-Python-encoding-is-tricky.html>)

## 2、`Python3`编码问题

* 在`Python3`版本中，把`'xxx'`和`u'xxx'`统一成`Unicode`编码，即写不写前缀`u`都是一样的。
* 在`Python3`版本中，所有的字符串都是使用`Unicode`编码的字符串序列。
* [参考](<https://foofish.net/how-Python3-handle-charset-encoding.html>)

## 3、产生一段时间的日期

```python
from datetime import datetime, date, timedelta
import pandas as pd
date_id_list = [datetime.strftime(x, '%Y%m%d') for x in list(pd.date_range(start='20190701', end='20190928'))]
```

