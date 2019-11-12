---
title: PYTHON笔记
---

[TOC]

# 一、PYTHON基础



# 二、PYTHON进阶



# 三、PYTHON参考书籍



# 四、工作常见问题

## 1、`PYTHON2`编码问题

* [参考](<https://foofish.net/why-python-encoding-is-tricky.html>)

## 2、`PYTHON3`编码问题

* 在`Python3`版本中，把`'xxx'`和`u'xxx'`统一成`Unicode`编码，即写不写前缀`u`都是一样的。
* 在`Python3`版本中，所有的字符串都是使用`Unicode`编码的字符串序列。
* [参考](<https://foofish.net/how-python3-handle-charset-encoding.html>)

## 3、产生一段时间的日期

```python
from datetime import datetime, date, timedelta
import pandas as pd
date_id_list = [datetime.strftime(x, '%Y%m%d') for x in list(pd.date_range(start='20190701', end='20190928'))]
```





[回到顶部](#一、PYTHON基础)