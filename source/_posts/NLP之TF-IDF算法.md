---
title: NLP之TF-IDF算法
author: Myhaa
top: false
cover: false
toc: true
mathjax: true
categories: AI/数据科学
tags:
  - 数据科学
  - NLP
date: 2021-03-10 11:55:41
img:
coverImg:
password:
summary: TF-IDF算法解释
---

# 基础

## 问题

* 一篇文章中如何提取出能代表这篇文章的关键词

## 解答

* 一个解决方案是提取出该篇文章中`高频词`
* 但是`高频词`可能是那些`的，是`等词，所以要剔除这些词，这些词被称为`停用词(stop words)`
* 还有就是类似`中国`等词高频，但是可能不是文章中的关键词，因为这种词在很多文章中都会出现，所以需要针对高频词作一个重要性调整，称为重要性调整系数
* 那如果一个词高频，且比较少见，那这种词最可能反映该文章特性
* 这就对应了`TF-IDF`算法

## `TF-IDF`算法

### `TF`

* $\text{词频}(TF)=\frac{\text{某个词在文章中的出现次数}}{\text{文章的总词数}}$
* 或者
* $\text{词频}(TF)=\frac{\text{某个词在文章中的出现次数}}{\text{该文出现次数最多的词的出现次数}}$

### `IDF`

* $\text{逆文档频率}(IDF)=\log{\frac{\text{语料库的文档总数}}{\text{包含该词的文档数}+1}}$

### `TF-IDF`

* $TF-IDF=TF \times IDF$

## 优缺点

1. 优点：简单快速，结果比较符合实际情况
2. 缺点：
   1. 单纯以`词频`衡量一个词的重要性，不够全面，有时重要的词可能出现次数并不多
   2. 而且，这种算法无法体现词的位置信息，出现位置靠前的词与出现位置靠后的词，都被视为重要性相同，这是不正确的。（一种解决方法是，对全文的第一段和每一段的第一句话，给予较大的权重。）

# 进阶

# 疑难

## 计算工具

1. [Jieba](https://github.com/fxsjy/jieba)

   ```python
   import jieba.analyse
   jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
   ```

# 参考

* [TF-IDF与余弦相似性的应用（一）：自动提取关键词](https://www.ruanyifeng.com/blog/2013/03/tf-idf.html)

