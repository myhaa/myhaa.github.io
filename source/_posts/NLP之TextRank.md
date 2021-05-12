---
title: NLP之TextRank
author: Myhaa
top: false
cover: false
toc: true
mathjax: true
categories: NLP
tags:
  - 关键词提取
date: 2021-03-10 16:57:10
img:
coverImg:
password:
summary: 关键词提取之TextRank
---

# 基础

## 问题

* 如何对一篇文章的关键词进行提取

## 解决

* 使用类似网页排名算法`PageRank`的思路
* 构建词与词之间的图，然后迭代计算词的排名

## `TextRank`

* 回顾`PageRank`的计算公式：$P=(1-d)\frac{I}{n}+dA^TP$
* 直接说`TextRank`的计算公式：$P=(1-d)\frac{I}{n}+dW^TP$
  * 其中$W=(w_{ij})_{m\times n}$为词与词之间的权重，一般为词$i$与词$j$在滑动窗口$k$内的共现次数

## 如何根据词构建图

1. 对文章$S$进行分词，得到词列表

2. 设定滑动窗口$k$的大小，统计滑动窗口内各词对的贡献次数

   1. 例如：`淡黄的长裙，蓬松的头发`，分词后为[`淡黄`, `长裙`, `蓬松`, `头发`]

   2. 设定滑动窗口$k=2$，则得到词对：

      1. `淡黄`,`长裙`
      2. `长裙`,`蓬松`
      3. `蓬松`,`头发`

   3. 根据这些词对构建`无向图`，注意`PageRank`是`有向图`

      ![图片来源于参考链接1](NLP%E4%B9%8BTextRank/image-20210310171957155.png)

   4. 然后使用公式计算



# 进阶

# 疑难

## 实现

* [Jieba](https://github.com/fxsjy/jieba)

  ```python
  jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')) 
  ```

  



# 参考

1.  [TextRank: Bringing Order into Texts](http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
2. [（九）通俗易懂理解——TF-IDF与TextRank](https://zhuanlan.zhihu.com/p/41091116)

