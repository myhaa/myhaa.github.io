---
title: '''NLP之分词原理'''
author: Myhaa
top: false
cover: false
toc: true
mathjax: true
categories: AI/数据科学
tags:
  - AI
  - NLP
  - 分词
date: 2021-03-09 10:29:55
img:
coverImg:
password:
summary: 文本挖掘之分词原理
---

# 基础

## 分词的基本原理

* 基于统计的分词，统计的样本内容来自一些标准的语料库

* 举个栗子，我们分词的时候希望~小爱/来到/天河/区~这个分词后句子出现的概率要比~小爱/来到/天/河/区~的大

* 上述例子用数学语言表示如下

* 如果有一个句子`S`，它有`m`中分词选项如下：
  $$
  A_{11}A_{12}...A_{1n_1}  \\
  A_{21}A_{22}...A_{2n_2}  \\
  ...  \\
  A_{m1}A_{m2}...A_{mn_m}
  $$

* 其中$n_i$代表第$i$种分词的词个数，我们希望从中选出的是：$r = \underbrace{arg\;max}_iP(A_{i1},A_{i2},...,A_{in_i})$

* 但是$P(A_{i1},A_{i2},...,A_{in_i})$不好求，故使用马尔可夫假设，假设当前词只与其前一个词有关，则：$P(A_{i1},A_{i2},...,A_{in_i}) = P(A_{i1})P(A_{i2}|A_{i1})P(A_{i3}|A_{i2})...P(A_{in_i}|A_{i(n_i-1)})$

* 通过标准语料库，可以近似计算所有分词的二元条件概率，即：
  $$
  P(w_2|w_1) = \frac{P(w_1,w_2)}{P(w_1)} \approx \frac{freq(w_1,w_2)}{freq(w_1)}  \\
  P(w_1|w_2) = \frac{P(w_2,w_1)}{P(w_2)} \approx \frac{freq(w_1,w_2)}{freq(w_2)}
  $$

* `N`元模型实际应用中会出现一些问题：

  * 某些生僻词，或者相邻分词联合分布在语料库中没有，概率为0。这种情况我们一般会使用拉普拉斯平滑，即给它一个较小的概率值
  * 第二个问题是如果句子长，分词有很多情况，计算量也非常大，这时我们可以用下一节==维特比算法==来优化算法时间复杂度。

* `N`元模型即假设当前词与其前`N`个词相关

## 使用维特比算法分词

### 举个栗子

* 输入：`S=`~人生如梦境~，它在语料库中可能的概率图如下：

  ![图片来源：文本挖掘的分词原理](NLP%E4%B9%8B%E5%88%86%E8%AF%8D%E5%8E%9F%E7%90%86/image-20210309171556507.png)

* 使用维特比算法如下：

* 首先初始化有：$\delta(人) = 0.26\;\;\Psi(人)=Start\;\;\delta(人生) = 0.44\;\;\Psi(人生)=Start$

* 对节点`生`：$\delta(生) = \delta(人)P(生|人) = 0.0442 \;\; \Psi(生)=人$

* 对节点`如`：$\delta(如) = max\{\delta(生)P(如|生)，\delta(人生)P(如|人生)\} = max\{0.01680, 0.3168\} = 0.3168 \;\; \Psi(如) = 人生$

* 其他节点类似如下：
  $$
  \delta(如梦) = \delta(人生)P(如梦|人生) = 0.242 \;\; \Psi(如梦)=人生  \\
  \delta(梦) = \delta(如)P(梦|如) = 0.1996 \;\; \Psi(梦)=如  \\
  \delta(境) = max\{\delta(梦)P(境|梦) ,\delta(如梦)P(境|如梦)\}= max\{0.0359, 0.0315\} = 0.0359 \;\; \Psi(境)=梦  \\
  \delta(梦境) = \delta(如)P(梦境|如) = 0.1616 \;\; \Psi(梦境)=如  \\
  \delta(End) = max\{\delta(梦境)P(End|梦境), \delta(境)P(End|境)\} = max\{0.0396, 0.0047\} = 0.0396\;\;\Psi(End)=梦境 \\
  \Psi(End)=梦境 \to \Psi(梦境)=如 \to \Psi(如)=人生 \to \Psi(人生)=start
  $$

* 从而最终分词结果为~人生/如/梦境~

# 进阶

# 疑难

## 常用中文分词工具

1. [Jieba](https://github.com/fxsjy/jieba)：`star:25.7k`做最好的 Python 中文分词组件
2. [SnowNLP](https://github.com/isnowfy/snownlp)：`star:5.3k`Simplified Chinese Text Processing
3. [pkuseg](https://github.com/lancopku/pkuseg-python)：`star:5.3k`一个多领域中文分词工具包
4. [THULAC](https://github.com/thunlp/THULAC-Python)：`star:1.5k`一个高效的中文词法分析工具包
5. [其他](https://www.52nlp.cn/python%e4%b8%ad%e6%96%87%e5%88%86%e8%af%8d%e5%b7%a5%e5%85%b7-%e5%90%88%e9%9b%86-%e5%88%86%e8%af%8d%e5%ae%89%e8%a3%85-%e5%88%86%e8%af%8d%e4%bd%bf%e7%94%a8-%e5%88%86%e8%af%8d%e6%b5%8b%e8%af%95)：详情请见

# 参考

* [文本挖掘的分词原理](https://www.cnblogs.com/pinard/p/6677078.html)

