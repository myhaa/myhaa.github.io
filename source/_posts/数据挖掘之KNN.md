---
title: 数据挖掘之KNN
author: Myhaa
top: false
cover: false
toc: true
mathjax: true
categories: AI/数据科学
tags:
  - 数据挖掘
  - 十大算法
date: 2021-03-15 09:37:08
img:
coverImg:
password:
summary: 数据挖掘十大算法之KNN
---

# 基础

## 算法

* 输入：

  * $x$：待分类的预测样本，
  * $k$：最近的$k$个作为标准
  * $T = \lbrace (x_1,y_1),(x_2,y_2), \cdots ,(x_n,y_n) \rbrace$：训练集，是多维特征空间向量，其中每个训练样本带有一个类别标签
  * 距离度量：一般情况下，将[欧氏距离](https://zh.wikipedia.org/wiki/欧氏距离)作为距离度量，但是这是只适用于连续变量。在文本分类这种离散变量情况下，另一个度量——**重叠度量**（或[海明距离](https://zh.wikipedia.org/wiki/海明距离)）可以用来作为度量

* 输出：

  * $y$：待分类预测样本$x$对应的类别

* 决策规则：
  $$
  \begin{equation} 
  y = \arg \mathop {\max }\limits_{c_j} \sum\limits_{x_i \in n_k(x)} I(y_i = c_j),\ i = 1,2, \cdots ,n; \ j = 1,2, \cdots ,k 
  \label{eq:obj} 
  \end{equation}
  $$

  * 找出训练样本集中与预测样本$x$距离相近的$k$个样本，根据这$k$个样本对应的类别，采取**多数表决**的规则来确定预测样本$x$的类别

## 参数选择

### $k$

* 在二元（两类）分类问题中，选取$k$为奇数有助于避免两个分类平票的情形
* 在此问题下，选取最佳经验$k$值的方法是[自助法](https://zh.wikipedia.org/wiki/自助法)
* 常用的是交叉验证方法来选取$k$

### 距离度量

* 一般情况下，将[欧氏距离](https://zh.wikipedia.org/wiki/欧氏距离)作为距离度量，但是这是只适用于连续变量。在文本分类这种离散变量情况下，另一个度量——**重叠度量**（或[海明距离](https://zh.wikipedia.org/wiki/海明距离)）可以用来作为度量

# 进阶

## 修正之加权`KNN`

* 上面提到的计算规则中，$k$邻域的样本点对预测结果的贡献度是相等的

* 但我们直观理解，距离更近的样本点应有更大的相似度，其贡献度应比距离更远的样本点大

* 所以可以加上权值$w_i = \frac{1}{\left\| {x_i - x} \right\|}$进行修正，则式(1)变成：
  $$
  \begin{equation} 
  y = \arg \mathop {\max }\limits_{c_j} \sum\limits_{x_i \in n_k(x)} {w_i \times I(y_i = c_j)},\ i = 1,2, \cdots ,n; \ j = 1,2, \cdots ,k 
  \label{eq:obj1} 
  \end{equation}
  $$
  

## 加快计算之`KD`树



# 疑难

# 参考

* [Michael Steinbach and Pang-Ning Tan, The Top Ten Algorithms in Data Mining.]()
* [wiki-K近邻算法](https://zh.wikipedia.org/wiki/K-%E8%BF%91%E9%82%BB%E7%AE%97%E6%B3%95)
* [【十大经典数据挖掘算法】kNN](https://www.cnblogs.com/en-heng/p/5000628.html)