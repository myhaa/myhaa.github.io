---
title: google-MapReduce
author: Myhaa
top: false
cover: false
toc: true
mathjax: false
categories: Google Three Papers
tags:
  - 数据科学
  - 读书笔记
  - Hadoop
date: 2020-06-03 09:25:31
img:
coverImg:
password:
summary: MapReduce的原理介绍
---

# Title

![image-20200603093104618](google_MapReduce/image-20200603093104618.png)

【译文】

* MapReduce：大型集群上的简化数据处理

## Abstract

【原文】

```markdown
	MapReduce is a programming model and an associated implementation for processing and generating large data sets. Users specify a map function that processes a key/value pair to generate a set of intermediate key/value pairs, and a reduce function that merges all intermediate values associated with the same intermediate key. Many real world tasks are expressible in this model, as shown in the paper.

	Programs written in this functional style are automatically parallelized and executed on a large cluster of commodity machines. The run-time system takes care of the details of partitioning the input data, scheduling the program’s execution across a set of machines, handling machine failures, and managing the required inter-machine communication. This allows programmers without any experience with parallel and distributed systems to easily utilize the resources of a large distributed system.

	Our implementation of MapReduce runs on a large cluster of commodity machines and is highly scalable: a typical MapReduce computation processes many terabytes of data on thousands of machines. Programmers find the system easy to use: hundreds of MapReduce programs have been implemented and upwards of one thousand MapReduce jobs are executed on Google’s clusters every day.
```

【译文】

```markdown
	MapReduce是用于处理和生成大数据集的编程模型（相关的实现）。 用户指定key\value对以生成一组中间key\value对的map函数，以及指定归纳与同一中间key\value关联的所有中间key\value的reduce函数。 如本文所示，许多现实世界的任务在这种模型中都是可以表达的。

	用这种函数式编写的程序会自动并行化，并在大型计算机集群上执行。运行时系统负责对输入数据进行分区、安排跨机器的程序执行、处理机器故障和管理所需的机器间通信等细节。这使得没有任何并行和分布式系统经验的程序员可以轻松地利用大型分布式系统的资源。

	我们的MapReduce实现运行在大量的普通机器上，并且具有高度的可伸缩性:典型的MapReduce计算在数千台机器上处理许多TB级的数据。程序员发现这个系统很容易使用:已经实现了数百个MapReduce程序，每天在谷歌集群上执行的MapReduce任务都超过1000个。
```

【重点】

* MapReduce是用于处理和生成大数据集的编程模型（相关的实现）
* 包含map函数和reduce函数，使用key\value对
* 高度的可伸缩性

## 1、Introduction

【原文】

```markdown
	Over the past five years, the authors and many others at Google have implemented hundreds of special-purpose computations that process large amounts of raw data, such as crawled documents, web request logs, etc., to compute various kinds of derived data, such as inverted indices, various representations of the graph structure of web documents, summaries of the number of pages crawled per host, the set of most frequent queries in a given day, etc. Most such computations are conceptually straightforward. However, the input data is usually large and the computations have to be distributed across hundreds or thousands of machines in order to finish in a reasonable amount of time. The issues of how to parallelize the computation, distribute the data, and handle failures conspire to obscure the original simple computation with large amounts of complex code to deal with these issues.

	As a reaction to this complexity, we designed a new abstraction that allows us to express the simple computations we were trying to perform but hides the messy details of parallelization, fault-tolerance, data distribution and load balancing in a library. Our abstraction is inspired by the map and reduce primitives present in Lisp and many other functional languages. We realized that most of our computations involved applying a map operation to each logical “record” in our input in order to compute a set of intermediate key/value pairs, and then applying a reduce operation to all the values that shared the same key, in order to combine the derived data appropriately. Our use of a functional model with userspecified map and reduce operations allows us to parallelize large computations easily and to use re-execution as the primary mechanism for fault tolerance.

	The major contributions of this work are a simple and powerful interface that enables automatic parallelization and distribution of large-scale computations, combined with an implementation of this interface that achieves high performance on large clusters of commodity PCs.

	Section 2 describes the basic programming model and gives several examples. Section 3 describes an implementation of the MapReduce interface tailored towards our cluster-based computing environment. Section 4 describes several refinements of the programming model that we have found useful. Section 5 has performance measurements of our implementation for a variety of tasks. Section 6 explores the use of MapReduce within Google including our experiences in using it as the basis for a rewrite of our production indexing system. Section 7 discusses related and future work.

```

【译文】

```markdown
	在过去的五年中，Google的作者和许多其他人已经实现了数百种特殊用途的计算，这些计算处理大量的原始数据（例如抓取的文档，Web请求日志等），以计算各种派生数据，例如：作为反向索引，Web文档的图形结构的各种表示形式，每个主机爬取的网页摘要，给定一天中最频繁的查询集等。大多数此类计算在概念上都很简单。 但是，输入数据通常很大，并且必须在数百或数千台计算机上分布计算，才能在合理的时间内完成计算。 如何并行化计算，分配数据和处理故障的问题，用大量复杂的代码来处理这些问题，使原来简单的计算变得模糊不清。
	
	为了应对这种复杂性，我们设计了一个新的抽象，该抽象使我们能够表达我们试图执行的简单计算，但在库中隐藏了并行化，容错，数据分发和负载平衡的混乱细节。 Lisp和许多其他功能语言中的map和reduce原语启发了我们的抽象。 我们意识到，大多数计算都涉及对输入中的每个逻辑“记录”应用映射操作，以便计算一组key/value键/值对，然后对共享同一key的所有值应用归约操作，适当地组合得出的数据。我们使用具有用户指定的映射和归约运算的功能模型，使我们能够轻松地并行进行大型计算，并将重新执行用作容错的主要机制。
	
	这项工作的主要贡献是一个简单而强大的界面，该界面可实现大规模计算的自动并行化和分配，并结合了该界面的实现，可在大型商用PC集群上实现高性能。
	
	第2节描述了基本的编程模型，并给出了一些示例。 第3节介绍了针对我们基于集群的计算环境量身定制的MapReduce接口的实现。 第4节描述了一些有用的编程模型改进。 第5节对我们执行各种任务的性能进行了度量。 第6节探讨了MapReduce在Google中的用法，包括我们使用它作为重写生产索引系统基础的经验。 第7节讨论相关和未来的工作。
```

