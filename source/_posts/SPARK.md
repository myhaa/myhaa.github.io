---
title: Spark
date: 2019-12-12 11:21:11
author: Myhaa
img:
top: false
cover: false
coverImg:
password:
toc: true
mathjax: false
summary: Spark自学笔记
categories: AI/数据科学
tags:
  - Spark
  - 数据科学
---

# Spark基础

## 简介

### 概述

* Spark是UC Berkeley AMP lab所开源的类似Hadoop MapReduce的通用并行框架，它能更好地适用于数据挖掘与机器学习等需要迭代的MapReduce的算法。

### 主要特点

#### 运行速度快：

* Spark使用先进的DAG（Directed Acyclic Graph，有向无环图）执行引擎，以支持循环数据流与内存计算，基于内存的执行速度可比Hadoop MapReduce快上百倍，基于磁盘的执行速度也能快十倍；

#### 容易使用：

* Spark支持使用Scala、Java、Python和R语言进行编程，简洁的API设计有助于用户轻松构建并行程序，并且可以通过Spark Shell进行交互式编程；

#### 通用性：

* Spark提供了完整而强大的技术栈，包括SQL查询、流式计算、机器学习和图算法组件，这些组件可以无缝整合在同一个应用中，足以应对复杂的计算；

#### 运行模式多样：

* Spark可运行于独立的集群模式中，或者运行于Hadoop中，也可运行于Amazon EC2等云环境中，并且可以访问HDFS、Cassandra、HBase、Hive等多种数据源。
* Spark的计算模式也属于MapReduce，但不局限于Map和Reduce操作，还提供了多种数据集操作类型，编程模型比MapReduce更灵活；
* Spark提供了内存计算，中间结果直接放到内存中，带来了更高的迭代运算效率；
* Spark基于DAG的任务调度执行机制，要优于MapReduce的迭代执行机制。

## 生态系统

### 大数据处理类型

#### 复杂的批量数据处理：

* 时间跨度通常在数十分钟到数小时之间；
* Hadoop MapReduce

#### 基于历史数据的交互式查询：

* 时间跨度通常在数十秒到数分钟之间；
* Impala：Impala与Hive相似，但底层引擎不同，提供了实时交互式
* SQL查询

#### 基于实时数据流的数据处理：

* 时间跨度通常在数百毫秒到数秒之间
* Storm：开源流计算框架

#### Spark所提供的生态系统足以应对上述三种场景

* 即同时支持批处理、交互式查询和流数据处理。

### 框架

#### 访问和接口

##### Spark Streaming

https://spark.apache.org/docs/latest/streaming-programming-guide.html

Spark Streaming支持高吞吐量、可容错处理的实时流数据处理，其核心思路是将流式计算分解成一系列短小的批处理作业。Spark Streaming支持多种数据输入源，如Kafka、Flume和TCP套接字等；

kafka：https://www.w3cschool.cn/apache_kafka/

Flume：https://juejin.im/post/5be4e549f265da61441f8dbe

TCP：http://www.ruanyifeng.com/blog/2017/06/tcp-protocol.html

##### BlinkDB

* BlinkDB 是一个用于在海量数据上运行交互式 SQL 查询的大规模并行查询引擎。

##### Spark Sql

* https://spark.apache.org/docs/latest/sql-programming-guide.html
* Spark SQL允许开发人员直接处理RDD，同时也可查询Hive、HBase等外部数据源。Spark SQL的一个重要特点是其能够统一处理关系表和RDD，使得开发人员可以轻松地使用SQL命令进行查询，并进行更复杂的数据分析；

##### GraphX

* https://spark.apache.org/docs/latest/graphx-programming-guide.html
* GraphX是Spark中用于图计算的API，可认为是Pregel在Spark上的重写及优化，Graphx性能良好，拥有丰富的功能和运算符，能在海量数据上自如地运行复杂的图算法。

##### MLBase

* https://amplab.cs.berkeley.edu/publication/mlbase-a-distributed-machine-learning-system/

##### MlLib

* https://spark.apache.org/docs/latest/ml-guide.html
* MLlib提供了常用机器学习算法的实现，包括聚类、分类、回归、协同过滤等，降低了机器学习的门槛，开发人员只要具备一定的理论知识就能进行机器学习的工作；

#### 处理引擎

##### Spark Core

* https://blog.csdn.net/bingoxubin/article/details/79076978
* Spark Core包含Spark的基本功能，如内存计算、任务调度、部署模式、故障恢复、存储管理等。Spark建立在统一的抽象RDD之上，使其可以以基本一致的方式应对不同的大数据处理场景；通常所说的Apache Spark，就是指Spark Core；

#### 存储

##### Tachyon

* https://www.ibm.com/developerworks/cn/opensource/os-cn-spark-tachyon/

##### HDFS

* https://hadoop.apache.org/docs/r1.0.4/cn/hdfs_design.html

##### Amazon S3

* https://aws.amazon.com/cn/s3/

#### 资源管理调度

##### Mesos

* http://mesos.apache.org/

##### Hadoop YARN

* https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html

## 运行架构

### 基本概念

#### RDD：

* 是弹性分布式数据集（Resilient Distributed Dataset）的简称，是分布式内存的一个抽象概念，提供了一种高度受限的共享内存模型；

#### DAG：

* 是Directed Acyclic Graph（有向无环图）的简称，反映RDD之间的依赖关系；

#### Executor：

* 是运行在工作节点（Worker Node）上的一个进程，负责运行任务，并为应用程序存储数据；

##### Spark所采用的Executor有两个优点

* 一是利用多线程来执行具体的任务（Hadoop MapReduce采用的是进程模型），减少任务的启动开销；
* 二是Executor中有一个BlockManager存储模块，会将内存和磁盘共同作为存储设备，当需要多轮迭代计算时，可以将中间结果存储到这个存储模块里，下次需要时，就可以直接读该存储模块里的数据，而不需要读写到HDFS等文件系统里，因而有效减少了IO开销；或者在交互式查询场景下，预先将表缓存到该存储系统上，从而可以提高读写IO性能。

#### 应用：

* 用户编写的Spark应用程序；

#### 任务：

* 运行在Executor上的工作单元；

#### 作业：

* 一个作业包含多个RDD及作用于相应RDD上的各种操作；

#### 阶段：

* 是作业的基本调度单位，一个作业会分为多组任务，每组任务被称为“阶段”，或者也被称为“任务集”。

### 架构设计

#### 集群资源管理器（Cluster Manager）

* 可以是Spark自带的资源管理器，也可以是YARN或Mesos等资源管理框架。

#### 运行作业任务的工作节点（Worker Node）

#### 每个应用的任务控制节点（Driver）

#### 每个工作节点上负责具体任务的执行进程（Executor）

### 运行流程

1. 当一个Spark应用被提交时，首先需要为这个应用构建起基本的运行环境，即由任务控制节点（Driver）创建一个SparkContext，由SparkContext负责和资源管理器（Cluster Manager）的通信以及进行资源的申请、任务的分配和监控等。SparkContext会向资源管理器注册并申请运行Executor的资源；
2. 资源管理器为Executor分配资源，并启动Executor进程，Executor运行情况将随着“心跳”发送到资源管理器上；
3. SparkContext根据RDD的依赖关系构建DAG图，DAG图提交给DAG调度器（DAGScheduler）进行解析，将DAG图分解成多个“阶段”（每个阶段都是一个任务集），并且计算出各个阶段之间的依赖关系，然后把一个个“任务集”提交给底层的任务调度器（TaskScheduler）进行处理；Executor向SparkContext申请任务，任务调度器将任务分发给Executor运行，同时，SparkContext将应用程序代码发放给Executor；
4. 任务在Executor上运行，把执行结果反馈给任务调度器，然后反馈给DAG调度器，运行完毕后写入数据并释放所有资源。

### 运行架构特点

			*  每个应用都有自己专属的Executor进程，并且该进程在应用运行期间一直驻留。Executor进程以多线程的方式运行任务，减少了多进程任务频繁的启动开销，使得任务执行变得非常高效和可靠；
			*  Spark运行过程与资源管理器无关，只要能够获取Executor进程并保持通信即可；
			*  Executor上有一个BlockManager存储模块，类似于键值存储系统（把内存和磁盘共同作为存储设备），在处理迭代计算任务时，不需要把中间结果写入到HDFS等文件系统，而是直接放在这个存储系统上，后续有需要时就可以直接读取；在交互式查询场景下，也可以把表提前缓存到这个存储系统上，提高读写IO性能；
			*  任务采用了数据本地性和推测执行等优化机制。数据本地性是尽量将计算移到数据所在的节点上进行，即“计算向数据靠拢”，因为移动计算比移动数据所占的网络资源要少得多。而且，Spark采用了延时调度机制，可以在更大的程度上实现执行过程优化。比如，拥有数据的节点当前正被其他的任务占用，那么，在这种情况下是否需要将数据移动到其他的空闲节点呢？答案是不一定。因为，如果经过预测发现当前节点结束当前任务的时间要比移动数据的时间还要少，那么，调度就会等待，直到当前节点可用。

## RDD的设计与运行原理

### RDD概念

* 一个RDD就是一个分布式对象集合，本质上是一个只读的分区记录集合，每个RDD可以分成多个分区，每个分区就是一个数据集片段，并且一个RDD的不同分区可以被保存到集群中不同的节点上，从而可以在集群中的不同节点上进行并行计算。

* RDD提供了一种高度受限的共享内存模型，即RDD是只读的记录分区的集合，不能直接修改，只能基于稳定的物理存储中的数据集来创建RDD，或者通过在其他RDD上执行确定的转换操作（如map、join和groupBy）而创建得到新的RDD。

* RDD提供了一组丰富的操作以支持常见的数据运算，分为“行动”（Action）和“转换”（Transformation）两种类型，

  1. 行动：行动操作（比如count、collect等）接受RDD但是返回非RDD（即输出一个值或结果）
  2. 转换：转换操作（比如map、filter、groupBy、join等）接受RDD并返回RDD

  **注意事项：**

  适合：对于数据集中元素执行相同操作的批处理式应用

  不适合：不适合用于需要异步、细粒度状态的应用，比如Web应用系统、增量式的网页爬虫等

#### 执行过程

* RDD读入外部数据源（或者内存中的集合）进行创建；
* RDD经过一系列的“转换”操作，每一次都会产生不同的RDD，供给下一个“转换”使用；
* 最后一个RDD经“行动”操作进行处理，并输出到外部数据源（或者变成Scala集合或标量）。

### RDD特性

#### 高效的容错性。

* 在RDD的设计中，数据只读，不可修改，如果需要修改数据，必须从父RDD转换到子RDD，由此在不同RDD之间建立了血缘关系。所以，RDD是一种天生具有容错机制的特殊集合，不需要通过数据冗余的方式（比如检查点）实现容错，而只需通过RDD父子依赖（血缘）关系重新计算得到丢失的分区来实现容错，无需回滚整个系统，这样就避免了数据复制的高开销，而且重算过程可以在不同节点之间并行进行，实现了高效的容错。
* RDD提供的转换操作都是一些粗粒度的操作（比如map、filter和join），RDD依赖关系只需要记录这种粗粒度的转换操作，而不需要记录具体的数据和各种细粒度操作的日志（比如对哪个数据项进行了修改），这就大大降低了数据密集型应用中的容错开销；

#### 中间结果持久化到内存。

* 数据在内存中的多个RDD操作之间进行传递，不需要“落地”到磁盘上，避免了不必要的读写磁盘开销；

#### 存放的数据可以是Java对象，

* 避免了不必要的对象序列化和反序列化开销。

### RDD间的依赖关系

#### 窄依赖（Narrow Dependency）

* 表现为一个父RDD的分区对应于一个子RDD的分区，或多个父RDD的分区对应于一个子RDD的分区；
* 窄依赖典型的操作包括map、filter、union等
* 对于窄依赖的RDD，可以以流水线的方式计算所有父分区，不会造成网络之间的数据混合。

#### 宽依赖（Wide Dependency）

* 表现为存在一个父RDD的一个分区对应一个子RDD的多个分区
* 宽依赖典型的操作包括groupByKey、sortByKey等
* 对于宽依赖的RDD，则通常伴随着Shuffle操作，即首先需要计算好所有父分区数据，然后在节点之间进行Shuffle。

**比较：**

* 相对而言，在两种依赖关系中，窄依赖的失败恢复更为高效，它只需要根据父RDD分区重新计算丢失的分区即可（不需要重新计算所有分区），而且可以并行地在不同节点进行重新计算。
* 而对于宽依赖而言，单个节点失效通常意味着重新计算过程会涉及多个父RDD分区，开销较大。
* 此外，Spark还提供了数据检查点和记录日志，用于持久化中间RDD，从而使得在进行失败恢复时不需要追溯到最开始的阶段。
* 在进行故障恢复时，Spark会对数据检查点开销和重新计算RDD分区的开销进行比较，从而自动选择最优的恢复策略。

### 阶段的划分

* 在DAG中进行反向解析，遇到宽依赖就断开，遇到窄依赖就把当前的RDD加入到当前的阶段中；将窄依赖尽量划分在同一个阶段中，可以实现流水线计算（具体的阶段划分算法请参见AMP实验室发表的论文《Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing》）

### RDD运行过程

1. 创建RDD对象；
2. SparkContext负责计算RDD之间的依赖关系，构建DAG；
3. DAGScheduler负责把DAG图分解成多个阶段，每个阶段中包含了多个任务，每个任务会被任务调度器分发给各个工作节点（Worker Node）上的Executor去执行。

## SPARK的部署模式

### 三种部署模式

#### standalone模式

* Spark框架本身也自带了完整的资源调度管理服务，可以独立部署到一个集群中，而不需要依赖其他系统来为其提供资源管理调度服务。

#### Spark on Mesos模式

* Spark运行在Mesos上，要比运行在YARN上更加灵活、自然。目前，Spark官方推荐采用这种模式，

#### Spark on YARN模式

* Spark可运行于YARN之上，与Hadoop进行统一部署，即“Spark on YARN”，

### 从“Hadoop+Storm”架构转向Spark架构

#### 采用“Hadoop+Storm”部署方式的一个案例

​				大数据层
​					数据存储
​						HDFS
​						HBase
​						Cassandra
​					调度器
​						YARN
​						Mesos
​					实时查询
​						Redis
​						Solr
​						HBase
​					离线分析
​						Hive
​						Pig
​						Impala
​						MapReduce
​					用户行为实时分析
​						UV
​						PV
​					Storm实时流处理
​						欺诈监控
​						系统报警
​						点击流推荐
​				数据收集
​					业务数据收集
​						Flume
​						Kafka
​						ETL
​					网站数据收集
​						Collector
​					用户行为数据收集
​						PV/UV
​						点击流信息
​						导航数据收集
​				业务应用层
​					应用数据
​						导航日志
​						应用日志
​					系统数据
​						系统日志
​						报警数据
​				繁琐！

#### Spark架构优点

* 实现一键式安装和配置、线程级别的任务监控和告警；
* 降低硬件集群、软件维护、任务监控和应用开发的难度；
* 便于做成统一的硬件、计算平台资源池。
* Spark Streaming的原理是将流数据分解成一系列短小的批处理作业，每个短小的批处理作业使用面向批处理的Spark Core进行处理，通过这种方式变相实现流计算，而不是真正实时的流计算，因而通常无法实现毫秒级的响应

#### Hadoop和Spark的统一部署

* 一方面，由于Hadoop生态系统中的一些组件所实现的功能，目前还是无法由Spark取代的，比如，Storm可以实现毫秒级响应的流计算，但是，Spark则无法做到毫秒级响应。
* 另一方面，企业中已经有许多现有的应用，都是基于现有的Hadoop组件开发的，完全转移到Spark上需要一定的成本。
* 实时计算工具
  1. storm：https://www.w3cschool.cn/apache_storm/
  2. flink：https://ci.apache.org/projects/flink/flink-docs-release-1.8/tutorials/local_setup.html

# Spark进阶

# 参考书籍

* [大数据技术原理与应用](<https://study.163.com/course/courseMain.htm?courseId=1002887002>)
* [SPARK编程指南-Python](<https://study.163.com/course/courseMain.htm?courseId=1209408816>)
* [w3school](https://www.w3cschool.cn/spark/)
* [厦门大学博客](http://dblab.xmu.edu.cn/blog/1709-2/)
* [SPARK官网](https://spark.apache.org/docs/latest/index.html)



# 疑难解答

