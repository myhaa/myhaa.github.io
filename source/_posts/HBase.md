---
title: HBase.md
author: Myhaa
top: false
cover: false
toc: true
mathjax: false
categories: AI/数据科学
tags:
  - 数据科学
  - HBase
date: 2020-06-11 11:19:55
img:
coverImg:
password:
summary: HBase笔记
---

# HBase基础

## 简介

```markdown
	HBase是一个开源的非关系型分布式数据库（NoSQL），它参考了谷歌的BigTable建模，实现的编程语言为 Java。它是Apache软件基金会的Hadoop项目的一部分，运行于HDFS文件系统之上，为 Hadoop 提供类似于BigTable 规模的服务。因此，它可以对稀疏文件提供极高的容错率。
```

# Getting Started

## 1. Introduction

* single-node
* standalone

## 2. 操作

### 2.1 启动

```shell
$ ./bin/hbase shell
hbase(main):001:0>
```

### 2.2 建表

* 必须指定表名`test`和列名`cf`

```hbase
hbase(main):001:0> create 'test', 'cf'
0 row(s) in 0.4170 seconds

=> Hbase::Table - test
```

### 2.3 列出表信息

```hbase
hbase(main):002:0> list 'test'
TABLE
test
1 row(s) in 0.0180 seconds

=> ["test"]
```

```hbase
hbase(main):003:0> describe 'test'
Table test is ENABLED
test
COLUMN FAMILIES DESCRIPTION
{NAME => 'cf', VERSIONS => '1', EVICT_BLOCKS_ON_CLOSE => 'false', NEW_VERSION_BEHAVIOR => 'false', KEEP_DELETED_CELLS => 'FALSE', CACHE_DATA_ON_WRITE => 'false', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', MIN_VERSIONS => '0', REPLICATION_SCOPE => '0', BLOOMFILTER => 'ROW', CACHE_INDEX_ON_WRITE => 'false', IN_MEMORY => 'false', CACHE_BLOOMS_ON_WRITE => 'false', PREFETCH_BLOCKS_ON_OPEN => 'false', COMPRESSION => 'NONE', BLOCKCACHE => 'true', BLOCKSIZE => '65536'}
1 row(s)
Took 0.9998 seconds
```

