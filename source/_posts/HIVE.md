---
title: Hive笔记
date: 2019-10-12 17:21:11
author: Myhaa
img:
top: false
cover: false
coverImg:
password:
toc: true
mathjax: false
summary: 有关Hive的笔记
categories: AI/数据科学
tags:
  - 数据科学
  - Hadoop
  - Hive
---

# 一、Hive基础

## 1、Hive数据类型

（1）原始类型

|    类型     |         描述         |     示例     |
| :---------: | :------------------: | :----------: |
|  `Boolean`  |   `true`、`false`    |    `TRUE`    |
|  `TINYINT`  |       -128~127       |     `1Y`     |
| `SMALLINT`  |     -32768~32767     |     `1S`     |
|    `INT`    | 4个字节的带符号整数  |      1       |
|  `BIGINT`   |   8字节带符号整数    |     `1L`     |
|   `FLOAT`   |  4字节单精度浮点数   |     1.0      |
|  `DOUBLE`   |  8字节双精度浮点数   |     1.0      |
|  `DECIMAL`  | 任意精度的带符号小数 |     1.0      |
|  `STRING`   |        字符串        |   `‘ABC’`    |
|  `VARCHAR`  |       长字符串       |              |
|   `CHAR`    |    固定长度字符串    |              |
|  `BINARY`   |       字节数组       |              |
| `TIMESTAMP` |   时间戳，纳秒精度   |              |
|   `DATE`    |         日期         | `2019-10-08` |

（2）复杂类型

|   类型   |                        描述                         |           示例           |
| :------: | :-------------------------------------------------: | :----------------------: |
| `ARRAY`  |                有序的的同类型的集合                 |       `array(1,2)`       |
|  `MAP`   | key-value<br />key必须为原始类型，value可以任意类型 |    `map(‘a’,1,’b’,2)`    |
| `UNION`  |              在有限取值范围内的一个值               | `create_union(1,’a’,63)` |
| `STRUCT` |                字段集合,类型可以不同                |   `struct(‘1’,1,1.0)`    |

## 2、Hive数据库操作

（1）创建数据库

```mysql
-- 命令
CREATE DATABASE|SCHEMA [IF NOT EXISTS] <database name>;

-- 例子
CREATE DATABASE IF NOT EXISTS database_name;
```

（2）删除数据库

```mysql
-- 命令
DROP DATABASE StatementDROP (DATABASE|SCHEMA) [IF EXISTS] database_name [RESTRICT|CASCADE];

-- 例子
DROP DATABASE IF EXISTS userdb;
```

## 3、Hive表操作

（1）创建表

```mysql
-- 命令
CREATE [TEMPORARY] [EXTERNAL] TABLE [IF NOT EXISTS] [db_name.] table_name
[(col_name data_type [COMMENT col_comment], ...)]
[COMMENT table_comment]
[ROW FORMAT row_format]
[STORED AS file_format]

-- 例子
CREATE TABLE IF NOT EXISTS employee
(eid int, 
 name String,
 salary String, 
 destination String)
COMMENT ‘Employee details’
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ‘\t’
LINES TERMINATED BY ‘\n’
STORED AS TEXTFILE;
```

（2）修改表

```mysql
-- 重命名表
ALTER TABLE name RENAME TO new_name

ALTER TABLE employee RENAME TO emp;

-- 添加列
ALTER TABLE name ADD COLUMNS (col_spec[, col_spec ...])

ALTER TABLE employee ADD COLUMNS (
    dept STRING COMMENT 'Department name');

-- 删除列
ALTER TABLE name DROP [COLUMN] column_name

-- 更改列属性
ALTER TABLE name CHANGE column_name new_name new_type

ALTER TABLE employee CHANGE salary salary Double;

-- 替换列
ALTER TABLE name REPLACE COLUMNS (col_spec[, col_spec ...])

ALTER TABLE employee REPLACE COLUMNS (
    eid INT empid Int,
    ename STRING name String);
```

（3）删除表

```mysql
DROP TABLE [IF EXISTS] table_name;
```

## 4、Hive表数据操作

（1）插入数据

* `LOAD DATA`：从`HDFS`中加载数据会直接`move`！

```mysql
--命令
LOAD DATA [LOCAL] INPATH 'filepath' [OVERWRITE] INTO TABLE tablename [PARTITION (partcol1=val1, partcol2=val2 ...)]

--例子
LOAD DATA LOCAL INPATH '/home/user/sample.txt' OVERWRITE INTO TABLE employee;
```

* `INSERT INTO`

```mysql
insert into table account 
select id, age, name from account_tmp;
```

* `INSERT OVERWRITE`

```mysql
insert overwrite table account2 
select id, age, name from account_tmp;
```

***注意***：

`insert overwrite `会覆盖已经存在的数据，假如原始表使用`overwrite `上述的数据，先现将原始表的数据`remove`，再插入新数据。

`insert into`只是简单的插入，不考虑原始表的数据，直接追加到表中。

（2）修改数据

```mysql
UPDATE tablename SET column = value [, column = value ...] [WHERE expression]
```

（3）删除数据

```mysql
DELETE FROM tablename [WHERE expression]
```

## 5、Hive分区操作

（1）添加分区

```mysql
-- 命令
ALTER TABLE table_name ADD [IF NOT EXISTS] PARTITION (p_column = p_col_value, p_column = p_col_value, ...)
[LOCATION 'location1'] (p2_column = p2_col_value, p2_column = p2_col_value, ...) [LOCATION 'location2'] ...;

-- 例子
ALTER TABLE employee ADD PARTITION (year=’2013’) location '/2012/part2012';
```

（2）重命名分区

```mysql
-- 命令
ALTER TABLE table_name PARTITION partition_spec RENAME TO PARTITION partition_spec;

-- 例子
ALTER TABLE employee PARTITION (year=’1203’) RENAME TO PARTITION (Yoj=’1203’);
```

（3）删除分区

```mysql
-- 命令
ALTER TABLE table_name DROP [IF EXISTS] PARTITION partition_spec, PARTITION partition_spec,...;

-- 例子
ALTER TABLE employee DROP [IF EXISTS] PARTITION (year=’1203’);
```

## 6、Hive内置运算符

（1）关系运算符

| 运算符        | 操作         | 描述                                            |
| ------------- | ------------ | ----------------------------------------------- |
| A = B         | 所有基本类型 |                                                 |
| A != B        | 所有基本类型 |                                                 |
| A < B         | 所有基本类型 |                                                 |
| A <= B        | 所有基本类型 |                                                 |
| A > B         | 所有基本类型 |                                                 |
| A >= B        | 所有基本类型 |                                                 |
| A IS NULL     | 所有类型     |                                                 |
| A IS NOT NULL | 所有类型     |                                                 |
| A LIKE B      | 字符串       | TRUE，如果字符串模式A匹配到B，<br />否则FALSE。 |
| A RLIKE B     | 字符串       | TRUE：A任何子字符串匹配Java正则表达式B；        |
| A REGEXP B    | 字符串       | 等同于RLIKE.                                    |

（2）算术运算符

| 运算符 | 操作         | 描述               |
| ------ | ------------ | ------------------ |
| A + B  | 所有数字类型 |                    |
| A - B  | 所有数字类型 |                    |
| A * B  | 所有数字类型 |                    |
| A / B  | 所有数字类型 |                    |
| A % B  | 所有数字类型 |                    |
| A & B  | 所有数字类型 | A和B的按位与结果   |
| A \| B | 所有数字类型 | A和B的按位或结果   |
| A ^ B  | 所有数字类型 | A和B的按位异或结果 |
| ~ A    | 所有数字类型 | A按位非的结果      |

（3）逻辑运算符

| 运算符   | 操作    | 描述 |
| -------- | ------- | ---- |
| A AND B  | Boolean | 与   |
| A && B   | Boolean |      |
| A OR B   | Boolean | 或   |
| A \|\| B | Boolean |      |
| NOT A    | Boolean | 非   |
| ! A      | Boolean |      |

（4）复杂运算符

| 运算符 | 操作                                | 描述                                          |
| ------ | ----------------------------------- | --------------------------------------------- |
| A[n]   | A是一个数组，n是一个int             | 它返回数组A的第n+1个元素，第一个元素的索引0。 |
| M[Key] | M 是一个 Map<K, V> 并 key 的类型为K | 它返回对应于映射中关键字的值。                |
| S.x    | S 是一个结构                        | 它返回S的s字段                                |

## 7、Hive内置函数

```mysql
-- 返回BIGINT最近的double值。
round(double a)

-- 返回最大BIGINT值等于或小于double。
floor(double a)

-- 它返回最小BIGINT值等于或大于double。
ceil(double a)

-- 它返回一个随机数，从行改变到行。
rand()
rand(int seed)

-- 它返回从A后串联B产生的字符串
concat(string A, string B, ...)

-- 它返回一个起始位置start到A结束的子字符串
substr(string A, int start)

-- 返回从给定长度length的从起始位置start开始的字符串。
substr(string A, int start, int length)

-- 它返回转换所有字符为大写的字符串。
upper(string A)
ucase(string A)

-- 它返回转换所有字符为小写的字符串。
lower(string A)
lcase(string A)

-- 它返回字符串从A两端修剪空格的结果
trim(string A)

-- 它返回从A左边开始修整空格产生的字符串(左手侧)
ltrim(string A)

-- 它返回从A右边开始修整空格产生的字符串(右侧)
rtrim(string A)

-- 它返回将A中的子字符串B替换为C的全新字符串
regexp_replace(string A, string B, string C)

-- 它返回在映射类型或数组类型的元素的数量。
size(Map<K.V>)
size(Array<T>)

-- 将字段expr的数据类型转换为type。如果转换不成功，返回的是NULL。
cast(<expr> as <type>)

-- 将10位的时间戳值unixtime转为日期函数
from_unixtime(int unixtime, 'yyyy-MM-dd HH:mm:ss')

-- 返回一个字符串时间戳的日期部分：to_date("1970-01-01 00:00:00") = "1970-01-01"
to_date(string timestamp, 'yyyy-MM-dd')

-- 返回指定日期的unix时间戳
unix_timestamp(string date, 'yyyy-MM-dd HH:mm:ss')  -- date的形式必须为’yyyy-MM-dd HH:mm:ss’的形式
unix_timestamp()  -- 返回当前时间的unix时间戳

-- 返回时间字段中的年月日
year(string date, 'yyMMdd')  -- 年
month(string date, 'yyMMdd')  -- 月
day(string date, 'yyMMdd')  -- 日

-- 返回时间字段是本年的第多少周
weekofyear(string date, 'yyMMdd')

-- 返回enddate与begindate之间的时间差的天数
datediff(string enddate,string begindate)

select datediff(‘2016-06-01’,’2016-05-01’) from Hive_sum limit 1;

-- 返回date增加days天后的日期
date_add(string date,int days)

-- 返回date减少days天后的日期
date_sub(string date,int days)

-- 提取从基于指定的JSON路径的JSON字符串JSON对象，并返回提取的JSON字符串的JSON对象。
get_json_object(string json_string, string path)

-- 返回检索行的总数。
count(*)
count(expr)

-- 返回该组或该组中的列的不同值的分组和所有元素的总和。
sum(col)
sum(DISTINCT col)

-- 返回上述组或该组中的列的不同值的元素的平均值。
avg(col)
avg(DISTINCT col)

-- 返回该组中的列的最大最小值。
min(col)
max(col)
```

## 8、Hive视图和索引

（1）创建视图

```mysql
-- 命令
CREATE VIEW [IF NOT EXISTS] view_name [(column_name [COMMENT column_comment], ...) ]
[COMMENT table_comment]
AS SELECT ...

-- 例子
CREATE VIEW emp_30000 AS
SELECT * FROM employee
WHERE salary>30000;
```

（2）删除视图

```mysql
DROP VIEW view_name
```

（3）创建索引

```mysql
-- 命令
CREATE INDEX index_name
ON TABLE base_table_name (col_name, ...)
AS 'index.handler.class.name'
[WITH DEFERRED REBUILD]
[IDXPROPERTIES (property_name=property_value, ...)]
[IN TABLE index_table_name]
[PARTITIONED BY (col_name, ...)]
[
   [ ROW FORMAT ...] STORED AS ...
   | STORED BY ...
]
[LOCATION hdfs_path]
[TBLPROPERTIES (...)]

-- 例子：使用字段 Id, Name, Salary, Designation, 和 Dept创建一个名为index_salary的索引，对employee 表的salary列索引。
CREATE INDEX inedx_salary ON TABLE employee(salary)
AS 'org.apache.hadoop.Hive.ql.index.compact.CompactIndexHandler';
```

（4）删除索引

```mysql
DROP INDEX <index_name> ON <table_name>
```

# 二、Hive进阶

## 1、Hive SELECT 数据

```mysql
SELECT [ALL | DISTINCT] select_expr, select_expr, ... 
FROM table_reference 
[WHERE where_condition] 
[GROUP BY col_list] 
[HAVING having_condition] 
[CLUSTER BY col_list | [DISTRIBUTE BY col_list] [SORT BY col_list]] 
[ORDER BY col_list]
[LIMIT number];
```

## 2、命令执行顺序

```mysql
-- 大致顺序
from... where.... select... group by... having ... order by...
```

**备注：**Hive语句和mysql都可以通过explain查看执行计划，使用explain + Hive语句



# 三、参考书籍

* [易百教程](<https://www.yiibai.com/Hive/>)

# 四、疑难解答

## 1、Hive 在指定位置添加字段

* 首先添加字段

  ```mysql
  alter table table_name add columns (c_time string comment '当前时间');
  ```

* 其次更改字段的顺序

  ```mysql
  alter table table_name change c_time c_time string after address;
  ```

## 2、Hive创建表时指定文件格式

* **TEXTFIEL**

  默认格式，数据不做压缩，磁盘开销大，数据解析开销大。
  可结合Gzip、Bzip2使用（系统自动检查，执行查询时自动解压），但使用这种方式，Hive不会对数据进行切分，从而无法对数据进行并行操作。

  ```mysql
  -- 固定格式
  create table test1(str STRING)  
  STORED AS TEXTFILE;
  
  -- 自定义格式
  create table test1(str STRING)  
  STORED AS
  INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat'
  OUTPUTFORMAT 'org.apache.hadoop.Hive.ql.io.HiveIgnoreKeyTextOutputFormat'
  ```

* **SEQUENCEFILE**

  SequenceFile是Hadoop API提供的一种二进制文件支持，其具有使用方便、可分割、可压缩的特点。
  SequenceFile支持三种压缩选择：NONE, RECORD, BLOCK。 Record压缩率低，一般建议使用BLOCK压缩。

  ```mysql
  -- 固定格式
  create table test1(str STRING)  
  STORED AS SEQUENCEFILE;
  
  -- 自定义格式
  create table test1(str STRING)  
  STORED AS
  INPUTFORMAT 'org.apache.hadoop.mapred.SequenceFileInputFormat'
  OUTPUTFORMAT 'org.apache.hadoop.Hive.ql.io.HiveSequenceFileOutputFormat'
  ```

* **RCFILE**

  RCFILE是一种行列存储相结合的存储方式。首先，其将数据按行分块，保证同一个record在一个块上，避免读一个记录需要读取多个block。其次，块数据列式存储，有利于数据压缩和快速的列存取。

  ```mysql
  -- 固定格式
  create table test1(str STRING)  
  STORED AS RCFILE;
  
  -- 自定义格式
  create table test1(str STRING)  
  STORED AS
  INPUTFORMAT 'org.apache.hadoop.Hive.ql.io.RCFileInputFormat'
  OUTPUTFORMAT 'org.apache.hadoop.Hive.ql.io.RCFileOutputFormat'
  ```

## 3、Hive同时拆分多列为多行

* [问题链接](<https://stackoverflow.com/questions/37585638/Hive-split-delimited-columns-over-multiple-rows-select-based-on-position?rq=1>)

**问题：**

I'm Looking for a way to split the column based on comma delimited data. Below is my dataset

```mysql
id  col1  col2
1   5,6   7,8
```

I want to get the result

```mysql
id col1 col2
1  5    7
1  6    8
```

**答案：**

You can use `posexplode()` to create position index columns for your split arrays. Then, select only those rows where the position indices are equal.

```
SELECT id, col3, col4
  FROM test
  lateral VIEW posexplode(split(col1,'\002')) col1 AS pos3, col3
  lateral VIEW posexplode(split(col2,'\002')) col2 AS pos4, col4
  WHERE pos3 = pos4;
```

Output:

```
id col3 col4
1  5    7
1  6    8
```

## 4、 Hive或spark中执行sql字符常量包含`;`时会报错

比如

> select instr('abc;abc', ';');

报错

> NoViableAltException(-1@[147:1: selectExpression : ( expression | tableAllColumns );])

**修改：**需要将`;`改为`ascii`

> select instr('abc\073abc', '\073');

## 5、如何在 Apache Hive 中解析 Json 数组

### 问题1：从**json字符串**中解析一个字段-get_json_object

```hive
hive>  SELECT get_json_object('{"website":"www.iteblog.com","name":"过往记忆"}', '$.website');
OK
www.iteblog.com
```

### 问题2：从**json字符串**中解析多个字段-json_tuple

```hive
hive> SELECT json_tuple('{"website":"www.iteblog.com","name":"过往记忆"}', 'website', 'name');
OK
www.iteblog.com 过往记忆
```

### 问题3：从**json数组**中解析某一个字段-get_json_object

```hive
hive> SELECT get_json_object('[{"website":"www.iteblog.com","name":"过往记忆"}, {"website":"carbondata.iteblog.com","name":"carbondata 中文文档"}]', '$[0].website');
OK
www.iteblog.com
```

**注意：**这里与参考链接[如何在 Apache Hive 中解析 Json 数组](https://www.iteblog.com/archives/2362.html)中不同的是，我使用的是`$[0].website`，而参考链接使用的`$.[0].website`，我按照**参考链接给的方法select不出答案**

### 问题4：从**json数组**中解析多个字段-先explode再get_json_object或json_tuple

* explode将**json数组**用一行拆分成多行
* 然后再对其进行**json字符串**解析

详情请参考[如何在 Apache Hive 中解析 Json 数组](https://www.iteblog.com/archives/2362.html)

