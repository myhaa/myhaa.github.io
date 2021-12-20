---
title: 实用工具之MapReduce
author: Myhaa
top: false
cover: false
toc: true
mathjax: false
categories: 实用工具
tags:
  - MapReduce
date: 2020-06-03 09:25:31
img:
coverImg:
password:
summary: MapReduce的原理介绍
---



# Title

![image-20200603093104618](%E5%AE%9E%E7%94%A8%E5%B7%A5%E5%85%B7%E4%B9%8BMapReduce/image-20200603093104618.png)

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

# MR的Java实现

## friendRecommendDriver.java

```java
package friendrecommend;

import utils.HDFSUtils;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Author: 
 * Email: 
 * Date: 2020/9/14
 * Desc: 好友推荐，计算两两用户的好友推荐得分，得分越高越值得推荐
 */
public class FriendRecommendDriver extends Configured implements Tool {
    //一些变量
    private static final Logger LOG = LoggerFactory.getLogger(FriendRecommendDriver.class);  //Return a logger named corresponding to the class passed as parameter, using the statically bound ILoggerFactory instance.

    //一些信息
    private static final String JOB_NAME_PREFIX = "friend_recommend1";  //项目自定义名

    //一些路径
    private static String INPUT_PATH = "";  //输入文件路径
    private static String OUTPUT_PATH = "";  // 输出路径

    // mr运行函数
    @Override
    public int run(String[] strings) throws Exception {
        //删除输出路径
        HDFSUtils.deleteFile(OUTPUT_PATH);

        //job配置
        Job FriendRecommendJob = Job.getInstance();  //Creates a new Job with no particular Cluster . A Cluster will be created with a generic Configuration.
        FriendRecommendJob.setJarByClass(FriendRecommendDriver.class);  //Set the Jar by finding where a given class came from.
        FriendRecommendJob.setJobName(JOB_NAME_PREFIX);  //Set the user-specified job name.

        FileInputFormat.addInputPath(FriendRecommendJob, new Path(INPUT_PATH));  //Add a Path to the list of inputs for the map-reduce job.
        FileOutputFormat.setOutputPath(FriendRecommendJob, new Path(OUTPUT_PATH));  //Set the Path of the output directory for the map-reduce job.
        FriendRecommendJob.setMapperClass(FriendRecommendMapper.class);  //Set the Mapper for the job.
        FriendRecommendJob.setReducerClass(FriendRecommendReducer.class);  //Set the Reducer for the job.

        FriendRecommendJob.setNumReduceTasks(300);  //todo 1 Set the number of reduce tasks for the job.
        //FriendRecommendJob.setInputFormatClass(SequenceFileInputFormat.class);  //Set the InputFormat for the job.
        FriendRecommendJob.setMapOutputKeyClass(Text.class);  //Set the key class for the map output data. This allows the user to specify the map output key class to be different than the final output value class.
        FriendRecommendJob.setMapOutputValueClass(Text.class);  //Set the value class for the map output data. This allows the user to specify the map output value class to be different than the final output value class.
        FriendRecommendJob.setOutputKeyClass(Text.class);  //Set the key class for the job output data.
        FriendRecommendJob.setOutputValueClass(Text.class);  //Set the value class for job outputs.
        FriendRecommendJob.getConfiguration().set("INPUT_PATH", INPUT_PATH);
        //Submit the job to the cluster and wait for it to finish.
        if (!FriendRecommendJob.waitForCompletion(true)) {
            return 1;
        }

        return 0;
    }

    //main函数
    public static void main(String[] args) throws Exception {
        //启动
        long start = System.currentTimeMillis();

        //判断输入参数是否合规
        if (args.length != 2) {
            System.err.println("Usage: friend_recommend <input path> <output path>");
            System.exit(-1);
        }

        //读取输入参数
        INPUT_PATH = args[0];
        OUTPUT_PATH = args[1];

        LOG.info("Begin to calc {}", JOB_NAME_PREFIX);
        LOG.info("inputFile:{}, outputPath:{}", INPUT_PATH, OUTPUT_PATH);

        //开始跑MR
        ToolRunner.run(new FriendRecommendDriver(), args);

        //完成并打印时间
        long finish = System.currentTimeMillis();
        System.out.println("Time in ms: " + (finish - start));
    }
}

```

## friendRecommendMapper.java

```java
package friendrecommend;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Author: 
 * Email: 
 * Date: 2020/9/14
 * Desc:
 */
public class FriendRecommendMapper extends Mapper<LongWritable, Text, Text, Text> {
    private final Logger LOG = LoggerFactory.getLogger(FriendRecommendMapper.class);

    //分隔符
    private static final String TAB_SEP = "\t";

    @Override
    protected void setup(Context context) {
        //在mapper启动时执行一次
        LOG.info("setup mapper...");
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        // 每个mapper的主逻辑

        //读取数据
        String[] contents = value.toString().trim().split(TAB_SEP, 2);

        //解析每行
        String udid = contents[0];
        String other_format = contents[1];

        context.write(new Text(udid), new Text(other_format));

    }

    @Override
    protected void cleanup(Context context) {
        //在mapper结束时执行一次
    }

}

```

## friendRecommendReducer.java

```java
package friendrecommend;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;

import java.io.IOException;
import java.util.*;

/**
 * Author: 
 * Email: 
 * Date: 2020/9/14
 * Desc: reducer每类只输出分数靠前的100个
 */
public class FriendRecommendReducer extends Reducer<Text, Text, Text, Text> {
    private final Logger LOG = LoggerFactory.getLogger(FriendRecommendReducer.class);

    //分隔符
    private static final String TAB_SEP = "\t";
    private static final String COMMA_SEP = ",";
    private static final int output_limit = 100;

    private static Map<String, String> udid_format_Map = new HashMap<>();

    @Override
    protected void setup(Context context) {
        //在reducer启动时执行一次
        LOG.info("setup reducer...");
        //一些变量
        String udid_format_file_path = context.getConfiguration().get("INPUT_PATH");
        udid_format_Map = FriendRecommendUtils.load_udid_format(udid_format_file_path);
    }

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        //reduce主逻辑

        //一些变量

        //处理
        for (Text value : values) {
            //解析
            String[] contents1 = value.toString().replace("~", "\"").split(TAB_SEP);
            String udid1 = key.toString();
            //判断是否是-2
            if (contents1[1].equals("-2")){
                contents1[1] = "{}";
            }
            if (contents1[3].equals("-2")){
                contents1[3] = "{}";
            }
            //format解析
            //String role_id1 = contents1[0];
            JSONObject social_format1 = JSON.parseObject(contents1[1]);
            String label_format1 = contents1[2];
            JSONObject game_format1 = JSON.parseObject(contents1[3]);
            String friend_format1 = contents1[4];
            //社交属性解析
            //Double board1 = social_format1.getDoubleValue("board");  //发博点赞评论次数
            double chat1 = social_format1.getDoubleValue("chat");  //社交人数
            double follow1 = social_format1.getDoubleValue("follow");  //主动添加好友次数
            double follow_room1 = social_format1.getDoubleValue("followRoom");  //跟房人数
            //double enter1 = social_format1.getDoubleValue("enter");  //进入房间途径——指定follow_room
            double gift1 = social_format1.getDoubleValue("gift");  //主动送礼人数
            String sex1 = social_format1.getString("sex");  //性别
            //游戏行为解析
            String cash1 = game_format1.getString("cash");  // 累计充值额
            String level1 = game_format1.getString("level");  //等级
            String hour_label1 = game_format1.getString("hour_label");  //在线时点出现次数
            String age1 = game_format1.getString("age");  //年龄
            String sect1 = game_format1.getString("sect");  //门派
            //String master1 = game_format1.getString("master");  //大师场比例
            String room_label1 = game_format1.getString("room_label");  //room_mode参与次数
            String log_lat1 = game_format1.getString("log_lat");  //经纬度

            //一些字段初始化
            Map<String, Double> score_Map = new HashMap<>();  //得分map
            int count1 = 0;
            int count2 = 0;
            int count3 = 0;
            int count4 = 0;
            Map<String, String> category1 = new HashMap<>();  //各类别计数
            Map<String, String> category2 = new HashMap<>();
            Map<String, String> category3 = new HashMap<>();
            Map<String, String> category4 = new HashMap<>();

            //社交活跃度得分
            //日均私聊人数
            double score_chat = FriendRecommendUtils.calc_score_chat(chat1);
            score_Map.put("chat", score_chat);
            //日均开局人数
            double score_begin = FriendRecommendUtils.calc_score_begin(follow_room1);
            score_Map.put("begin", score_begin);
            //日均送礼人数
            double score_gift = FriendRecommendUtils.calc_score_gift(gift1);
            score_Map.put("gift", score_gift);

            //开始循环
            for (Map.Entry<String, String> entry : udid_format_Map.entrySet()) {
                //同一个人不计算
                String udid2 = entry.getKey();
                if (udid1.equals(udid2)){
                    continue;
                }

                //解析
                String[] contents2 = entry.getValue().replace("~", "\"").split(TAB_SEP);
                //判断是否是-2
                if (contents2[1].equals("-2")){
                    contents2[1] = "{}";
                }
                if (contents2[3].equals("-2")){
                    contents2[3] = "{}";
                }
                //format解析
                String role_id2 = contents2[0];
                JSONObject social_format2 = JSON.parseObject(contents2[1]);
                String label_format2 = contents2[2];
                JSONObject game_format2 = JSON.parseObject(contents2[3]);
                String friend_format2 = contents2[4];
                //社交属性解析
                //Double board2 = social_format2.getDoubleValue("board");  //发博点赞评论次数
                //double chat2 = social_format2.getDoubleValue("chat");  //社交人数
                double follow2 = social_format2.getDoubleValue("follow");  //主动添加好友次数
                //Double follow_room2 = social_format2.getDoubleValue("followRoom");  //跟房人数
                //double enter2 = social_format2.getDoubleValue("enter");  //进入房间途径——指定follow_room
                double gift2 = social_format2.getDoubleValue("gift");  //主动送礼人数
                String sex2 = social_format2.getString("sex");  //性别
                //游戏行为解析
                String cash2 = game_format2.getString("cash");  // 累计充值额
                String level2 = game_format2.getString("level");  //等级
                String hour_label2 = game_format2.getString("hour_label");  //在线时点出现次数
                String age2 = game_format2.getString("age");  //年龄
                String sect2 = game_format2.getString("sect");  //门派
                //String master2 = game_format2.getString("master");  //大师场比例
                String room_label2 = game_format2.getString("room_label");  //room_mode参与次数
                String log_lat2 = game_format2.getString("log_lat");  //经纬度

                //判断他俩是不是好友，是好友则不再计算
                String flag = "0";
                if (!role_id2.equals("-2") && !friend_format1.equals("[]")){
                    for (String s:role_id2.split(COMMA_SEP)){
                        if (friend_format1.contains(s)){
                            flag = "1";
                            break;
                        }
                    }
                }
                if (flag.equals("1")){
                    //System.out.println("friends");
                    continue;
                }

                //互补性得分
                //主动添加好友数
                double score_follow = FriendRecommendUtils.calc_score_follow_friend(follow1, follow2);
                score_Map.put("follow", score_follow);
                //主动送礼次数
                double score_give_gift = FriendRecommendUtils.calc_score_give_gift(gift1, gift2);
                score_Map.put("gift2", score_give_gift);
                //性别是否相同
                double score_sex = FriendRecommendUtils.calc_score_sex(sex1, sex2);
                score_Map.put("sex", score_sex);

                //匹配度得分
                //兴趣爱好
                //兴趣标签交集
                double score_label = FriendRecommendUtils.calc_score_label(label_format1, label_format2);
                score_Map.put("label", score_label);
                //score_Map.put("label_weibo", score_label_weibo);
                //游戏行为
                //等级
                double score_level = FriendRecommendUtils.calc_score_level(level1, level2);
                score_Map.put("level", score_level);
                //在线时点重合度
                double score_hour = FriendRecommendUtils.calc_score_hour(hour_label1, hour_label2);
                score_Map.put("hour", score_hour);
                //门派
                double score_sect = FriendRecommendUtils.calc_score_sect(sect1, sect2);
                score_Map.put("sect", score_sect);
                //历史累计充值额
                double score_cash = FriendRecommendUtils.calc_score_cash(cash1, cash2);
                score_Map.put("cash", score_cash);
                //大师场占比
                //double score_master = FriendRecommendUtils.calc_score_master(master1, master2);
                //score_Map.put("master", score_master);
                //参与room_mode重合度
                double score_room = FriendRecommendUtils.calc_score_room(room_label1, room_label2);
                score_Map.put("room", score_room);
                //年龄
                double score_age = FriendRecommendUtils.calc_score_age(age1, age2);
                score_Map.put("age", score_age);
                //lbs
                double score_lbs = FriendRecommendUtils.calc_score_lbs(log_lat1, log_lat2);
                score_Map.put("lbs", score_lbs);

                //好友链条距离
                double score_friend = FriendRecommendUtils.calc_score_friend_distance(friend_format1, friend_format2);
                score_Map.put("friend", score_friend);

                //打上标签
                String label = FriendRecommendUtils.get_label(score_chat, score_begin, score_gift, score_follow,
                        score_give_gift, score_sex, score_level, score_sect, score_cash, score_label,
                        score_age, score_lbs, score_hour, score_room);

                //得分求和
                double score_sum = FriendRecommendUtils.get_score_sum(score_Map);

                //一些字段
                String key_ = udid1 + TAB_SEP + udid2;
                String value_ = score_sum + TAB_SEP + label + TAB_SEP + score_Map;

                //判断是属于哪类
                switch (label.split(COMMA_SEP, 2)[0]) {
                    case "1":
                        count1 ++;
                        FriendRecommendUtils.get_category(count1, output_limit, category1, key_, value_, score_sum);
                        break;
                    case "2":
                        count2 ++;
                        FriendRecommendUtils.get_category(count2, output_limit, category2, key_, value_, score_sum);
                        break;
                    case "3":
                        count3 ++;
                        FriendRecommendUtils.get_category(count3, output_limit, category3, key_, value_, score_sum);
                        break;
                    default:
                        count4 ++;
                        FriendRecommendUtils.get_category(count4, output_limit, category4, key_, value_, score_sum);
                        break;
                }
            }

            //输出
            for (Map.Entry<String, String> tmp:category1.entrySet()){
                context.write(new Text(tmp.getKey()), new Text(tmp.getValue()));
            }
            for (Map.Entry<String, String> tmp:category2.entrySet()){
                context.write(new Text(tmp.getKey()), new Text(tmp.getValue()));
            }
            for (Map.Entry<String, String> tmp:category3.entrySet()){
                context.write(new Text(tmp.getKey()), new Text(tmp.getValue()));
            }
            for (Map.Entry<String, String> tmp:category4.entrySet()){
                context.write(new Text(tmp.getKey()), new Text(tmp.getValue()));
            }
            //context.write(key, value);
        }
    }

    @Override
    protected void cleanup(Context context) {
    }
}

```

## friendRecommendUtils.java

```java
package friendrecommend;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import utils.HDFSUtils;
import utils.LBSUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Author: 
 * Email: 
 * Date: 2020/9/16
 * Desc: 支持函数
 */

public class FriendRecommendUtils {
    private static final Logger LOG = LoggerFactory.getLogger(FriendRecommendUtils.class);

    private static final Map<String, String> udid_format_Map = new HashMap<>();
    private static final String TAB_SEP = "\t";
    private static final String COMMA_SEP = ",";

    public static Map<String, String> load_udid_format(String filePath) {
        LOG.info("begin to load udid format");
        Configuration conf = new Configuration();
        try {
            FileSystem fs = FileSystem.get(URI.create(filePath), conf);
            List<String> list = HDFSUtils.getAllHdfsFile(filePath);
            for (String dsf : list) {
                LOG.debug(dsf);
                FSDataInputStream hdfsInStream = fs.open(new Path(dsf));

                String line;
                BufferedReader in = new BufferedReader(new InputStreamReader(hdfsInStream, StandardCharsets.UTF_8));
                while ((line = in.readLine()) != null) {
                    String[] contents = line.trim().split(TAB_SEP, 2);

                    //解析每行
                    String udid = contents[0];
                    String other_format = contents[1];

                    //输出
                    udid_format_Map.put(udid, other_format);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return udid_format_Map;
    }

    public static String[] intersect(String[] arr1, String[] arr2) {
        Map<String, Boolean> map = new HashMap<>();
        List<String> list = new ArrayList<>();
        for (String str : arr1) {
            if (!map.containsKey(str)) {
                map.put(str, Boolean.FALSE);
            }
        }
        for (String str : arr2) {
            if (map.containsKey(str)) {
                map.put(str, Boolean.TRUE);
            }
        }

        for (Map.Entry<String, Boolean> e : map.entrySet()) {
            if (e.getValue().equals(Boolean.TRUE) && !e.getKey().equals("-2")) {
                list.add(e.getKey());
            }
        }

        String[] result = {};
        return list.toArray(result);
    }

    public static void main(String[] args) {
    }

}

```

## HDFSUtils.java

```java
package utils;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * AUTHOR: LUJINHONG
 * CREATED ON: 17/1/9 17:24
 * PROJECT NAME: etl_hadoop_aplus
 * DESCRIPTION:
 */
public class HDFSUtils {
    private static Logger LOG = LoggerFactory.getLogger(HDFSUtils.class);

    public static List<String> getFileNameList(String dir) throws IOException {
        List<String> list = new ArrayList<>();
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        FileStatus[] stats = fs.listStatus(new Path(dir));
        for (int i = 0; i < stats.length; ++i) {
            if (!stats[i].isDirectory()) {
                // regular file
                list.add(stats[i].getPath().toString());
                //LOG.debug("Load file : {}", stats[i].getPath().toString());
            } else {
                // dir
                LOG.info("Ignore directory : {}", stats[i].getPath().toString());
            }
        }
        fs.close();
        return list;
    }

    /**
     * 递归删除文件或目录，或者其集合
     * @param fileNameSet
     */
    public static void  deleteFile(Set<String> fileNameSet){
        //多次构建FileSystem对象，如果调用很频繁的话考虑用单例。
        for(String fileName : fileNameSet){
            deleteFile(fileName);
        }
    }
    public static void  deleteFile(String fileName){
        try(FileSystem fs = FileSystem.get(URI.create(fileName),new Configuration())){
            if(fs.exists(new Path(fileName))){
                fs.delete(new Path(fileName),true);
                LOG.info("{} deleted.",fileName);
            }
        }catch (IOException e){
            LOG.info("Error happens when deleting file: {}.", fileName);
            e.printStackTrace();
        }

    }

    //递归列出目录中的所有文件。
    public static List<String> getAllHdfsFile(String dir) throws IOException {
        List<String> fileList = new ArrayList<>();
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(dir), conf);

        RemoteIterator<LocatedFileStatus> iterator = fs.listFiles(
                new Path(dir), true);

        while (iterator.hasNext()) {
            LocatedFileStatus fileStatus = iterator.next();
            fileList.add(fileStatus.getPath().toString());
        }
        return fileList;

    }


    /**
     * 文件检测并删除
     *
     * @param path
     * @param conf
     * @return
     */
    public static boolean checkAndDel(final String path, Configuration conf) {
        Path dstPath = new Path(path);
        try {
            FileSystem fs = dstPath.getFileSystem(conf);
            if (fs.exists(dstPath)) {
                fs.delete(dstPath, true);
            } else {
                return false;
            }
        } catch (IOException ie) {
            ie.printStackTrace();
            return false;
        }
        return true;
    }

    public boolean checkHdfsFiledir(final String path, Configuration conf) {
        Path dstPath = new Path(path);
        try {
            FileSystem dhfs = dstPath.getFileSystem(conf);
            if (dhfs.exists(dstPath)) {
                return true;
            } else {
                return false;
            }
        } catch (IOException ie) {
            ie.printStackTrace();
            return false;
        }
    }

}

```

