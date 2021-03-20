---
title: 深度学习之TF安装
author: Myhaa
top: false
cover: false
toc: true
mathjax: false
categories: AI/数据科学
tags:
  - 深度学习
  - 安装教程
date: 2021-03-20 11:56:00
img:
coverImg:
password:
summary: 深度学习之TF安装
---

# 参考

* [windows tensorflow-gpu的安装](https://zhuanlan.zhihu.com/p/35717544)
* [官网](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)

# windows安装

## 先更新显卡驱动

* 右键更新显卡驱动

  ![image-20210320115932959](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTF%E5%AE%89%E8%A3%85/image-20210320115932959.png)

## 查看显卡驱动对应`cuda`版本

* 桌面右键显卡控制面板

  ![image-20210320120246190](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTF%E5%AE%89%E8%A3%85/image-20210320120246190.png)

## 下载安装`cuda`

* [下载地址](https://developer.nvidia.com/cuda-toolkit-archive)：非常之慢

* [下载地址2](https://developer.nvidia.com/zh-cn/cuda-downloads)

* [安装参考地址](https://blog.csdn.net/XunCiy/article/details/89070315)

  ![image-20210320153609006](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTF%E5%AE%89%E8%A3%85/image-20210320153609006.png)

* 以下两个地方注意比选就行

  ![image-20210320122925046](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTF%E5%AE%89%E8%A3%85/image-20210320122925046.png)

  ![image-20210320122936325](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTF%E5%AE%89%E8%A3%85/image-20210320122936325.png)



## 查看`cuda`对应的`cudnn`版本

* [地址](https://www.tensorflow.org/install/source_windows#gpu)

  ![image-20210320121040837](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTF%E5%AE%89%E8%A3%85/image-20210320121040837.png)

## 下载安装`cudnn`

* [下载地址](https://developer.nvidia.com/zh-cn/cudnn)：需要登录

* [下载地址2](https://developer.nvidia.cn/rdp/cudnn-download)

  ![image-20210320162205564](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTF%E5%AE%89%E8%A3%85/image-20210320162205564.png)

* 安装

  ![image-20210320163021436](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTF%E5%AE%89%E8%A3%85/image-20210320163021436.png)

## 安装`tensorflow-gpu`

* 根据`cuda`对应版本安装`tf`的对应版本，使用`pycharm`安装

  ![image-20210320163946401](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8BTF%E5%AE%89%E8%A3%85/image-20210320163946401.png)

## 测试是否安装成功

```python
import tensorflow as tf
print(tf.__version__)
print('GPU', tf.test.is_gpu_available())

# output
# GPU [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

* 出现以下报错时：

  ```shell
  2021-03-20 16:51:38.885475: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found
  ```

* 将`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\bin`目录下的`cusolver64_11.dll`复制，并将副本改名为`cusolver64_10.dll`

## 注意

1. 显卡有对应的显卡驱动
2. 显卡驱动对应`cuda`的版本
3. 根据`cuda`的版本对应`cudnn`、`tf`版本