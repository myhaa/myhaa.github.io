---
title: 实用工具之FFmpeg
author: Myhaa
top: false
cover: false
toc: true
mathjax: false
categories: 实用工具
tags:
  - FFmpeg
date: 2021-09-06 18:16:11
img:
coverImg:
password:
summary: FFmpeg视频处理入门教程
---

# FFmpeg视频处理工具

## 入门

### 安装

```shell
conda install ffmpeg
```

### 概念

#### 容器

视频文件本身是一个`容器`，里面包含`视频`、`音频`、`字幕`等等

常见的容器格式有以下几种（一般后缀能反映其格式）

1. mp4
2. mkv
3. webm
4. avi

##### 查看`ffmpeg`支持容器格式

```shell
ffmpeg -formats
```

#### 编码格式

视频、音频都需要编码才能保存成文件。不同的编码格式，有不同的压缩率，会导致文件大小和清晰度的差异

常见的视频编码格式有以下几种

* 有版权，免费使用
  * H.262
  * H.264
  * H.265
* 无版权
  * VP8
  * VP9
  * AV1

常见的音频编码格式

* 无损
  * FLAC
  * ALAC
* 有损
  * MP3
  * AAC

##### 查看ffmpeg支持编码格式

```shell
ffmpeg -codecs
```

#### 编码器

编码器是实现某种编码格式的库文件，只有安装了对应编码格式的编码器，才能实现该格式视频、音频编码和解码

常见的视频编码器

1. libx264：最流行的开源H.264编码器
2. NVENC：基于NVIDIA GPU的H.264编码器
3. libx265：开源的HEVC编码器
4. libvpx：谷歌的VP8和VP9编码器
5. libaom：AV1编码器

常见音频编码器

1. libfdk-aac：当前最高质量的AAC编码，编码模式分为CBR和VBR
2. aac

##### 查看ffmpeg已安装编码器

```shell
ffmpeg -encoders
```

### 使用格式

#### 命令

```shell
ffmpeg {1} {2} -i {3} {4} {5}
```

其中：

1. 全局参数
2. 输入文件参数
3. 输入文件
4. 输出文件参数
5. 输出文件

#### 例子

```shell
ffmpeg \
-y \  # 全局参数
-c \  # 输入文件参数
-i input.mp4 \  # 输入文件
-c:v libvpx-vp9 -c:a libvorbis \  # 输出文件参数
output.webm  # 输出文件
```

上面的命令是将mp4文件转成webm文件，两个都是容器格式。输入的mp4文件的音频编码格式是aac，视频编码格式是H.264；输出的webm文件的视频编码格式是vp9，音频格式是vorbis

* 如果不指明编码格式，FFmpeg会自己判断输入文件编码

```shell
ffmpeg -i input.avi output.mp4
```

### 常用命令行参数

* `-c`：指定编码器
* `-c copy`：直接复制，不经过重新编码（这样比较快）
* `-c:v`：指定视频编码器
* `-c:a`：指定音频编码器
* `-i`：指定输入文件
* `-an`：去除音频流
* `-vn`：去除视频流
* `-preset`：指定输出视频的视频质量，会影响文件生成速度，以下几个可用值：
  * ultrafast
  * superfast
  * veryfast
  * faster
  * fast
  * medium
  * slow
  * slower
  * veryslow
* `-y`：不经过确认，直接覆盖同名文件

### 常见用法

#### 查看文件信息

```shell
ffmpeg -i input.mp4

ffmpeg -i input.mp4 -hide_banner
```

#### 转换编码格式

```shell
ffmpeg -i input.mp4 -c:v libx265 outpu.mp4
```

#### 转换容器格式

```shell
ffmpeg -i input.mp4 -c copy output.webm
```

#### 调整码率

调整码率指的是改变编码的比特率，一般用来将视频文件的体积变小

```shell
ffmpeg \
-i input.mp4 \
-minrate 964K -maxrate 3856K -bufsize 2000K \
output.mp4
```

#### 改变分辨率

```shell
ffmpeg \
-i input.mp4 \
-vf scale=480:-1 \
output.mp4
```

#### 提取音频

```shell
ffmpeg \
-i input.mp4\
-vn -c:a copy \
output.aac
```

#### 添加音轨

```shell
ffmpeg \
-i input.aac -i input.mp4 \
output.mp4
```

#### 截图

* 指定时间开始，连续对1秒钟的视频进行截图

```shell
ffmpeg \
-y \
-i input.mp4 \
-ss 00:01:24 -t 00:00:01 \
output_%3d.jpg
```

* 只需要截一张图

```shell
ffmpeg \
-ss 01:23:45 \
-i input.mp4 \
-vframes 1 -q:v 2 \  # 只截取一帧，输出图片质量为2，1最高
output.jpg
```

#### 裁剪

```shell
ffmpeg -ss [start] -i [input] -t [duration] -c copy [output]

ffmpeg -ss [start] -i [input] -to [end] -c copy [output]
```

#### 为音频添加封面

```shell
ffmpeg \
-loop 1 \  # 表示图片无限循环
-i cover.jpg -i input.mp3 \
-c:v libx264 -c:a aac -b:a 192k -shortest \ # 表示音频文件结束，输出视频就结束
output.mp4
```





## 进阶

## 疑难解答

## 参考

1. [阮一峰的网络日志](https://www.ruanyifeng.com/blog/2020/01/ffmpeg.html)

