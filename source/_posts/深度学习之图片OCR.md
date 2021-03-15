---
title: 深度学习之图片OCR
author: Myhaa
top: false
cover: false
toc: true
mathjax: false
categories: AI/数据科学
tags:
  - AI
  - 图像识别
date: 2021-03-11 15:45:18
img:
coverImg:
password:
summary: 图片相关处理技术
---

# 基础

## 视频截帧成一张张图片

### 使用`opencv`

* [参考opencv实现视频切割](https://github.com/drsanwujiang/video-subtitle-recognize)

```python
# -*- coding:utf-8 _*-
"""
Author: meiyunhe
Email: 
Date: 2021/03/12
File: opencv_test.py
Software: PyCharm
Description: 使用opencv将视频切成若干张静态图片（默认每秒1张）
"""


# load modules
import shutil
import time

import cv2
import os
import sys


# config类
class Config:
	Maps = {
		# 以下请根据需要调整数值
		"split_duration": 1.5,  # 切片间隔,每 split_duration 秒输出一帧
		"jpg_quality": 40,  # 图片输出质量, 0~100
		"probability": 0.66,  # OCR可信度下限, 0~1
		"subtitle_top_rate": 0.66,  # 字幕范围倍率
		"remove_duplicate": False,  # 强制去重
		
		# 目录信息,在下方定义
		"video_dir": "",
		"video_path": "",
		"video_frames": "",
		"image_dir": "",
		"output_dir": "",
		
		# 视频信息,自动生成
		"video_name": "",
		"video_suffix": "",
		"video_width": 0,
		"video_height": 0,
		"subtitle_top": 0,  # 字幕范围 = 字幕范围倍率 * 视频高度,此高度以下的文字被认为是字幕
	}
	
	@staticmethod
	def set_path(video_name="", video_suffix=""):
		current_path = sys.path[0]
		
		Config.Maps["video_dir"] = '%s/video/' % current_path  # 视频源文件目录
		Config.Maps["video_path"] = '%s/video/%s%s' % (current_path, video_name, video_suffix)  # 指定视频文件路径
		Config.Maps["video_frames"] = '%s/video_frames/' % current_path  # 视频切片文件目录
		Config.Maps["image_dir"] = '%s/video_frames/%s/' % (current_path, video_name)  # 指定视频切片文件目录
		Config.Maps["output_dir"] = '%s/output/' % current_path  # 字幕输出目录
		
		Config.Maps["video_name"] = video_name
		Config.Maps["video_suffix"] = video_suffix
	
	@staticmethod
	def set_video_props(video_width, video_height):
		Config.Maps["video_width"] = video_width
		Config.Maps["video_height"] = video_height
		Config.Maps["subtitle_top"] = Config.Maps["subtitle_top_rate"] * video_height
	
	@staticmethod
	def get_value(key):
		return Config.Maps[key]


# getFrame类
class GetFrames:
	@staticmethod
	def main():
		# 读取路径信息
		video_path = Config.get_value("video_path")
		image_dir = Config.get_value('image_dir')
		jpg_quality = Config.get_value('jpg_quality')
		split_duration = Config.get_value('split_duration')
		
		if (os.path.exists(image_dir)):
			shutil.rmtree(image_dir)  # 递归删除目录，os.rmdir只能删除空目录
		os.mkdir(image_dir)
		
		cv = cv2.VideoCapture(video_path)  # 读入视频文件
		current_frame = 1
		saved_frames = 1
		
		if cv.isOpened():  # 判断是否正常打开
			retval, frame = cv.read()
		else:
			cv.release()
			print("Video open error")
			return False
		
		duration = int(cv.get(cv2.CAP_PROP_FPS) * split_duration)  # 间隔频率 = 帧率 * 切片时间间隔(四舍五入)
		frame_count = int(cv.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
		video_width = int(cv.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
		video_height = int(cv.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
		Config.set_video_props(video_width, video_height)
		
		while retval:  # 循环读取视频帧
			retval, frame = cv.read()
			
			if current_frame % duration == 0:  # 每 duration 帧进行存储操作
				cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])[1]. \
					tofile(image_dir + str(current_frame).zfill(6) + '.jpg')
				
				print(("Now: frame %d, saved: %d frame(s), process: %d%%" %
					   (current_frame, saved_frames, (current_frame * 100) // frame_count)).ljust(60, ' '))
				saved_frames += 1
			
			current_frame += 1
			cv2.waitKey(1)
		
		print(("Now: frame %d, saved: %d frame(s), process: %d%%" %
			   (current_frame, saved_frames, (current_frame * 100) // frame_count)).ljust(60, ' '))
		cv.release()
		print("\nSaved: %d frame(s)" % saved_frames)
		return True
	

# 主类
class Main:
	@staticmethod
	def clear():
		if sys.platform.find("win") > -1:
			os.system("cls")
		else:
			print()
	
	@staticmethod
	def main():
		# 配置路径
		Config.set_path()
		
		if not (os.path.exists(Config.get_value('video_dir'))):
			os.mkdir(Config.get_value('video_dir'))
		
		if not (os.path.exists(Config.get_value('video_frames'))):
			os.mkdir(Config.get_value('video_frames'))
		
		if not (os.path.exists(Config.get_value('output_dir'))):
			os.mkdir(Config.get_value('output_dir'))
		
		# 列出所有video
		Main.clear()
		print("\n")
		print("-"*40)
		print("List Video")
		print("-" * 40)
		video_list = os.listdir(Config.get_value('video_dir'))
		print(video_list)
		
		if len(video_list) < 1:
			print("Nothing found\n\n")
			print("Process finished")
			input()
			return
		
		# 对所有video进行处理
		start_all = time.time()
		print("\n")
		print("-" * 40)
		print("All Video Division")
		print("-" * 40)
		print("Start All video division")
		for video in video_list:
			print("%d.%s" % (video_list.index(video) + 1, video))
			video_name = video[: video.rfind(".")]
			video_suffix = video[video.rfind("."):]
		
			Config.set_path(video_name, video_suffix)
			
			start = time.time()
			print("\n")
			print("-" * 40)
			print("Video: %s Division" % video)
			print("-" * 40)
			print("Start video division")
			
			if not GetFrames.main():
				print("Video division FAILED!")
				print("Process finished")
				input()
				return
			
			print("Video: %s division finished" % video)
			print("Time: %.2fs\n" % (time.time() - start))
		
		print("All Video division finished")
		print("Time: %.2fs\n" % (time.time() - start_all))
		return
	
	
if __name__ == "__main__":
	Main.main()

```



# 进阶

# 疑难

# 参考

1. [opencv实现视频切割](https://github.com/drsanwujiang/video-subtitle-recognize)
   1. [OpenCV-Python中文教程](https://www.kancloud.cn/aollo/aolloopencv)
2. [FFmpeg视频处理](https://www.ruanyifeng.com/blog/2020/01/ffmpeg.html)
3. [darknet-ocr](https://link.zhihu.com/?target=https%3A//github.com/chineseocr/darknet-ocr)：`star:851`
4. [chineseocr](https://link.zhihu.com/?target=https%3A//github.com/chineseocr/chineseocr)：`star:3.9k`
5. [https://github.com/ouyanghuiyu/chineseocr_litegithub.com](https://link.zhihu.com/?target=https%3A//github.com/ouyanghuiyu/chineseocr_lite)：`star:6.4k`，支持windows
   1. [机器之心：实测超轻量中文OCR开源项目，总模型仅17Mzhuanlan.zhihu.com)](https://zhuanlan.zhihu.com/p/111533615)
6. 包括AlexNet、RCNN、ResNet、YOLO、SSD等。
7. [文本检测的资源汇总](https://github.com/hwalsuklee/awesome-deep-text-detection-recognition)
   1. [Github：深度学习文本检测识别（OCR）精选资源汇总](https://zhuanlan.zhihu.com/p/71028209)
8. [xiaofengshi：chinese-ocr](https://github.com/xiaofengShi/CHINESE-OCR)：`star:2.4k`
