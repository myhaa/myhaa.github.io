---
title: 实用工具之python
date: 2019-10-15 15:21:11
author: Myhaa
img:
top: false
cover: false
coverImg:
password:
toc: true
mathjax: false
summary: 有关Python的笔记
categories: 实用工具
tags:
  - python
  - 编程语言
---



![image-20210621144956956](%E5%AE%9E%E7%94%A8%E5%B7%A5%E5%85%B7%E4%B9%8Bpython/image-20210621144956956.png)

# 一、Python基础



# 二、Python进阶



# 三、参考书籍



# 四、疑难解答

## 1、编码问题

### `Python2`编码问题

* [参考](<https://foofish.net/why-Python-encoding-is-tricky.html>)

### `Python3`编码问题

* 在`Python3`版本中，把`'xxx'`和`u'xxx'`统一成`Unicode`编码，即写不写前缀`u`都是一样的。
* 在`Python3`版本中，所有的字符串都是使用`Unicode`编码的字符串序列。
* [参考](<https://foofish.net/how-Python3-handle-charset-encoding.html>)

## 2、日期操作

* [参考菜鸟教程](<https://www.runoob.com/python/python-date-time.html>)

### 产生一段时间的日期

```python
from datetime import datetime, date, timedelta
import pandas as pd
date_id_list = [datetime.strftime(x, '%Y%m%d') for x in list(pd.date_range(start='20190701', end='20190928'))]
```

### 获取指定日期的前-后N天

```python
import datetime
n = 1
tomorrow = datetime.datetime(2015, 10, 28) + datetime.timedelta(days=1)	# 2015-10-29 00:00:00
tomorrow_format = tomorrow.strftime('%Y%m%d')	# '20151029'
```



## 3、mrjob运行参数详情

```shell
python xxx.py -r hadoop --local-tmp-dir 'xxx' --hadoop-tmp-dir 'hdfs:xxx' --file 'xxx.txt' --jobconf mapred.map.tasks=20 --jobconf mapred.reduce.tasks=2 input.txt -o output_dir
```

**参考官方文档：**[mrjob](<https://mrjob.readthedocs.io/en/latest/>)

## 4、list中排列组合

```python
from itertools import combinations

combine_2 = list(combinations([1,2,3,4], 2))
```

## 5、文件操作

### 获取指定目录下指定文件

```python
import os

L = []
for root, dirs, files in os.walk(os.getcwd()):
    for x in files:
        if os.path.splitext(x)[1] == '.txt':
            L.append(os.path.join(root, x))
file_path = L[0]
```

### pandas读取excel文件

```python
roc_data = pd.read_excel(file_path, sheet_name='20200323_10000')
roc_data
```



## 6、命令行参数`sys.argv[1:]`解析

* [python类库31--命令行解析](https://www.cnblogs.com/itech/archive/2010/12/31/1919017.html)

### 手动解析

### getopt解析

### optionparser解析【推荐】

## 7、字典排序

```python
result = {}
sorted(result.items(), key=lambda x: x[1], reverse=True)
```

## 8、python自动登录Linux等服务

[参考](https://pexpect.readthedocs.io/en/stable/overview.html)

```python
import pexpect
child = pexpect.spawn('ftp ftp.openbsd.org')
child.expect('Name .*: ')
child.sendline('anonymous')
child.expect('Password:')
child.sendline('noah@example.com')
child.expect('ftp> ')
child.sendline('lcd /tmp')
child.expect('ftp> ')
child.sendline('cd pub/OpenBSD')
child.expect('ftp> ')
child.sendline('get README')
child.expect('ftp> ')
child.sendline('bye')
```

## 9、 生成requirements.txt

### 第一种方法：太多太杂，对整个虚拟环境的

```python
# 生成
pip freeze > requirements.txt

# 安装
pip install -r requirements.txt
```

### 第二种方法：需要pip安装模块，但是可以对指定目录进行生成

```
# pip 安装模块
pip3 install pipreqs

# 对指定目录进行生成requirements.txt
cd 到指定目录
pipreqs ./ --encoding=utf8
# 这样在指定目录就会有requirements.txt的依赖文件
```



## 10、python取mysql中文乱码

1. mysql代码中将中文字段用`hex`函数转换
2. python代码中用`bytes.fromhex(取出的字段).decode('utf-8')`来转换

## 11、更新所有模块

* [参考](https://pypi.org/project/pip-review/)

```python
pip3 install pip-review
pip-review --interactive
```

## 12、多进程与多线程

* [参考](https://github.com/jackfrued/Python-100-Days/blob/master/Day01-15/13.%E8%BF%9B%E7%A8%8B%E5%92%8C%E7%BA%BF%E7%A8%8B.md)
* 例子1：多进程

```python
    def processTask(self, role_data, index):
        print('启动计算子进程，进程号[%d].' % os.getpid())
        print(role_data.shape)

        # post请求参数
        url = ''
        headers = {
            'game_code': '',  # 游戏代号
            'secret': '',  # secret key，线下提供
            'Content-Type': 'application/json'
        }

        # 推荐结果写入结果文件
        outputPath = 'g_role_item_test_result_multi_%s.txt' % str(index)
        with open(outputPath, mode='w', encoding='utf-8') as f:
            role_data.reset_index(drop=True, inplace=True)
            for i in range(role_data.shape[0]):
                role_id = role_data.role_id[i]
                gender = role_data.gender[i]
                purchased_items = role_data.item_set[i].replace('[', '').replace(']', '').replace('"', '').split(',')
                current_amount = role_data.left_yuanbao[i]
                argsData = {
                    "role_id": role_id,  # 角色id
                    "gender": int(gender) + 1,  # 角色性别：1-男，2-女
                    "purchased_items": purchased_items,  # 角色已购买的道具id列表, e.g. ["123", "321"],
                    "current_amount": int(current_amount)  # 角色当前账户剩余元宝
                }
                # print(i, argsData)
                t1 = time.time()
                try:
                    result = self.getRecommendResultInit(url, headers, argsData)
                except:
                    result = []
                t_diff = time.time() - t1

                # 输出结果
                result1 = {}
                for i, x in enumerate(result):
                    result1['R' + str(i)] = x
                output1 = [role_id, json.dumps(argsData), str(round(t_diff, 5))] + [json.dumps(result1)]
                f.write('\t'.join(output1) + '\n')

    def getRecommendResultMulti(self):
        """
        给定数据，从接口获取推荐结果，多进程
        :return:
        """
        # data参数获取
        inputPath = "g_role_item_test_data.csv"
        role_data = pd.read_csv(inputPath, sep=',', encoding='utf-8')
        role_data.fillna('[]', inplace=True)
        print(role_data.shape)

        # 多进程
        print('启动计算母进程，进程号[%d].' % os.getpid())
        index = 0
        processes = []
        for _ in range(4):
            p = Process(target=self.processTask,
                        args=(role_data.loc[index:index+70000], index))
            index += 70000
            processes.append(p)
            p.start()
            # 开始记录所有进程执行完成花费的时间
        start = time.time()
        for p in processes:
            p.join()
        end = time.time()
        print('Execution time: ', (end - start), 's', sep='')
```

* 例子2：多线程

```python
import time
import tkinter
import tkinter.messagebox
from threading import Thread


def main():

    class DownloadTaskHandler(Thread):

        def run(self):
            time.sleep(10)
            tkinter.messagebox.showinfo('提示', '下载完成!')
            # 启用下载按钮
            button1.config(state=tkinter.NORMAL)

    def download():
        # 禁用下载按钮
        button1.config(state=tkinter.DISABLED)
        # 通过daemon参数将线程设置为守护线程(主程序退出就不再保留执行)
        # 在线程中处理耗时间的下载任务
        DownloadTaskHandler(daemon=True).start()

    def show_about():
        tkinter.messagebox.showinfo('关于', '作者: 骆昊(v1.0)')

    top = tkinter.Tk()
    top.title('单线程')
    top.geometry('200x150')
    top.wm_attributes('-topmost', 1)

    panel = tkinter.Frame(top)
    button1 = tkinter.Button(panel, text='下载', command=download)
    button1.pack(side='left')
    button2 = tkinter.Button(panel, text='关于', command=show_about)
    button2.pack(side='right')
    panel.pack(side='bottom')

    tkinter.mainloop()


if __name__ == '__main__':
    main()
```

## 13、文本相似度

1. $similarity=\frac{|A\cap B|}{|A\cup B|}$
2. 修正，对长度做出惩罚：$similarity=\frac{|A\cap B|}{|A\cup B|+\alpha \times abs{(len(A)-len(B)})}$

## 14、one hot 编码=multiple values

* [pandas-Series.str.get_dummiers](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.get_dummies.html#pandas.Series.str.get_dummies)

```python
pd.Series(['80001,800002', '150001,150002', '80001,80002,150001,150002']).str.get_dummies(sep=',')
```

* [sklearn-MultiLabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html#sklearn.preprocessing.MultiLabelBinarizer)
* [one-hot过后压缩矩阵（稀疏矩阵）](https://stackoverflow.com/questions/63544536/convert-pd-get-dummies-result-to-df-str-get-dummies)

## 15、pandas使用apply返回多列

* [参考](https://stackoverflow.com/questions/23586510/return-multiple-columns-from-pandas-apply)

## 16、[Python 3: os.walk() file paths UnicodeEncodeError: 'utf-8' codec can't encode: surrogates not allowed](https://stackoverflow.com/questions/27366479/python-3-os-walk-file-paths-unicodeencodeerror-utf-8-codec-cant-encode-s)

```python
for p,d,f in os.walk(b'.'):
    print(p.decode('utf-8', 'replace'))
```

