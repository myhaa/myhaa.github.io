---
title: ShuffleNetV2pytorch代码实战
author: Myhaa
top: false
cover: false
toc: true
mathjax: false
categories: 代码实战
tags:
  - 代码实战
  - ShuffleNet
  - 深度学习
date: 2022-05-06 10:16:11
img:
coverImg:
password:
summary: ShuffleNetV2pytorch代码实战
---

# ShuffleNetV2pytorch代码实战

## 本文概述

1. 使用pytorch在MNIST数据集上，实战ShuffleNetV2
2. 本文参考[同济子豪兄b站视频](https://www.bilibili.com/video/BV1d3411Y7Ms?spm_id_from=333.999.0.0); [ShuffleNet-Series](https://github.com/megvii-model/ShuffleNet-Series)

* [论文地址](https://arxiv.org/abs/1807.11164)

## ShuffleNetV1核心算法

### group pointwise convolution：分组1*1卷积

#### 常规卷积操作

* 对于一张`5*5*3`的输入
* 经过`3×3`卷积核的卷积层(假设输出`channel=4`，则卷积核`shape=3*3*3*4`), 最终输出`4`个`Feature Map`
* 如果有`same padding`则尺寸与输入层相同(`5*5`)，如果没有则尺寸变为`3*3`
* 如果有`stride`和`padding`，输出尺寸计算公式如下
$$ output=\frac{(input\_size - kernel\_size + 2*padding)} {stride} + 1 $$

#### depthwise convolution

* 一个卷积核负责一个通道，一个通道只被一个卷积核卷积
* 卷积核的数量与上一层的通道数相同
* 相比常规卷积的好处在于参数量少，模型可以做得更深

#### pointwise convolution

* `1*1`卷积核

#### group convolution

* 将通道数一分为`n`组, 每组单独使用卷积核进行卷积，互不干扰
* 如图
![分组卷积](../images/group_convolution.png)


* 卷积核的尺寸为`1×1×m`，`m`为上一层的通道数
* 相比常规卷积的好处在于参数量少，模型可以做得更深


### channel shuffle： 通道重排

* 为了防止近亲繁殖：也就是组与组之间老死不相往来而设定，如下图
![通道重排1](../images/channel_shuffle_1.png)
![通道重排2](../images/channel_shuffle_2.png)

### 网络结构

* 如下图

![ShuffleNet网络结构1](../images/shuffle_net_architecture_1.png)
![ShuffleNet网络结构2](../images/shuffle_net_architecture_2.png)

## ShuffleNetV2核心算法

### 四个问题

1. 乘-加浮点运算次数FLOPs仅反应卷积层，仅为间接指标
2. 不同硬件上的测试结果不同
3. 数据读写的内存MAC占用影响很大
4. Element-wise逐元素操作带来的开销不可忽略

### 四个准则

1. 输入输出通道数相同时，内存访问量MAC最小
2. 分组数过大的分组卷积会增加MAC
3. 碎片化操作对并行加速不友好
4. 逐元素操作带来的内存和消耗不可忽略

### 1\*1卷积的FLOPs和MAC

![1\*1卷积的FLOPs和MAC](../images/shuffle_net_v2_1_1_conv_flops_mac.png)

### ShuffleNetV2网络结构
![ShuffleNetV2网络结构](../images/shuffle_net_v2_architecture.png)

### 改进点

1. 去掉了分组卷积，降低MAC
2. 将channel shuffle提前成channel split（对半分）
3. 将add操作改为concat操作，避免逐元素操作


## 导入module


```python
import torch
import torchvision
from torchvision import transforms
from torchinfo import summary
from torch.utils.data import DataLoader
```


```python

```

## 参数设置


```python
# 设置随机数，便于复现
my_seed = 2030
torch.manual_seed(my_seed)

# gpu or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 是否加速卷积运算
torch.backends.cudnn.benchmark = False

batch_size = 32


models_dir_path = './models/ShuffleNetV2'
logs_dir_path = './logs/ShuffleNetV2'

device
```




    device(type='cpu')




```python

```

## 加载数据集


```python
train_dataset = torchvision.datasets.MNIST(root='./datasets/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
```


```python
test_dataset = torchvision.datasets.MNIST(root='./datasets/',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)
```


```python
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```


```python
for x, y in train_loader:
    print(x.shape, y.shape)
    print(x.size(), y.size())
    break
```

    torch.Size([32, 1, 28, 28]) torch.Size([32])
    torch.Size([32, 1, 28, 28]) torch.Size([32])


## utils.py

* LogSoftmax公式

$$
\text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)
$$


```python
import os
import re
import torch
import torch.nn as nn
```


```python
class CrossEntropyLabelSmooth(nn.Module):
    """
    标签平滑（Label Smoothing）是一个有效的正则化方法，可以在分类任务中提高模型的泛化能力。
    其思想相当简单，即在通常的Softmax-CrossEntropy中的OneHot编码上稍作修改，将非目标类概率值设置为一个小量，相应地在目标类上减去一个值，从而使得标签更加平滑
    """
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes  # 分类数量
        self.epsilon = epsilon  # 小量
        self.logsoftmax = nn.LogSoftmax(dim=1)  # log soft Max函数，按行soft Max，即每行元素soft Max总和为1
        
    def forward(self, inputs, targets):
        # inputs是yi^hat，即预测值，targets是真实值
        log_probs = self.logsoftmax(inputs)  
        # 和log_probs一样的size，以targets填充
        # PyTorch 中，一般函数加下划线代表直接在原来的 Tensor 上修改
        targets = torch.zeros_like(log_probs).scatter_(dim=1, index=targets.unsqueeze(1), value=1)  # 用value替换dim=1方向对应index位置的数  
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes  # 标签平滑
        loss = (-targets * log_probs).mean(0).sum()  # 按列求均值，再求和
        return loss
```


```python
# 测试
num_classes = 8
batch_size = 32

test_inputs = torch.randn(batch_size, num_classes)
test_targets = torch.randint(low=0, high=num_classes-1, size=[batch_size])
print(test_inputs.size(), test_targets.size())

label_smoth_class = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1)
label_smoth_loss = label_smoth_class(test_inputs, test_targets)
label_smoth_loss
```

    torch.Size([32, 8]) torch.Size([32])





    tensor(2.5863)




```python

```


```python
class AvgrageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
```


```python

```


```python
def accuracy(output, target, topk=(1,)):
    """
    计算topk正确率
    """
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)  # 每行取出topk的类别
    pred = pred.t()  # pred转至
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # view相当于reshape, expand_as
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res
```


```python
# 测试
num_classes = 8
batch_size = 32

test_inputs = torch.randn(batch_size, num_classes)
test_targets = torch.randint(low=0, high=num_classes-1, size=[batch_size])
print(test_inputs.size(), test_targets.size())

print(accuracy(test_inputs, test_targets))
```

    torch.Size([32, 8]) torch.Size([32])
    [tensor(6.2500)]



```python

```


```python
def save_checkpoint(models_dir_path, state, iters, tag=''):
    """
    保存checkpoint
    :param models_dir_path: 模型保存路径
    :param state: 模型
    :param iters: 第几步
    :param tag: 标签名
    """
    if not os.path.exists(models_dir_path):
        os.makedirs(models_dir_path)
    filename = os.path.join('{}/{}checkpoints-{:06}.pth.tar'.format(models_dir_path, tag, iters))
    torch.save(state, filename)
```


```python

```


```python
def get_lastest_model(models_dir_path):
    """
    获取最新模型的路径+文件名
    :param models_dir_path: 模型保存路径
    """
    if not os.path.exists(models_dir_path):
        os.makedirs(models_dir_path)
    models_list = os.listdir(models_dir_path)
    if models_list == []:
        return None, 0
    models_list.sort() 
    lastest_model = models_list[-1]
    iters = re.findall(r'\d+', lastest_model)
    return os.path.join(models_dir_path, lastest_model), int(iters[0])
```


```python

```


```python
def get_parameters(model):
    """
    获取模型参数值
    :param model: 模型实例
    """
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            group_weight_decay.append(p)
        else:
            group_no_weight_decay.append(p)
    
    assert len(list(model.parameters())) == len(group_no_weight_decay) + len(group_weight_decay)
    groups = [dict(params=group_weight_decay), dict(params=group_no_weight_decay, weight_decay=0.)]
    return groups
```


```python

```

## blocks.py

* Conv2d

In the simplest case, the output value of the layer with input size
$(N, C_{\text{in}}, H, W)$ and output $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$
can be precisely described as:

$$
    \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
    \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
$$


where $\star$ is the valid 2D `cross-correlation`_ operator,
$N$ is a batch size, $C$ denotes a number of channels,
$H$ is a height of input planes in pixels, and $W$ is
width in pixels.

* BatchNorm2d

$$
    y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$

The mean and standard-deviation are calculated per-dimension over
the mini-batches and $\gamma$ and $\beta$ are learnable parameter vectors
of size $C$ (where $C$ is the input size). By default, the elements of $\gamma$ are set
to 1 and the elements of $\beta$ are set to 0. The standard-deviation is calculated
via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

* AvgPool2d

In the simplest case, the output value of the layer with input size $(N, C, H, W)$,
output $N, C, H_{out}, W_{out})$ and `kernel_size` $(kH, kW)$
can be precisely described as:

$$
    out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                           input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)
$$

If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
for :attr:`padding` number of points.


```python
import torch
import torch.nn as nn
```


```python
class ShuffleV2Block(nn.Module):
    # * 后面是限制命名关键字参数，**kw是命名关键字参数，可以传任意多参数
    # 相比ShuffleNetV1，它去掉了分组卷积
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        self.mid_channels = mid_channels
        self.ksize = ksize  # kernel size
        pad = ksize // 2  # padding
        self.pad = pad
        self.inp = inp
        
        outputs = oup - inp
            
        branch_main = [
            # ponitwise  1*1分组卷积降维
            nn.Conv2d(in_channels=inp, 
                      out_channels=mid_channels, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # depthwise  3*3 depth卷积
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            
            # pointwise-linear  1*1分组卷积升维
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True)
        ]
        
        self.branch_main = nn.Sequential(*branch_main)
        
        if stride == 2:
            # 下采样concat拼接
            branch_proj = [
                # depthwise  3*3 depth卷积
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),

                # pointwise-linear  1*1分组卷积升维
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True)
            ]
            
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None
        
    def channel_shuffle(self, x):
        batch_size, num_channels, height, width = x.data.size()
        assert num_channels % 4 == 0
        
        x = x.reshape(batch_size * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)  # 转置
        x = x.reshape(2, -1, num_channels // 2, height, width)
        
        return x[0], x[1]
    
    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)
```


```python

```

## network.py



```python
import torch
import torch.nn as nn
# from blocks import ShuffleV2Block
```


```python
class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=224, n_class=1000, model_size='1.5x'):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)
        
        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError
        
        # building first layer
        input_channel = self.stage_out_channels[1]  # 第一层输出channel
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=input_channel, 
                      kernel_size=3, 
                      stride=2, 
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # stage阶段
        self.features = []
        for idx_stage in range(len(self.stage_repeats)):
            num_repeat = self.stage_repeats[idx_stage]
            output_channel = self.stage_out_channels[idx_stage+2]
            
            for i in range(num_repeat):
                if i == 0:
                    self.features.append(
                        ShuffleV2Block(input_channel, output_channel, 
                                       mid_channels=output_channel // 2,
                                       ksize=3, stride=2))
                else:
                    self.features.append(
                        ShuffleV2Block(input_channel // 2, output_channel, 
                                       mid_channels=output_channel // 2,
                                       ksize=3, stride=1))
                input_channel = output_channel
        
        self.features = nn.Sequential(*self.features)
        
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        self.globalpool = nn.AvgPool2d(kernel_size=7)
        
        if self.model_size == '2.0x':
            self.dropout = nn.Dropout(0.2)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False)
        )
        self._initialize_weights()
        
    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.last_conv(x)
        
        x = self.globalpool(x)
        if self.model_size == '2.0x':
            x = self.dropout(x)
        
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
```


```python
model = ShuffleNetV2()
summary(model)
```

    model size is  1.5x





    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    ShuffleNetV2                             --
    ├─Sequential: 1-1                        --
    │    └─Conv2d: 2-1                       648
    │    └─BatchNorm2d: 2-2                  48
    │    └─ReLU: 2-3                         --
    ├─MaxPool2d: 1-2                         --
    ├─Sequential: 1-3                        --
    │    └─ShuffleV2Block: 2-4               --
    │    │    └─Sequential: 3-1              16,936
    │    │    └─Sequential: 3-2              888
    │    └─ShuffleV2Block: 2-5               --
    │    │    └─Sequential: 3-3              16,808
    │    └─ShuffleV2Block: 2-6               --
    │    │    └─Sequential: 3-4              16,808
    │    └─ShuffleV2Block: 2-7               --
    │    │    └─Sequential: 3-5              16,808
    │    └─ShuffleV2Block: 2-8               --
    │    │    └─Sequential: 3-6              64,592
    │    │    └─Sequential: 3-7              33,264
    │    └─ShuffleV2Block: 2-9               --
    │    │    └─Sequential: 3-8              64,592
    │    └─ShuffleV2Block: 2-10              --
    │    │    └─Sequential: 3-9              64,592
    │    └─ShuffleV2Block: 2-11              --
    │    │    └─Sequential: 3-10             64,592
    │    └─ShuffleV2Block: 2-12              --
    │    │    └─Sequential: 3-11             64,592
    │    └─ShuffleV2Block: 2-13              --
    │    │    └─Sequential: 3-12             64,592
    │    └─ShuffleV2Block: 2-14              --
    │    │    └─Sequential: 3-13             64,592
    │    └─ShuffleV2Block: 2-15              --
    │    │    └─Sequential: 3-14             64,592
    │    └─ShuffleV2Block: 2-16              --
    │    │    └─Sequential: 3-15             253,088
    │    │    └─Sequential: 3-16             128,480
    │    └─ShuffleV2Block: 2-17              --
    │    │    └─Sequential: 3-17             253,088
    │    └─ShuffleV2Block: 2-18              --
    │    │    └─Sequential: 3-18             253,088
    │    └─ShuffleV2Block: 2-19              --
    │    │    └─Sequential: 3-19             253,088
    ├─Sequential: 1-4                        --
    │    └─Conv2d: 2-20                      720,896
    │    └─BatchNorm2d: 2-21                 2,048
    │    └─ReLU: 2-22                        --
    ├─AvgPool2d: 1-5                         --
    ├─Sequential: 1-6                        --
    │    └─Linear: 2-23                      1,024,000
    =================================================================
    Total params: 3,506,720
    Trainable params: 3,506,720
    Non-trainable params: 0
    =================================================================




```python
test_data = torch.rand(5, 3, 224, 224)
test_data.size()
```




    torch.Size([5, 3, 224, 224])




```python
test_outputs = model(test_data)
```


```python
print(test_outputs.size())
```

    torch.Size([5, 1000])



```python
test_outputs
```




    tensor([[-0.1431,  0.0366, -0.0717,  ...,  0.1740,  0.0371,  0.0341],
            [-0.1007, -0.0392, -0.0115,  ...,  0.1421,  0.0067,  0.0377],
            [-0.0906, -0.0042, -0.0363,  ...,  0.1792,  0.0322,  0.0410],
            [-0.0787,  0.0394, -0.0279,  ...,  0.1573,  0.0437,  0.0645],
            [-0.1096,  0.0807, -0.0481,  ...,  0.1727,  0.0823,  0.0252]],
           grad_fn=<MmBackward0>)




```python

```

## train.py


```python
import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
# pip install opencv-python
import numpy as np
import PIL
from PIL import Image
import time
import logging
```


```python
class OpencvResize(object):
    """
    利用opencv进行图片resize
    """
    def __init__(self, size=256):
        self.size = size
    
    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img)  # (H, W, 3) RGB
        img = img[:, :, ::-1]  # BGR
        img = np.ascontiguousarray(img)  # Return a contiguous array (ndim >= 1) in memory (C order).
        H, W, _ = img.shape
        # 按长宽比进行缩放
        target_size = (int(self.size / H * W + 0.5), self.size) if H < W else (self.size, int(self.size / W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)  # 插值
        img = img[:, :, ::-1]  # RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img
```


```python
img_path = '../images/group_convolution.png'
img = Image.open(img_path)
print(img.size)
img
```

    (956, 555)






![png](output_45_1.png)
    




```python
img_resize = OpencvResize(256)(img)
print(img_resize.size)
img_resize
```

    (441, 256)






![png](output_46_1.png)
    




```python
class ToBGRTensor(object):
    """
    转为BGRTensor
    """
    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:, :, ::-1]  # 2 BGR
        img = np.transpose(img, [2, 0, 1])  # 2 [3, H, W]
        img = np.ascontiguousarray(img)  # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        img = torch.from_numpy(img).float()
        return img
```


```python
img_tensor = ToBGRTensor()(img_resize)
img_tensor.shape
```




    torch.Size([4, 256, 441])




```python
class DataIterator(object):
    """
    数据导入类
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)
    
    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]
```


```python

```


```python
def get_args():
    """
    获取运行参数
    """
    parser = argparse.ArgumentParser('ShuffleNetV2_Plus')
    
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str, default='./models/snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('--total-iters', type=int, default=300000, help='total iters')
    parser.add_argument('--learning-rate', type=float, default=0.5, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')
    
    parser.add_argument('--auto-continue', type=bool, default=True, help='auto continue')
    parser.add_argument('--display-interval', type=int, default=20, help='display interval')
    parser.add_argument('--val-interval', type=int, default=10000, help='val interval')
    parser.add_argument('--save-interval', type=int, default=10000, help='save interval')
    
    parser.add_argument('--model-size', type=str, default='2.0x', choices=['0.5x', '1.0x', '1.5x', '2.0x'], help='size of the model')
    
    parser.add_argument('--train-dir', type=str, default='./datasets/train', help='path to training dataset')
    parser.add_argument('--val-dir', type=int, default=20, help='path to validation dataset')
    
    args = parser.parse_args()
    return args
```


```python
# args = get_args()
# args
```


```python
def load_checkpoint(net, checkpoint):
    """
    导入模型
    """
    from collections import OrderedDict
    
    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    for k in checkpoint:
        k2 = 'module.' + k if not k.startswith('module.') else k
        temp[k2] = checkpoint[k]
    
    net.load_state_dict(temp, strict=True)
```


```python
def validate(model, device, args, *, all_iters=None):
    """
    利用现有模型对验证集数据进行推断评估
    """
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    
    loss_function = args.loss_function
    val_dataprovider = args.val_dataprovider
    
    model.eval()
    max_val_iters = 250
    
    t1 = time.time()
    with torch.no_grad():
        for _ in range(1, max_val_iters+1):
            data, target = val_dataprovider.next()
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = loss_function(output, target)
            
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
    
    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(1 - top1.avg / 100) + \
              'Top-5 err = {:.6f},\t'.format(1 - top5.avg / 100) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    logging.info(logInfo)
```


```python

```


```python
def adjust_bn_momentum(model, iters):
    """
    调整batch normalization的momentum
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters
```


```python

```


```python
def train(model, device, args, *, val_interval, bn_process=False, all_iters=None):
    """
    训练过程
    """
    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_dataprovider = args.train_dataprovider
    
    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    
    model.train()
    
    for iters in range(1, val_interval + 1):
        scheduler.step()
        if bn_process:
            adjust_bn_momentum(model, iters)
        
        all_iters += 1
        d_st = time.time()
        data, target = train_dataprovider.next()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data_time = time.time() - d_st
        
        output = model(data)
        
        loss = loss_function(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        
        Top1_err += 1 - prec1.item() / 100
        Top5_err += 1 - prec5.item() / 100
        
        if all_iters % args.display_interval == 0:
            printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_lr()[0], loss.item()) + \
                        'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                        'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
                        'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / args.display_interval)
            logging.info(printInfo)
            t1 = time.time()
            Top1_err, Top5_err = 0.0, 0.0
        
        if all_iters % args.save_interval == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
            }, all_iters)
    return all_iters
```


```python

```


```python
def main():
    args = get_args()
    
    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists(logs_dir_path):
        os.makedirs(logs_dir_path)
    fh = logging.FileHandler(os.path.join('{}/train-{}{:02}{}'.format(logs_dir_path, local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    
    # 导入数据
    assert os.path.exists(args.train_dir)
    train_dataset = datasets.ImageFolder(
        args.train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(0.5),
            ToBGRTensor(),
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=use_gpu
    )
    train_dataprovider = DataIterator(train_loader)
    
    assert os.path.exists(args.val_dir)
    val_dataset = datasets.ImageFolder(
        args.val_dir, 
        transforms.Compose([
            OpencvResize(256), 
            transforms.CenterCrop(224),
            ToBGRTensor(),
        ])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=200, shuffle=False,
        num_workers=1, pin_memory=use_gpu
    )
    val_dataprovider = DataIterator(val_loader)
    print('load data successfully')
    
    # 构建模型
    model = ShuffleNetV2(model_size=args.model_size)
    
    # 构建优化器
    optimizer = torch.optim.SGD(get_parameters(model),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # 构建损失函数
    criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1)
    
    if use_gpu:
        model = nn.DataParallel(model)
        loss_function = criterion_smooth.cuda()
        device = torch.device('cuda')
    else:
        loss_function = criterion_smooth
        device = torch.device('cpu')
    
    # learning_rate随着训练步数调整
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lambda step: (1.0-step/args.total_iters) if step <= args.total_iters else 0,
        last_epoch=-1
    )
    
    model = model.to(device)
    
    # 是否往上次训练的地方开始继续训练
    all_iters = 0
    if args.auto_continue:
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(iters):
                scheduler.step()
                
    args.optimizer = optimizer
    args.loss_function = loss_function
    args.scheduler = scheduler
    args.train_dataprovider = train_dataprovider
    args.val_dataprovider = val_dataprovider
    
    # 如果是线上推断
    if args.eval:
        if args.eval_resume is not None:
            checkpoint = torch.load(args.eval_resume, map_location=None if use_gpu else 'cpu')
            load_checkpoint(model, checkpoint)
            validate(model, device, args, all_iters=all_iters)
        exit(0)
    
    # 如果没达到设定的训练步数，则需要继续训练
    while all_iters < args.total_iters:
        all_iters = train(model, device, args, val_interval=args.val_interval, bn_process=False, all_iters=all_iters)
        validate(model, device, args, all_iters=all_iters)
    all_iters = train(model, device, args, val_interval=int(1280000/args.batch_size), bn_process=True, all_iters=all_iters)
    validate(model, device, args, all_iters=all_iters)
    
    # 保存模型
    save_checkpoint({'state_dict': model.state_dict(), }, args.total_iters, tag='bnps-')
    torch.save(model.state_dict(), models_dir_path+'/model.mdl')
```


```python

```


```python

```
