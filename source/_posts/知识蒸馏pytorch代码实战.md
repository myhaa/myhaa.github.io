---
title: 代码实战之知识蒸馏
author: Myhaa
top: false
cover: false
toc: true
mathjax: false
categories: 代码实战
tags:
  - 代码实战
  - 知识蒸馏
  - 深度学习
date: 2022-03-09 10:16:11
img:
coverImg:
password:
summary: 知识蒸馏pytorch代码实战
---

# 知识蒸馏pytorch代码实战

## 本文概述

1. 使用pytorch在MNIST数据集上，从头训练教师网络、学生网络、知识蒸馏训练学生网络，比较性能
2. 本文参考[同济子豪兄b站视频](https://www.bilibili.com/video/BV1zP4y1F7g4/?spm_id_from=333.788)

## 导入module


```python
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
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
lr = 1e-4  # learning rate
epochs = 6

temp = 10  # 蒸馏温度，根据经验设置，越大soft label越平
alpha = 0.3  # hard loss权重
```


```python
device
```




    device(type='cuda')



## 加载数据集


```python
train_dataset = torchvision.datasets.MNIST(root='./datasets/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
```

    D:\meiyunhe\softwares\Miniconda3\envs\env_pytorch\lib\site-packages\torchvision\datasets\mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ..\torch\csrc\utils\tensor_numpy.cpp:180.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)



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

## 教师模型

### 构建教师模型


```python
class TeacherModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(TeacherModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        return x
```


```python
teacher_model = TeacherModel()
teacher_model = teacher_model.to(device)

summary(teacher_model)
```




    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    TeacherModel                             --
    ├─ReLU: 1-1                              --
    ├─Linear: 1-2                            942,000
    ├─Linear: 1-3                            1,441,200
    ├─Linear: 1-4                            12,010
    ├─Dropout: 1-5                           --
    =================================================================
    Total params: 2,395,210
    Trainable params: 2,395,210
    Non-trainable params: 0
    =================================================================



### 训练教师模型


```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(teacher_model.parameters(), lr=lr)
```


```python
for epoch in range(epochs):
    teacher_model.train()
    
    # 训练集上训练权重
    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        # 前向推断
        preds = teacher_model(data)
        loss = criterion(preds, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 测试集上评估模型
    teacher_model.eval()
    num_correct = 0
    num_samples = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            preds = teacher_model(data)
            predictions = preds.max(1).indices
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        acc = (num_correct / num_samples).item()
        
    teacher_model.train()
    print('epoch: {} \t acc: {:.5f}'.format(epoch+1, acc))
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1875/1875 [00:12<00:00, 150.31it/s]


    epoch: 1 	 acc: 0.94270


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1875/1875 [00:11<00:00, 161.99it/s]


    epoch: 2 	 acc: 0.96180


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1875/1875 [00:11<00:00, 159.42it/s]


    epoch: 3 	 acc: 0.97000


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1875/1875 [00:11<00:00, 162.90it/s]


    epoch: 4 	 acc: 0.97400


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1875/1875 [00:12<00:00, 156.18it/s]


    epoch: 5 	 acc: 0.97800


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1875/1875 [00:11<00:00, 162.00it/s]


    epoch: 6 	 acc: 0.97810


## 学生模型

### 构建学生模型


```python
class StudentModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(StudentModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, num_classes)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        return x
```


```python
student_model = StudentModel()
student_model = student_model.to(device)

summary(student_model)
```




    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    StudentModel                             --
    ├─ReLU: 1-1                              --
    ├─Linear: 1-2                            15,700
    ├─Linear: 1-3                            420
    ├─Linear: 1-4                            210
    =================================================================
    Total params: 16,330
    Trainable params: 16,330
    Non-trainable params: 0
    =================================================================



### 训练学生模型


```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
```


```python
epochs = 3
for epoch in range(epochs):
    student_model.train()
    
    # 训练集上训练权重
    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        # 前向推断
        preds = student_model(data)
        loss = criterion(preds, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 测试集上评估模型
    student_model.eval()
    num_correct = 0
    num_samples = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            preds = student_model(data)
            predictions = preds.max(1).indices
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        acc = (num_correct / num_samples).item()
        
    student_model.train()
    print('epoch: {} \t acc: {:.5f}'.format(epoch+1, acc))
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1875/1875 [00:11<00:00, 159.62it/s]


    epoch: 1 	 acc: 0.83810


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1875/1875 [00:11<00:00, 157.98it/s]


    epoch: 2 	 acc: 0.88360


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1875/1875 [00:13<00:00, 135.53it/s]


    epoch: 3 	 acc: 0.89970


## 知识蒸馏训练学生模型


```python
# 教师模型开启预测模式
teacher_model.eval()
```




    TeacherModel(
      (relu): ReLU()
      (fc1): Linear(in_features=784, out_features=1200, bias=True)
      (fc2): Linear(in_features=1200, out_features=1200, bias=True)
      (fc3): Linear(in_features=1200, out_features=10, bias=True)
      (dropout): Dropout(p=0.5, inplace=False)
    )




```python
# 训练学生模型参数
student_model = StudentModel()
student_model = student_model.to(device)
# student_model.train()
```


```python
# hard loss
hard_loss = nn.CrossEntropyLoss()

# soft loss
soft_loss = nn.KLDivLoss(reduction='batchmean')  # KL散度

optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
```


```python
epochs = 3

for epoch in range(epochs):
    student_model.train()
    
    # 训练集上训练权重
    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        # 教师模型前向推断
        with torch.no_grad():
            teacher_preds = teacher_model(data)
        
        # 学生模型前向推断
        student_preds = student_model(data)
        student_loss = hard_loss(student_preds, targets)
        
        # 计算蒸馏后的预测结果及soft_loss
        ditillation_loss = soft_loss(F.softmax(student_preds / temp, dim=1),
                                     F.softmax(teacher_preds / temp, dim=1))
        
        # 将hard_loss和soft_loss加权求和
        loss = alpha * student_loss + (1 - alpha) * ditillation_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 测试集上评估模型
    student_model.eval()
    num_correct = 0
    num_samples = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            preds = student_model(data)
            predictions = preds.max(1).indices
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        acc = (num_correct / num_samples).item()
        
    student_model.train()
    print('epoch: {} \t acc: {:.5f}'.format(epoch+1, acc))
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1875/1875 [00:13<00:00, 138.17it/s]


    epoch: 1 	 acc: 0.84210


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1875/1875 [00:13<00:00, 141.64it/s]


    epoch: 2 	 acc: 0.88100


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1875/1875 [00:12<00:00, 145.44it/s]


    epoch: 3 	 acc: 0.89420



```python

```


```python

```
