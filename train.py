import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import matplotlib.pyplot as plt
from utils import train



# 定义数据预处理
train_tf = transforms.Compose([
	transforms.Resize(256),
	transforms.RandomHorizontalFlip(),
	transforms.CenterCrop(224),
	transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
valid_tf = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 使用ImageFolder定义数据集
train_set = ImageFolder('./train/', train_tf)
valid_set = ImageFolder('./val/', valid_tf)
# 使用DataLoader定义迭代器
train_data = DataLoader(train_set, 8, True, num_workers=4)
valid_data = DataLoader(valid_set, 16, False, num_workers=4)

# 使用预训练模型
net = models.resnet50(pretrained=True)
# 将最后一层全连接改成2分类
net.fc = nn.Linear(2048, 2)
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)

# 开始训练
train(net, train_data, valid_data, 40, optimizer, criterion)















