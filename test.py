from torchvision import models
from torchvision import transforms
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader as DataLoader
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
    
dataset_dir = './test/'                                                    # 数据集路径
model_file = './model/model_10.pth'                # 模型保存路径
N = 8                                                                                    #测试图片个数
workers = 10                                                                    # PyTorch读取数据线程数量
batch_size = 20                                                              # batch_size大小

# 数据转换
valid_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def test():

    # setting model
    model = torch.load(model_file)
    model.cuda()                                        # 送入GPU，利用GPU计算
    model = nn.DataParallel(model)
    model.eval()                                        # 设定为评估模式，即计算过程中不要dropout

    # get data
    files = random.sample(os.listdir(dataset_dir), N)   # 随机获取N个测试图像
    imgs = []           # img
    imgs_data = []      # img data
    for file in files:
        img = Image.open(dataset_dir + file)            # 打开图像
        img_data = valid_tf(img)           # 转换成torch tensor数据

        imgs.append(img)                                # 图像list
        imgs_data.append(img_data)                      # tensor list
    imgs_data = torch.stack(imgs_data)                  # tensor list合成一个4D tensor

    # calculation
    out = model(imgs_data)                              # 对每个图像进行网络计算
    
    out = F.softmax(out, dim=1)                         # 输出概率化
    out = out.data.cpu().numpy()                        # 转成numpy数据
    # pring results         显示结果
    for idx in range(N):
        plt.figure()
        if out[idx, 0] > out[idx, 1]:
            plt.suptitle('ant:{:.1%},bee:{:.1%}'.format(out[idx, 0], out[idx, 1]))
        else:
            plt.suptitle('bee:{:.1%},ant:{:.1%}'.format(out[idx, 1], out[idx, 0]))
        plt.imshow(imgs[idx])
    plt.show()

if __name__ == '__main__':
    test()