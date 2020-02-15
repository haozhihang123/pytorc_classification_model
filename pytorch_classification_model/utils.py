#coding=utf-8
from mpl_toolkits.axes_grid1 import host_subplot
from datetime import datetime
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    model_cp = './model/'# 网络参数保存位置
    train_val_num = []
    train_acc_plot = []
    val_acc_plot = []
    f = open("log.txt", 'a+')
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        im = Variable(im.cuda())
                        label = Variable(label.cuda())
                    else:
                        im = Variable(im)
                        label = Variable(label)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        train_val_num.append(epoch)
        train_acc_plot.append( float( '%.4f'%(train_acc / len(train_data))))
        val_acc_plot.append(float( '%.4f'%(valid_acc / len(valid_data))))
        print(epoch_str + time_str)
        f.write(epoch_str + time_str + '\n')
        if (epoch+1) % 11 == 0:
            torch.save(net, '{0}/model_{1}.pth'.format(model_cp,epoch))            # 训练所有数据后，保存网络的参数
            print("save {0}/model_{1}.pth successfully!".format(model_cp,epoch))
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)
    par1 = host.twinx()
    # set labels
    host.set_xlabel("train_val_num")
    host.set_ylabel("train_acc")
    par1.set_ylabel("val_acc")
    
    # plot curves
    p1, = host.plot(train_val_num, train_acc_plot, label="train_acc")
    p2, = par1.plot(train_val_num, val_acc_plot, label="val_acc")
    host.legend(loc=1)
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    plt.title('Train_acc & val_acc')
    plt.draw()
    plt.savefig('./train_acc_val_acc.png')
    plt.show()

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(
        in_channel, out_channel, 3, stride=stride, padding=1, bias=False)


class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)

        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x + out, True)


class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)

        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2), residual_block(64, 64), residual_block(64, 64))

        self.block3 = nn.Sequential(
            residual_block(64, 128, False), residual_block(128, 128))

        self.block4 = nn.Sequential(
            residual_block(128, 256, False), residual_block(256, 256))

        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512), nn.AvgPool2d(3))

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
