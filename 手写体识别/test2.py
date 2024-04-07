
import gzip  #解压缩模块
import os
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# LeNet是网络名称
class LeNet(nn.Module):
    # 定义网络需要的操作（卷积、池化、全连接）
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    # 定义的网络实现过程，前向传播的路径
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1,784)
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def dataloader(train_X, train_Y, test_X, test_Y):
    train_X = train_X / 255.
    test_X = test_X / 255.
    # train_X 从 ndarry数据格式转换为 tensor数据格式: from_numpu
    dataset = Data.TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).long())
    trainLoader = Data.DataLoader(dataset, batch_size=4, shuffle=True)

    dataset = Data.TensorDataset(torch.from_numpy(test_X).float(), torch.from_numpy(test_Y).long())
    testLoader = Data.DataLoader(dataset, batch_size=4, shuffle=True)
    return trainLoader, testLoader

# trainLoader, testLoader = dataloader(train_X, train_Y, test_X, test_Y)
train_X = load_mnist_images('train-images-idx3-ubyte.gz')
train_X = train_X.reshape((60000, 1, 28, 28)) # CNN 需要的数据组织形式 （样本个数，通道数，图像行数，图像列数）
train_Y = load_mnist_labels('train-labels-idx1-ubyte.gz')

## Load the testing set
test_X = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_X = test_X.reshape((10000, 1, 28, 28))
test_Y = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

trainLoader, testLoader = dataloader(train_X, train_Y, test_X, test_Y)
net = LeNet()


criterion = nn.CrossEntropyLoss() #使用交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #使用SGD梯度下降

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
if use_gpu:
    net.cuda() #是否启用GPU加速

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        #获取输入和标签
        inputs, labels = data

        #转成Variable类型，否则无法进行梯度计算
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        #参数梯度清零
        optimizer.zero_grad()
        #前向传播，损失计算，后向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        #参数更新
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#[1,  2000] loss: 1.323
#[1,  4000] loss: 0.261
#[1,  6000] loss: 0.167
#[1,  8000] loss: 0.130
#[1, 10000] loss: 0.117
#[1, 12000] loss: 0.101
#[1, 14000] loss: 0.092
#[2,  2000] loss: 0.081
#[2,  4000] loss: 0.066
#[2,  6000] loss: 0.071
#[2,  8000] loss: 0.062
#[2, 10000] loss: 0.061
#[2, 12000] loss: 0.059
#[2, 14000] loss: 0.052
#Finished Training

correct = 0
total = 0
for data in testLoader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item() # use .item() to get the number from the device

print('Accuracy of the network on test images: %d %%' % (
    100 * correct / total))

#Accuracy of the network on  test images: 98 %
