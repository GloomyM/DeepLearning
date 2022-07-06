from torchvision import datasets
import torch.utils.data
import torchvision
import torch.nn as nn
from torchvision import transforms
from Lenet5 import Lenet5
import numpy as np
import matplotlib.pyplot as plt
import os


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    download = False
    if 'res' not in os.listdir():
        download = True
    train_set = datasets.MNIST(root='./res/data', transform=transforms.ToTensor(), train=True, download=download)
    test_set = datasets.MNIST(root='./res/data', transform=transforms.ToTensor(), train=False, download=download)
    train_set = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    test_set = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=True)
    imshow(torchvision.utils.make_grid(next(iter(train_set))[0]))
    device = torch.device("mps")  # mac选择gpu加速
    lenet5 = Lenet5().to(device)
    loss_function = nn.CrossEntropyLoss()  # 定义损失函数（交叉熵）
    optimizer = torch.optim.Adam(lenet5.parameters(), lr=0.01)  # 使用adam进行反向传播
    epochs = 5
    for epoch in range(epochs):

        # 训练训练集
        loss = 0
        for i, data in enumerate(train_set):
            inputs, labels = data  # 获取参数和对应的标签
            inputs, labels = inputs.to(device), labels.to(device)  # gpu
            optimizer.zero_grad()  # 重置优化器为0

            output = lenet5(inputs)  # 计算预测值
            loss_per = loss_function(output, labels)  # 计算损失值
            loss_per.backward()  # 反向传播
            optimizer.step()

            loss += loss_per.item()  # 计算每次的损失和

            if i % 100 == 99:
                print('[Epoch:%d, batch:%d] train loss: %.03f' % (epoch + 1, i + 1, loss / 100))
                loss = 0.0

        # 对测试集进行测试
        total, right = 0, 0
        for data in test_set:
            test_inputs, test_labels = data
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            output_test = lenet5(test_inputs)
            output_predict = torch.max(output_test.data, 1)[1]
            total += test_labels.size(0)
            temp = (output_predict == test_labels).sum()
            right += temp
        print("第{}轮的准确率为:{:.2f}%".format(epoch + 1, 100.0 * right.item() / total))
